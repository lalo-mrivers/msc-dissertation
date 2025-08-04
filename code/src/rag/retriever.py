import time
from typing import Final
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import abc

from transformers import AutoModel, AutoTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast

from rag.reranker import ColBERTReranker
from utils.vprint import vprint
import math
from rank_bm25 import BM25Okapi
import os
import pickle
import uuid

class QRetriever(abc.ABC):
    """
    Base retriever interface.
    """
    
    @abc.abstractmethod
    def index(self, documents, batch_size: int = 2048):
        """
        Index documents with the corresponding encoder.
        """
        pass

    @abc.abstractmethod
    def retrieve(self, query: str, k: int, *args, **kwargs) -> list[str]:
        """
        Retrieve documents based on a query.

        :param query: The user's text query.
        :param k: The number of documents to retrieve.
        :return: A list of strings, where each string is the text of a document.
        """
        pass



class BM25Retriever(QRetriever):
    """
    BM25 retirever implementation
    """
    def __init__(self, index_path: str = "bm25_index.pkl", corpus_path: str = "bm25_corpus.pkl", load_from_disk: bool = True, verbose: bool = True):
        """
        """
        self._verbose = verbose
        self._index_path = index_path
        self._corpus_path = corpus_path
        self.bm25 = None
        self.corpus = None

        if os.path.exists(self._index_path) and os.path.exists(self._corpus_path) and load_from_disk:
            vprint(self._verbose, f"--> [BM25 Retriever] Loading index and corpus from disk: '{self._index_path}' and '{self._corpus_path}'...")
            
            with open(self._index_path, 'rb') as f_index:
                self.bm25 = pickle.load(f_index)
            with open(self._corpus_path, 'rb') as f_corpus:
                self.corpus = pickle.load(f_corpus)
            
            vprint(self._verbose, "--> [BM25 Retriever] Index and corpus loaded successfully from disk.")
        else:
            vprint(self._verbose, "--> [BM25 Retriever] No index or corpus files found. You need to run the indexing script first.")
        
    def index(self, documents, batch_size: int = 2048):
        tmetrics = {}
        if self.bm25 is not None:
            vprint(self._verbose, "--> [BM25 Retriever] [BM25Retriever] Index already loaded.")
            return
        
        self.corpus = [doc.get('contents') for doc in documents]
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        
        vprint(self._verbose, "--> [BM25 Retriever] Creting BM25 index...")

        t_start_whole = time.time()
        t_start_misc = t_start_whole
        self.bm25 = BM25Okapi(tokenized_corpus)
        t_embedding = time.time() - t_start_misc
        
        vprint(self._verbose, "--> [BM25 Retriever] BM25 index ready.")

        vprint(self._verbose, f"--> [BM25 Retriever] Saving index ({self._index_path})...")
        t_start_misc = time.time()
        with open(self._index_path, 'wb') as f_index:
            pickle.dump(self.bm25, f_index)
        t_saving_index = time.time() - t_start_misc

        vprint(self._verbose, f"--> [BM25 Retriever] Saving corpus ({self._corpus_path})...")
        t_start_misc = time.time()
        with open(self._corpus_path, 'wb') as f_corpus:
            pickle.dump(self.corpus, f_corpus)
        t_saving_corpus = time.time() - t_start_misc

        t_total = time.time() - t_start_whole
        tmetrics = {'n_docs': len(documents),
                    'batch_size': batch_size,
                    't_total': t_total,
                    't_embedding': t_embedding,
                    't_saving_index': t_saving_index,
                    't_saving_corpus': t_saving_corpus,
                    }
        
        vprint(self._verbose, "--> [BM25 Retriever] Index process finished. ---")

        return tmetrics


    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """
        Tokenize a query and retrieves the top k relevant documents from the corpus.
        """
        if self.bm25 is None or self.corpus is None:
            raise RuntimeError("BM25 index has not been created yet.")
            
        tokenized_query = query.split(" ")
        top_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return top_docs

class DenseRetriever(QRetriever):
    """
    Dense retriever that uses Sentence Transformers and Qdrant for vector search.
    """
    @abc.abstractmethod
    def __init__(self):
        self._verbose:bool = False
        self._query_model_name:str = None
        self._passage_model_name:str = None
        self._collection_name:str = None
        self._device:str = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_retrieve_tmetrics:dict = None
        
        self.query_encoder = None
        self.passage_encoder = None

        self.distance_metric:models.Distance = None

        self._embedding_dimension:int = None

        self.db_client:QdrantClient = None

    def _setup_db_client(self, db_host: str, db_port: int):
        """
        Setup the Qdrant client.
        """
        if db_host and db_host.startswith(('http', 'https')):
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Using remote Qdrant at {db_host}...")
            self.db_client = QdrantClient(url=db_host, timeout=300.0)
        else:
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Using remote Qdrant at {db_host}:{db_port}...")
            self.db_client = QdrantClient(host=db_host, port=db_port, timeout=300.0)
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Qdrant client initialized.")

    def _setup_collection(self):
        """
        Ensure the collection exists in Qdrant. If not, create it.
        """
        try:
            self.db_client.get_collection(collection_name=self._collection_name)
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Collection '{self._collection_name}' already exists in db.")
            return
        except Exception as e:
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Error:{e}")
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Collection '{self._collection_name}' does not exist, creating it now...")
        
        print(f'self._embedding_dimension: {self._embedding_dimension}')
        self.db_client.recreate_collection(
            collection_name = self._collection_name,
            vectors_config = models.VectorParams(size = self._embedding_dimension,
                                                 distance = self.distance_metric 
            )
        )
        
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Collection '{self._collection_name}' created.")

    def _save_embedding(self, batch_ids:list[str], batch_embeddings, payload:list):
        # Saving batch embeddings to Qdrant
        retry_count = 5
        while retry_count >= 0:
            try:
                self.db_client.upsert(
                    collection_name=self._collection_name,
                    points=models.Batch(
                        ids=batch_ids, 
                        vectors=batch_embeddings.tolist(),
                        payloads=payload
                    ),
                    wait=True
                )
                retry_count = -1
                break
            except Exception as e:
                vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Retrying ({retry_count})")
                time.sleep(1)
            
            retry_count = retry_count - 1

    def _search(self, query_embedding, k:int, with_payload=True):
        """
        Search for the top k documents in the collection based on the query embedding.
        """
        retry = 5
        while retry >= 0:
            try:
                search_result = self.db_client.search(
                    collection_name=self._collection_name,
                    query_vector=query_embedding,
                    limit=k,
                    with_payload=with_payload
                )
                retry = -1
            except Exception as e:
                vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Error {e}. Retrying...")
                time.sleep(1)
                retry = retry - 1
                if retry < 0:
                    raise RuntimeError(f'Connection failed: {e}')
        return search_result

    def _generate_id(self, seed_str: str) -> str:
        """
        Generate a unique ID based on a seed string.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_str))
    
    @abc.abstractmethod
    def _get_passage_embedding(self, passage):
        #return self.passage_encoder.encode(passage, show_progress_bar=self._verbose, batch_size=128)
        pass

    @abc.abstractmethod
    def _get_query_embedding(self, query):
        #return self.query_encoder.encode(query)
        pass
    
    def _average_pooling(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        sum_hidden_states = torch.sum(last_hidden_states * attention_mask.unsqueeze(-1), dim=1)
        sum_attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)

        average_embedding = sum_hidden_states / torch.clamp(sum_attention_mask, min=1e-9)

        return average_embedding
    
    def index(self, documents, batch_size: int = 2048):
        """
        Index documents into the Qdrant collection.

        Args:
            :param documents: A dataset of documents to index. Each document should have a 'contents' key.
            :param batch_size: The number of documents to process in each batch (default is 2048).
        """
        tmetrics = {}
        # 1.- Verify if the collection exists and get the number of indexed points
        try:
            collection_info = self.db_client.get_collection(collection_name=self._collection_name)
            #n_indexes = collection_info.points_count
            n_datapoints = collection_info.points_count
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Collection contains {n_datapoints} datapoints.")

        except Exception:
            self._setup_collection()
            #n_indexes = 0

        # 3.- Process the documents in batches
        n_batches = math.ceil(len(documents) / batch_size)
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Documents to be indexed: {len(documents)}. Batch size: {batch_size}. Total batches: {n_batches}.")

        t_start_whole = time.time()
        t_sum_embedding = 0
        t_sum_saving = 0
        for i in range(n_batches):
            start_batch = i * batch_size
            end_batch = min((i + 1) * batch_size, len(documents))
            
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Processing batch #{i+1}/{n_batches} (Document {start_batch} tp {end_batch-1} [{documents[start_batch].get('id')}-{documents[end_batch-1].get('id')}])...")
            
            batch_docs_data = documents.select(range(start_batch, end_batch))
            batch_texts = [doc.get('contents') for doc in batch_docs_data]
            batch_hf_ids = [doc.get('id') for doc in batch_docs_data]
            batch_ids = [self._generate_id(doc.get('id')) for doc in batch_docs_data]
            
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] [#{i+1}/{n_batches}] Processing {len(batch_texts)} documents...")
            
            #***** Generate embeddings *****
            t_start_embedding = time.time()
            batch_embeddings = self._get_passage_embedding(batch_texts)
            t_sum_embedding += (time.time() - t_start_embedding) 
            #*******************************

            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] [#{i+1}/{n_batches}] Documents encoded. Saving embeddings ({self._collection_name})...")

            start_time = time.time()
            t_start_saving = time.time()
            try:
                # Saving batch embeddings to Qdrant
                self._save_embedding(batch_ids=batch_ids,
                                     batch_embeddings=batch_embeddings,
                                     payload=[{'text': text, 'id_hf': id_hf} for text, id_hf in zip(batch_texts, batch_hf_ids)])
                
                vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Batch #{i+1} indexed .")
                
            except Exception as e:
                vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Error indexing batch #{i+1}: {e}")
                vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Elapsed {time.time() - start_time:.4f} s.")
                raise e
            t_sum_saving += (time.time() - t_start_saving) 
            vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Elapsed {time.time() - start_time:.4f} s.")
            
        t_total = (time.time() - t_start_whole)
        t_avg_batch = t_total / (n_batches if n_batches != 0 else 1)
        t_avg_embedding = t_sum_embedding / (n_batches if n_batches != 0 else 1)
        t_avg_saving = t_sum_saving / (n_batches if n_batches != 0 else 1)
        tmetrics = {'n_batches': n_batches,
                    'n_docs': len(documents),
                    'batch_size': batch_size,
                    't_total': t_total,
                    't_avg_batch': t_avg_batch,
                    't_avg_embedding': t_avg_embedding,
                    't_avg_saving': t_avg_saving,
                    }
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Index process completed '{self._collection_name}'. Datapoints: {len(documents)}. ---")
        return tmetrics
    
    def get_last_retrieve_metrics(self):
        return self._last_retrieve_tmetrics


class BaseDenseRetriever(DenseRetriever):
    """
    Dense retriever that uses Sentence Transformers and Qdrant for vector search.
    """
    def __init__(self,
                 passage_model_name:str,
                 query_model_name: str,
                 collection_name: str,
                 distance: models.Distance = models.Distance.COSINE,
                 db_host: str = "localhost", db_port=6333,
                 verbose: bool = True):
        self._verbose = verbose
        self._query_model_name = query_model_name
        self._passage_model_name = passage_model_name
        self._collection_name = collection_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_retrieve_tmetrics = {}
        
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Device: {self._device}")
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Query Model: {self._query_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Passge Model: {self._passage_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] vCollection: {db_host}:{db_port}/{self._collection_name}")  

        #self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder_name)
        self.query_encoder = SentenceTransformer(self._query_model_name, device=self._device)
        self.passage_encoder = SentenceTransformer(self._passage_model_name, device=self._device)

        self.distance_metric = distance

        self._embedding_dimension = self.passage_encoder.get_sentence_embedding_dimension()

        self._setup_db_client(db_host, db_port)
        self._setup_collection()
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] loaded.")

    
    
    def _get_passage_embedding(self, passage):
        return self.passage_encoder.encode(passage, show_progress_bar=self._verbose, batch_size=128)

    def _get_query_embedding(self, query):
        return self.query_encoder.encode(query)
    

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        #*** Initial Retrival ***
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Stage 1: Initial retrieval with k={k} candidates...")
        self._last_retrieve_tmetrics = {}
        t_start = time.time()
        query_embedding = self._get_query_embedding(query)
        self._last_retrieve_tmetrics['t_q_embedding'] = time.time() - t_start

        t_start = time.time()
        search_result = self._search(query_embedding, k=k, with_payload=True)
        self._last_retrieve_tmetrics['t_search'] = time.time() - t_start
        
        final_documents = [hit.payload['text'] for hit in search_result]

        return final_documents


class DPRRetriever(BaseDenseRetriever):
    """
    Dense Passage Retrieval (DPR) retriever using Sentence Transformers and Qdrant.
    """
    MULTI_QA_MPNET_BASE_COS_V1: Final[str] = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    FACEBOOK_DPR_QUERY_NQ: Final[str] = "facebook/dpr-question_encoder-single-nq-base"#"facebook/dpr-multiset-base-uncased"
    FACEBOOK_DPR_PASAGE_NQ: Final[str] = "facebook/dpr-ctx_encoder-single-nq-base"

    def __init__(self, #model_name: str = MULTI_QA_MPNET_BASE_COS_V1,
                 passage_model_name: str = FACEBOOK_DPR_PASAGE_NQ,
                 query_model_name: str = FACEBOOK_DPR_QUERY_NQ,
                 collection_name: str = "medical_dpr_collection", 
                 db_host: str = "localhost", db_port=6333,
                 distance: models.Distance = models.Distance.COSINE,
                 verbose: bool = True):
        #super().__init__(query_model_name=query_model_name, passage_model_name=passage_model_name, collection_name=collection_name, db_host=db_host, db_port=db_port, distance=distance, verbose=verbose)

        self._verbose = verbose
        self._query_model_name = query_model_name
        self._passage_model_name = passage_model_name
        self._collection_name = collection_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Device: {self._device}")
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Query Model: {self._query_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Passge Model: {self._passage_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] vCollection: {db_host}:{db_port}/{self._collection_name}")  


        self.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self._passage_model_name)
        self.passage_encoder = DPRContextEncoder.from_pretrained(self._passage_model_name).to(self._device)
        
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self._query_model_name)
        self.query_encoder = DPRQuestionEncoder.from_pretrained(self._query_model_name).to(self._device)
    

        self.distance_metric = distance
        self._embedding_dimension = self.passage_encoder.config.hidden_size

        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Extracted embedding dimension: {self._embedding_dimension}")

        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Model max length (from tokenizer): {self.passage_tokenizer.model_max_length}")
        

        self._setup_db_client(db_host, db_port)
        self._setup_collection()
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] loaded.") 

    def _get_passage_embedding(self, passage):
        #encoded_input = self.passage_tokenizer(passage,
        #                                           padding=True, 
        #                                           truncation=True,
        #                                           max_length = 512,
        #                                           return_tensors='pt'
        #                                        ).to(self._device)
        single_input = isinstance(passage, str)
        passages = [passage] if single_input else passage

        encoded_input = self.passage_tokenizer(passages,
                                               padding=True, 
                                               truncation=True,
                                               max_length = 512,
                                               return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            output = self.passage_encoder(**encoded_input)
            embeddings = output.pooler_output.cpu().numpy()

        return embeddings[0] if single_input else embeddings
    
    def _get_query_embedding(self, query):
        single_input = isinstance(query, str)
        queries = [query] if single_input else query

        encoded_input = self.query_tokenizer(queries, 
                                             padding=True, 
                                             truncation=True,
                                             max_length = 512,
                                             return_tensors='pt').to(self._device) #padding=True, truncation=True

        with torch.no_grad():
            output = self.query_encoder(**encoded_input)
            embeddings = output.pooler_output.cpu().numpy()
        
        return embeddings[0] if single_input else embeddings
    

class MedragFoundationRetriever(BaseDenseRetriever):
    """
    MedRAG retriever using a foundation model for dense retrieval.
    """
    BIOMEDNLP_PUBMEDBERT_BASE: Final[str] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    NCBI_MEDRAG_PUBMEDBERT_BASE: Final[str] = "ncbi/MedCPT-Query-Encoder"
    NCBI_MEDRAG_PSG : Final[str] = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, #model_name: str = BIOMEDNLP_PUBMEDBERT_BASE,
                 query_model_name: str = NCBI_MEDRAG_PUBMEDBERT_BASE,
                 passage_model_name: str = NCBI_MEDRAG_PSG,
                 collection_name: str = "medical_pubmedbert_collection", 
                 db_host: str = "localhost", db_port=6333,
                 distance: models.Distance = models.Distance.COSINE,
                 verbose: bool = True):
        super().__init__(query_model_name=query_model_name,
                         passage_model_name=passage_model_name,
                         collection_name=collection_name,
                         db_host=db_host, db_port=db_port,
                         distance=distance,
                         verbose=verbose)


class ModernBERTRetriever(BaseDenseRetriever):
    """
    
    """

    def __init__(self, encoder_model_name:str,
                 collection_name: str,
                 distance: models.Distance = models.Distance.COSINE,
                 db_host: str = "localhost", db_port=6333,
                 verbose: bool = True):
        self._verbose = verbose
        self._encoder_model_name = encoder_model_name
        #self._reranker_model_name = reranker_model_name
        self._collection_name = collection_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Device: {self._device}")
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Model: {self._encoder_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] vCollection: {db_host}:{db_port}/{self._collection_name}")  

        # Initial Retriever
        self.passage_tokenizer = AutoTokenizer.from_pretrained(self._encoder_model_name)
        self.passage_encoder = AutoModel.from_pretrained(self._encoder_model_name, torch_dtype=torch.bfloat16).to(self._device)

        self.query_tokenizer = self.passage_tokenizer
        self.query_encoder = self.passage_encoder

        # Reranker
        #self.modernbert_encoder = SentenceTransformer(self._encoder_model_name, device=self._device)
        #self.reranker = ColBERTReranker(model_name=self._reranker_model_name, verbose = self._verbose)#RAGPretrainedModel.from_pretrained(self._reranker_model_name).to(self._device)

        #**

        self.distance_metric = distance

        self._embedding_dimension = self.passage_encoder.config.hidden_size #self.passage_encoder.get_sentence_embedding_dimension()

        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Extracted embedding dimension: {self._embedding_dimension}")

        self._setup_db_client(db_host, db_port)
        self._setup_collection()
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] loaded.") 
    
    

    def _get_passage_embedding(self, passage):
        single_input = isinstance(passage, str)
        passages = [passage] if single_input else passage

        encoded_input = self.passage_tokenizer(passages,
                                               padding=True,
                                               truncation=True,
                                               #max_length=8192,    
                                               return_tensors="pt"
                                               ).to(self._device)
            
        with torch.no_grad():
            output = self.passage_encoder(**encoded_input)
            last_hidden_state = output.last_hidden_state
            attention_mask = encoded_input['attention_mask']
            
            embeddings = self._average_pooling(last_hidden_state, attention_mask)

        return embeddings[0] if single_input else embeddings
    
    def _get_query_embedding(self, query):
        single_input = isinstance(query, str)
        queries = [query] if single_input else query

        encoded_input = self.query_tokenizer(queries,
                                             padding=True,
                                             truncation=True,
                                             #max_length=8192,
                                             return_tensors='pt'
                                             ).to(self._device) #padding=True, truncation=True

        with torch.no_grad():
            output = self.query_encoder(**encoded_input)
            last_hidden_state = output.last_hidden_state
            attention_mask = encoded_input['attention_mask']

            embeddings = self._average_pooling(last_hidden_state, attention_mask)

        return embeddings[0] if single_input else embeddings
    

#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
class MCSymmetricRetriever(ModernBERTRetriever):
    """
    
    """
    
    
    MODERNBERT_BASE: Final[str] = "answerdotai/ModernBERT-base"
    MODERNBERT_CLINICAL: Final[str] = 'Simonlee711/Clinical_ModernBERT'
    MODERNBERT_TUNED_10K_COS_RAND = 'models/ModernBERT-tuned-10k-cos-rand42'
    MODERNBERT_TUNED_10K_COS_BM25 = 'models/ModernBERT-tuned-10k-cos-bm25'
    MODERNBERT_TUNED_10K_COS_IBNS = 'models/ModernBERT-tuned-10k-cos-ibns'
    
    MODERNBERT_TUNED_10K_DOT_RAND = 'models/ModernBERT-tuned-10k-dot-rand42'
    MODERNBERT_TUNED_10K_DOT_BM25 = 'models/ModernBERT-tuned-10k-dot-bm25'
    MODERNBERT_TUNED_10K_DOT_IBNS = 'models/ModernBERT-tuned-10k-dot-ibns'

    MODERNBERT_CLINIC_TUNED_10K_COS_IBNS = 'models/Clinical_ModernBERT-tuned-10k-cos-ibns'
    
    COLBERT_BASE : Final[str] = "colbert-ir/colbertv2.0"
    COLBERT_TUNED_10K_COS_RAND = 'models/ColBERT-10k-cos-rand42/checkpoint-25000'
    COLBERT_TUNED_10K_COS_BM25 = 'models/ColBERT-10k-cos-bm25/checkpoint-25000'

    def __init__(self, encoder_model_name:str, reranker_model_name: str,
                 collection_name: str,
                 distance: models.Distance = models.Distance.COSINE,
                 db_host: str = "localhost", db_port=6333,
                 verbose: bool = True):
        super().__init__(encoder_model_name = encoder_model_name,
                 collection_name = collection_name,
                 distance = distance,
                 db_host=db_host, db_port=db_port,
                 verbose = verbose)
        self._reranker_model_name = reranker_model_name
        self._last_retrieve_init_docs = None
        self.reranker = ColBERTReranker(model_name=self._reranker_model_name, verbose = self._verbose)#RAGPretrainedModel.from_pretrained(self._reranker_model_name).to(self._device)
        vprint(self._verbose, f"--> [Re-ranker {self._reranker_model_name}] loaded.") 
    
    
    def _rerank(self, query, passages, top_k: int):
        result = self.reranker.rerank(query, passages, top_k=top_k)

        reranked_passages = [r[0] for r in result]
        
        return reranked_passages

    def retrieve(self, query: str, k: int = 3, init_k: int = None, keep_last_init_docs:bool=False) -> list[str]:
        #*** Initial Retrival ***
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Stage 1: Initial retrieval with k={init_k} candidates...")
        self._last_retrieve_tmetrics = {}
        t_start = time.time()
        query_embedding = self._get_query_embedding(query)
        self._last_retrieve_tmetrics['t_q_embedding'] = time.time() - t_start

        t_start = time.time()
        search_result = self._search(query_embedding, k=k, with_payload=True)
        self._last_retrieve_tmetrics['t_search'] = time.time() - t_start

        candidate_docs = [hit.payload['text'] for hit in search_result]

        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Retrieved {len(candidate_docs)} candidates.")

        #*** Re-ranking ***
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Stage 2: Reranking candidates with k={k}...")
        t_start = time.time()
        final_documents = self._rerank(query, candidate_docs, k)
        self._last_retrieve_tmetrics['t_rerank'] = time.time() - t_start
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Re-ranking complete.")

        if keep_last_init_docs:
            self._last_retrieve_init_docs = candidate_docs
        return final_documents
    
    def get_last_retrieve_init_docs(self):
        return self._last_retrieve_init_docs



class MedGemmaRetriever(BaseDenseRetriever):
    """
    
    """
    
    MEDGEMA_IT_4B: Final[str] = "google/medgemma-4b-it"
    HF_TOKEN = 'HFTOKEN'

    def __init__(self,
                 query_model_name: str = MEDGEMA_IT_4B,
                 passage_model_name: str = MEDGEMA_IT_4B,
                 collection_name: str = "medical_dpr_collection", 
                 db_host: str = "localhost", db_port=6333,
                 distance: models.Distance = models.Distance.COSINE,
                 verbose: bool = True):
        
        self._verbose = verbose
        self._query_model_name = query_model_name
        self._passage_model_name = passage_model_name
        self._collection_name = collection_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Device: {self._device}")
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Query Model: {self._query_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Passge Model: {self._passage_model_name}") 
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] vCollection: {db_host}:{db_port}/{self._collection_name}")  

        self.passage_tokenizer = AutoTokenizer.from_pretrained(self._passage_model_name, token=MedGemmaRetriever.HF_TOKEN)
        self.passage_encoder = AutoModel.from_pretrained(self._passage_model_name, torch_dtype=torch.bfloat16, token=MedGemmaRetriever.HF_TOKEN).to(self._device)
        
        self.query_tokenizer = self.passage_tokenizer
        self.query_encoder = self.passage_encoder

        if self.passage_tokenizer.pad_token is None:
            self.passage_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.passage_encoder.resize_token_embeddings(len(self.passage_tokenizer))

        #self.passage_tokenizer.eval()
        #self.passage_encoder.eval()

        self.distance_metric = distance
        self._embedding_dimension = self.passage_encoder.config.text_config.hidden_size #2560
        #self._embedding_dimension = self.passage_encoder.config.hidden_size #self.passage_encoder.config.projection_dim #self.passage_encoder.config.hidden_size
        #if self._embedding_dimension == 0:
        #    self._embedding_dimension = self.passage_encoder.config.hidden_size

        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Extracted embedding dimension: {self._embedding_dimension}")

        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] Model max length (from tokenizer): {self.passage_tokenizer.model_max_length}")
        

        self._setup_db_client(db_host, db_port)
        self._setup_collection()
        vprint(self._verbose, f"--> [Retriever {type(self).__name__}] loaded.") 

    
    def _get_passage_embedding(self, passage):
        single_input = isinstance(passage, str)
        passages = [passage] if single_input else passage

        encoded_input = self.passage_tokenizer(passages,
                                               padding=True,
                                               #truncation=True,
                                               #max_length=self.passage_tokenizer.model_max_length,
                                               return_tensors='pt'
                                            ).to(self._device)
        
        with torch.no_grad():
            output = self.passage_encoder(**encoded_input)
            last_hidden_state = output.last_hidden_state
            attention_mask = encoded_input['attention_mask']

            embeddings = self._average_pooling(last_hidden_state, attention_mask)

        return embeddings[0] if single_input else embeddings
    

    def _get_query_embedding(self, query):
        single_input = isinstance(query, str)
        queries = [query] if single_input else query

        encoded_input = self.query_tokenizer(queries,
                                             padding=True,
                                             #truncation=True,
                                             #max_length=self.passage_tokenizer.model_max_length,
                                             return_tensors='pt'
                                             ).to(self._device) #padding=True, truncation=True

        with torch.no_grad():
            output = self.query_encoder(**encoded_input)
            last_hidden_state = output.last_hidden_state
            attention_mask = encoded_input['attention_mask']

            embeddings = self._average_pooling(last_hidden_state, attention_mask)

        return embeddings[0] if single_input else embeddings
