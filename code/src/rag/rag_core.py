from utils.vprint import vprint
from rag.retriever import BM25Retriever, DenseRetriever, ModernBERTRetriever, DPRRetriever, MedGemmaRetriever, QRetriever, MedragFoundationRetriever, MCSymmetricRetriever
from rag.generator import Generator
from dataset import data_loader as dl
from rag.app_settings import AppSettings, RetrieverType, getRetrieverCollection
from qdrant_client import models
import time

class RAG:
    """
    Retrieval-Augmented Generation (RAG) system for medical documents.
    """
    _DEFAULT_PROMPT_TEMPLATE = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful and precise medical assistant. Your role is to answer user questions based *only* on the provided medical context.
        If the context does not contain the information needed to answer the question, you must state: "I cannot answer the question based on the provided context."
        Be concise and do not include information not present in the context.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CONTEXT:
        ---
        {context_str}
        ---
        QUESTION: {query}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    _DEFAULT_PROMPT_MULTIPLE_CHOICE_TEMPLATE_OLD = """
        You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as {{"step_by_step_thinking": Str(explanation), "answer_choice": Str{{A/B/C/...}}}}. answer_choice must be exclusively a letter from the answer choises. Your responses will be used for research purposes only, so please have a definite answer.
        Here are the relevant documents:
        {context_str}
        Here is the question:
        {query}
        Here are the potential choices:
        {options}
        Please think step-by-step and generate all your output only in json, do not include any other text.
        """
    
    _DEFAULT_PROMPT_MULTIPLE_CHOICE_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a meticulous and highly accurate medical AI assistant.
Your sole purpose is to analyze provided medical documents to answer a multiple-choice question.
You MUST strictly follow all instructions and provide your output ONLY in the specified JSON format.
<|eot_id|><|start_header_id|>user<|end_header_id|>
### TASK ###
Analyze the documents and answer the question according to the rules below.

### CONTEXT DOCUMENTS ###
{context_str}

### QUESTION AND OPTIONS ###
**Question:** {query}

**Options:**
{options}

### OUTPUT RULES ###
1.  **Reasoning:** First, think step-by-step to arrive at your conclusion. Your entire thought process must be captured in the `step_by_step_thinking` field.
2.  **Relevance Check:** Determine if the context documents were relevant and necessary to answer the question. Use "YES" or "NOT" for the `relevant_context` field.
3.  **Final Answer:** Choose one single, definitive letter corresponding to the correct option. This will be the value for the `answer_choice` field.
4.  **Strict JSON Format:** Your entire response MUST be a single, raw JSON object. Do not write any text, explanation, or markdown formatting (like ```json) before or after the JSON object.

Your response must conform to this exact JSON structure:
```json
{{
  "step_by_step_thinking": "Your detailed analysis and reasoning to reach the answer.",
  "relevant_context": "YES",
  "answer_choice": "C"
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    
    def __init__(self, retriever: QRetriever, generator: Generator, top_k: int = 3, prompt_template: str = None, init_top_k: int = 20, verbose: bool = True):
        """
        Initializes the RAG system.
        Args:
            retriever: An instance of a retriever class to fetch relevant documents.
            ollama_model (str): The name of the Ollama model to use for generation.
            ollama_host (str): The host address of the Ollama server.
            ollama_port (int): The port number of the Ollama server.
            verbose (bool): Whether to print verbose output.
        """
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._verbose = verbose
        self._prompt_template = prompt_template if prompt_template else RAG._DEFAULT_PROMPT_TEMPLATE
        self._init_top_k = init_top_k

    def _build_prompt(self, query: str, context: list[str], is_multi_opt: bool = False, options: dict = None) -> str:
        """
        Builds the prompt for the LLM using the user query and retrieved context.
        Args:
            query (str): The user's question.
            context (list[str]): The list of retrieved document texts.
            is_multi_opt (bool): Whether the query is a multiple-choice question.
            options (dict): A dictionary of options for multiple-choice questions.
                    If `is_multi_opt` is True, `options` should be provided in the format
                    {
                        "A": "Option A text",
                        "B": "Option B text",
                        "C": "Option C text",
                        ...
                    }
                    If `is_multi_opt` is False, `options` should be None.
        Returns:
            str: The formatted prompt for the LLM.
        """
        #context_str = "\n\n\n\n".join(context)

        context_str = ''

        for i, doc in enumerate(context):
            context_str += f"Document {i+1}:\n{doc}\n\n"

        if is_multi_opt:
            options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
            prompt = self._prompt_template.format(context_str=context_str, query=query, options=options_str)
        else:
            prompt = self._prompt_template.format(context_str=context_str, query=query)

        return prompt
    
    def index_documents(self, data_source_name: str = "MedRAG/pubmed", subset_size: int = 5000):
        """
        Indexes the documents from the specified data source.
        Args:
            data_source_name (str): The source of the documents to index.
        """
        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Indexing documents from {data_source_name}...")
        dataset = dl.retrieve_data(dataset_name=data_source_name, hugging_face=True, verbose=self._verbose)
        dataset = dataset.select(range(subset_size)) if subset_size > 0 else dataset

        self._retriever.index(documents=dataset, batch_size=2048)

        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Indexing completed.")
    
    def query(self, user_query: str, is_multi_opt: bool = False, options: dict = None,
              return_context:bool = False,
              return_init_context:bool = False,
              return_time_metrics:bool = False) -> dict:
        """
        Processes a user query by retrieving relevant documents and generating a response using the LLM.
        Args:
            user_query (str): The user's question.
            is_multi_opt (bool): Whether the query is a multiple-choice question.
            options (dict): A dictionary of options for multiple-choice questions.
                If `is_multi_opt` is True, `options` should be provided in the format
                    {
                        "A": "Option A text",
                        "B": "Option B text",
                        "C": "Option C text",
                        ...
                    }
                If `is_multi_opt` is False, `options` should be None.
        Returns:
            dict: A dictionary containing the generated answer and the retrieved context.
        """
        t_metrics = dict()
        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Query: '{user_query}, is_multi_opt: {is_multi_opt}, options: {options}'")

        #Retrieving documents
        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Retrieving documents...")
        t_start = time.time()
        if isinstance(self._retriever, MCSymmetricRetriever):
            retrieved_docs_text = self._retriever.retrieve(user_query, k=self._top_k, init_k=self._init_top_k, keep_last_init_docs=return_init_context)
        else:
            retrieved_docs_text = self._retriever.retrieve(user_query, k=self._top_k)
            
        t_metrics['t_retrieving'] = time.time() - t_start
        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Document retrieved : {len(retrieved_docs_text)}")

        #Building prompt
        t_start = time.time()
        prompt = self._build_prompt(user_query, retrieved_docs_text, is_multi_opt=is_multi_opt, options=options)
        t_metrics['t_building_prompt'] = time.time() - t_start

        #Generating response
        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Sending prompt to Generator...")
        t_start = time.time()
        answer = self._generator.query(prompt)
        t_metrics['t_query'] = time.time() - t_start

        results = {
            "answer": answer,
            #"retrieved_context": retrieved_docs_text
        }

        if return_context:
            results['retrieved_context'] = retrieved_docs_text
        if return_init_context and isinstance(self._retriever, MCSymmetricRetriever) :
            results['retrieved_init_context'] = self._retriever.get_last_retrieve_init_docs()

        if return_time_metrics:
            if isinstance(self._retriever, DenseRetriever):
                t_metrics_retirve = self._retriever.get_last_retrieve_metrics()
                t_metrics.update(t_metrics_retirve if t_metrics_retirve else {})
            results['t_metrics'] = t_metrics

        vprint(self._verbose, f"--> [RAG Core][{type(self).__name__}] Process finished.")
        return results

        
class RAGSystemFactory:
    """
    Factory class to create RAG systems based on the specified retriever type.
    This class is used to encapsulate the logic of creating different RAG systems.
    """
    
    @staticmethod
    def create_rag_system(retriever_type: RetrieverType, db_host: str = None, db_port: int = 6333, 
                          local_index_filename:str = 'bm25_index.pkl', local_corpus_filename:str = 'bm25_corpus.pkl',
                          ollama_model: str = 'llama3', ollama_host: str='0.0.0.0', ollama_port: int=11434,
                          top_k: int = 5, reranker_top_k: int = 20,
                          verbose: bool = True) -> RAG:
        """
        Creates a RAG system based on the specified retriever type.
        Args:
            retriever_type (RetrieverType): The type of retriever to use.
            db_host (str): The host address of the database.
            db_port (int): The port number of the database.
            ollama_model (str): The name of the Ollama model to use for generation.
            ollama_host (str): The host address of the Ollama server.
            ollama_port (int): The port number of the Ollama server.
            verbose (bool): Whether to print verbose output.
        Returns:
            MedicalRAGSystem: An instance of the RAG system with the specified retriever type.
        """
        generator = Generator(model_name=ollama_model, host=ollama_host, port=ollama_port, verbose=verbose)

        retriever = RAGSystemFactory.create_retriever(retriever_type = retriever_type,
                                                      db_host = db_host,
                                                      db_port = db_port,
                                                      local_index_filename=local_index_filename,
                                                      local_corpus_filename=local_corpus_filename,
                                                      verbose = verbose)

        return RAG(retriever=retriever, generator=generator, top_k=top_k, prompt_template=RAG._DEFAULT_PROMPT_MULTIPLE_CHOICE_TEMPLATE,
                   init_top_k=reranker_top_k if retriever_type in [RetrieverType.MC_SYMETRIC_BASE,
                                                                    RetrieverType.MC_SYM_CLINICAL,
                                                                    RetrieverType.MC_SYM_TUNED_10K_COS_IBNS,
                                                                    RetrieverType.MC_SYM_TUNED_10K_COS_RAND,
                                                                    RetrieverType.MC_SYC_TUNED_10K_COS_BM25,
                                                                    RetrieverType.MC_SYM_TUNED_10K_DOT_RAND,
                                                                    RetrieverType.MC_SYM_TUNED_10K_DOT_BM25,
                                                                    RetrieverType.MC_SYM_TUNED_10K_DOT_IBNS,
                                                                    RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS,
                                                                   ] else None,
                   verbose=verbose)
    
    @staticmethod
    def create_retriever(retriever_type: RetrieverType, db_host: str = None, db_port: int = 6333,
                         local_index_filename:str = 'bm25_index.pkl', local_corpus_filename:str = 'bm25_corpus.pkl',
                         verbose: bool = True) -> QRetriever:
        """
        Creates a RAG system based on the specified retriever type.
        Args:
            retriever_type (RetrieverType): The type of retriever to use.
            db_host (str): The host address of the database.
            db_port (int): The port number of the database.
            ollama_model (str): The name of the Ollama model to use for generation.
            ollama_host (str): The host address of the Ollama server.
            ollama_port (int): The port number of the Ollama server.
            verbose (bool): Whether to print verbose output.
        Returns:
            MedicalRAGSystem: An instance of the RAG system with the specified retriever type.
        """
        retriever = None
            
        #***** BM25 *****#
        if retriever_type == RetrieverType.BM25:            
            retriever = BM25Retriever(index_path = local_index_filename, 
                                      corpus_path = local_corpus_filename, 
                                      load_from_disk = True, 
                                      verbose = verbose)
        #***** DPR *****#
        elif retriever_type == RetrieverType.DPR:
            retriever = DPRRetriever( #model_name = DPRRetriever.MULTI_QA_MPNET_BASE_COS_V1,
                query_model_name = DPRRetriever.FACEBOOK_DPR_QUERY_NQ, 
                passage_model_name = DPRRetriever.FACEBOOK_DPR_PASAGE_NQ,
                collection_name = getRetrieverCollection(RetrieverType.DPR), #"medical_dpr_collection", 
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
                )
        #***** MedRAG *****#
        elif retriever_type == RetrieverType.MEDRAG:
            retriever  = MedragFoundationRetriever(
                #model_name = MedragFoundationRetriever.BIOMEDNLP_PUBMEDBERT_BASE,
                query_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PUBMEDBERT_BASE,
                passage_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PSG,
                collection_name = getRetrieverCollection(RetrieverType.MEDRAG), #"medical_pubmedbert_collection",
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        #***** MedGemma *****#
        elif retriever_type == RetrieverType.MED_GEMMA:
            retriever  = MedGemmaRetriever(
                query_model_name = MedGemmaRetriever.MEDGEMA_IT_4B, 
                passage_model_name = MedGemmaRetriever.MEDGEMA_IT_4B,
                collection_name = getRetrieverCollection(RetrieverType.MED_GEMMA), #'medical_medgemma_collection', 
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
                )
        #***** ModernBERT Base *****#
        elif retriever_type == RetrieverType.MODERNBERT_BASE:
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_BASE,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_BASE), #"medical_sync_mc_collection", #<---------- Same ModernBERT
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        #***** ModernBERT Clinical *****#
        elif retriever_type == RetrieverType.MODERNBERT_CLINICAL:
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_CLINICAL,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_CLINICAL), #"medical_sync_clinic_mc_collection", #<---------- Same ModernBERT Clinical
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        #***** M+C Base *****#
        elif retriever_type == RetrieverType.MC_SYMETRIC_BASE:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_BASE,
                reranker_model_name = MCSymmetricRetriever.COLBERT_BASE,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYMETRIC_BASE), #"medical_sync_mc_collection", #<---------- Same ModernBERT
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose

            )
        #***** M+C Clinical *****#
        elif retriever_type == RetrieverType.MC_SYM_CLINICAL:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_CLINICAL,#<---------- Same ModernBERT Clinical
                reranker_model_name = MCSymmetricRetriever.COLBERT_BASE,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_CLINICAL), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )

        #***** ModernBERT Tuned: 10k Cos Rand42 *****#
        elif retriever_type == RetrieverType.MODERNBERT_TUNED_10K_COS_RAND: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_COS_RAND,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_TUNED_10K_COS_RAND),
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        #***** M+C Tuned: 10k Cos Rand42 *****#
        elif retriever_type == RetrieverType.MC_SYM_TUNED_10K_COS_RAND:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_COS_RAND, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_RAND,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_TUNED_10K_COS_RAND), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
            
        #***** ModernBERT Tuned: 10k Cos BM25 *****#
        elif retriever_type == RetrieverType.MODERNBERT_TUNED_10K_COS_BM25: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_COS_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_TUNED_10K_COS_BM25),
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        #***** M+C Tuned: 10k Cos BM25 *****#
        elif retriever_type == RetrieverType.MC_SYC_TUNED_10K_COS_BM25:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_COS_BM25, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYC_TUNED_10K_COS_BM25), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )

        #***** ModernBERT Tuned: 10k Cos IBNS *****#
        elif retriever_type == RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_COS_IBNS,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS),
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        #***** M+C Tuned: 10k Cos IBNS *****#
        elif retriever_type == RetrieverType.MC_SYM_TUNED_10K_COS_IBNS:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_COS_IBNS, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_TUNED_10K_COS_IBNS), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )


        #***** ModernBERT Tuned: 10k Dot RAND *****#
        elif retriever_type == RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_DOT_RAND,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND),
                distance =  models.Distance.DOT,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        elif retriever_type == RetrieverType.MC_SYM_TUNED_10K_DOT_RAND:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_DOT_RAND, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_RAND,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_TUNED_10K_DOT_RAND), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.DOT,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )

        #***** ModernBERT Tuned: 10k Dot BM25 *****#
        elif retriever_type == RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_DOT_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25),
                distance =  models.Distance.DOT,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        elif retriever_type == RetrieverType.MC_SYM_TUNED_10K_DOT_BM25:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_DOT_BM25, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_TUNED_10K_DOT_BM25), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.DOT,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )

        
        #***** ModernBERT Tuned: 10k Dot IBNS *****#
        elif retriever_type == RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_DOT_IBNS,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS),
                distance =  models.Distance.DOT,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        elif retriever_type == RetrieverType.MC_SYM_TUNED_10K_DOT_IBNS:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_TUNED_10K_DOT_IBNS, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_TUNED_10K_DOT_IBNS), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.DOT,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )

        
        #***** ModernBERT Tuned: 10k Dot IBNS *****#
        elif retriever_type == RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS: # Base
            retriever  = ModernBERTRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_CLINIC_TUNED_10K_COS_IBNS,
                collection_name = getRetrieverCollection(RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS),
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        elif retriever_type == RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS:
            retriever  = MCSymmetricRetriever(
                encoder_model_name = MCSymmetricRetriever.MODERNBERT_CLINIC_TUNED_10K_COS_IBNS, # Re-Ranker
                reranker_model_name = MCSymmetricRetriever.COLBERT_TUNED_10K_COS_BM25,
                collection_name = getRetrieverCollection(RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS), #"medical_sync_clinic_mc_collection",
                distance =  models.Distance.COSINE,
                db_host = db_host, db_port=db_port,
                verbose = verbose
            )
        else:
            raise ValueError(f"Retriever type not valid: '{retriever_type}'. Use one of: {RetrieverType}")
        
        return retriever
        

class APIRAGSystem:
    """
    API wrapper for the MedicalRAGSystem.
    This class provides methods to index documents and process queries via an API.
    """
    
    def __init__(self, app_settings: AppSettings):
        self._app_settings = app_settings
        self._rag_systems_cache = {}

    async def initialize_rag_systems(self):
        """
        Initializes the RAG systems for different retriever types.
        """
        self._rag_systems_cache = {
            RetrieverType.BM25: RAGSystemFactory.create_rag_system(retriever_type = RetrieverType.BM25,
                                                                   ollama_model = self._app_settings.ollama_model,
                                                                   ollama_host = self._app_settings.ollama_host,
                                                                   ollama_port = self._app_settings.ollama_port,
                                                                   top_k = self._app_settings.retriever_top_k),

            RetrieverType.DPR: RAGSystemFactory.create_rag_system(retriever_type = RetrieverType.DPR, 
                                                db_host = self._app_settings.db_host, 
                                                db_port = self._app_settings.db_port, 
                                                ollama_model = self._app_settings.ollama_model, 
                                                ollama_host = self._app_settings.ollama_host, 
                                                ollama_port = self._app_settings.ollama_port,
                                                top_k = self._app_settings.retriever_top_k),

            RetrieverType.MEDRAG: RAGSystemFactory.create_rag_system(retriever_type = RetrieverType.MEDRAG, 
                                                   db_host = self._app_settings.db_host, 
                                                   db_port = self._app_settings.db_port, 
                                                   ollama_model = self._app_settings.ollama_model, 
                                                   ollama_host = self._app_settings.ollama_host, 
                                                   ollama_port = self._app_settings.ollama_port,
                                                   top_k = self._app_settings.retriever_top_k),

            RetrieverType.MC_SYMETRIC_BASE: RAGSystemFactory.create_rag_system(retriever_type = RetrieverType.MC_SYMETRIC_BASE,
                                                    db_host = self._app_settings.db_host, 
                                                    db_port = self._app_settings.db_port, 
                                                    ollama_model = self._app_settings.ollama_model, 
                                                    ollama_host = self._app_settings.ollama_host, 
                                                    ollama_port = self._app_settings.ollama_port,
                                                    top_k = self._app_settings.retriever_top_k,
                                                    reranker_top_k = self._app_settings.reranker_top_k
                                                    )
                                                                          
                                                                          
        }

    def get_rag_system(self, retriever_type: RetrieverType) -> RAG:
        """
        Gets the RAG system for the specified retriever type.
        """

        if retriever_type not in self._rag_systems_cache:
            raise Exception(f"Retriever type '{retriever_type}' not found.")
        
        rag_system = self._rag_systems_cache[retriever_type]
        
        return rag_system