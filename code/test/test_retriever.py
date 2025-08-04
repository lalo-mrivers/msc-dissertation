
import torch
import numpy as np
from src.rag.retriever import DPRRetriever, BaseDenseRetriever, MCSymmetricRetriever, MedGemmaRetriever, MedragFoundationRetriever
from datasets import Dataset

def _test_dense_retriever_encode_passage(dense_retriever: BaseDenseRetriever):
    #**********#
    text = "Hola mundo!"
    embeddings = dense_retriever._get_passage_embedding(text)
    print(type(embeddings))
    print(embeddings.shape)
    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 1, "Expected 1D tensor for single input"
    assert embeddings.shape[0] > 0, "Embedding should have some dimensions"

    #**********#
    texts = ["Hola mundo!", "Es otro test", "De pruebas para verificar el codigo"]
    embeddings = dense_retriever._get_passage_embedding(texts)
    print(type(embeddings))
    print(embeddings.shape)
    
    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2, "Expected 2D tensor for multiple inputs"
    assert embeddings.shape[0] == len(texts), "Number of embeddings must match input size"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    #**********#
    texts = ["Hola mundo!"]
    embeddings = dense_retriever._get_passage_embedding(texts)
    print(type(embeddings))
    print(embeddings.shape)

    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2, "Even for one-item list, output should be 2D"
    assert embeddings.shape[0] == 1

    
def _test_dense_retriever_encode_query(dense_retriever: BaseDenseRetriever):
    #**********#
    text = "Hola mundo!"
    embeddings = dense_retriever._get_query_embedding(text)
    print(type(embeddings))
    print(embeddings.shape)
    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 1, "Expected 1D tensor for single input"
    assert embeddings.shape[0] > 0, "Embedding should have some dimensions"

    #**********#
    texts = ["Hola mundo!", "Es otro test", "De pruebas para verificar el codigo"]
    embeddings = dense_retriever._get_query_embedding(texts)
    print(type(embeddings))
    print(embeddings.shape)
    
    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2, "Expected 2D tensor for multiple inputs"
    assert embeddings.shape[0] == len(texts), "Number of embeddings must match input size"
    assert embeddings.shape[1] > 0, "Embedding dimension should be > 0"

    #**********#
    texts = ["Hola mundo!"]
    embeddings = dense_retriever._get_query_embedding(texts)
    print(type(embeddings))
    print(embeddings.shape)

    assert isinstance(embeddings, torch.Tensor) or isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2, "Even for one-item list, output should be 2D"
    assert embeddings.shape[0] == 1

def _test_dense_index(dense_retriever: BaseDenseRetriever):
    docs = _get_test_data()
    indexed = False
    try:
        dense_retriever.index(docs, batch_size = 8)
        indexed = True
    except Exception as e:
        print(f'Error: {e}')
        indexed = False
        
    assert indexed, 'Index process failed'


def _test_dense_retrieve(dense_retriever: BaseDenseRetriever):
    docs = _get_test_data()
    retrieved = False
    ex = ''
    try:
        res = dense_retriever.retrieve(docs[0].get('contents'), k=1)
        retrieved = True
    except Exception as e:
        print(f'Error: {e}')
        ex = e
        retrieved = False
        
    assert retrieved, f'Retrieve process failed {ex}'
    assert res is not None, 'Result is None'
    assert len(res) > 0, 'Result is empty'

def _test_dense_retrieve_reranker(dense_retriever: BaseDenseRetriever):
    docs = _get_test_data()
    retrieved = False
    ex = ''
    try:
        res = dense_retriever.retrieve(docs[0].get('contents'), k=1, init_k=2)
        retrieved = True
    except Exception as e:
        print(f'Error: {e}')
        ex = e
        retrieved = False
        
    assert retrieved, f'Retrieve process failed {ex}'
    assert res is not None, 'Result is None'
    assert len(res) > 0, 'Result is empty'


def _get_test_data():
    data = [{'id': 'a',
             'contents': 'Texto de prueba A'},
            {'id': 'b',
             'contents': 'Texto de prueba B'},
            {'id': 'c',
             'contents': 'Texto de prueba C'},]

    dataset = Dataset.from_list(data)
    return dataset

def test_dpr_encode_passage():
    retriever = DPRRetriever(
                passage_model_name = DPRRetriever.FACEBOOK_DPR_PASAGE_NQ,
                query_model_name = DPRRetriever.FACEBOOK_DPR_QUERY_NQ,
                collection_name = 'test_dpr', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    _test_dense_retriever_encode_passage(retriever)


def test_dpr_encode_query():
    retriever = DPRRetriever(
                passage_model_name = DPRRetriever.FACEBOOK_DPR_PASAGE_NQ,
                query_model_name = DPRRetriever.FACEBOOK_DPR_QUERY_NQ,
                collection_name = 'test_dpr', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    _test_dense_retriever_encode_query(retriever)

def test_dpr_index():
    retriever = DPRRetriever(
                passage_model_name = DPRRetriever.FACEBOOK_DPR_PASAGE_NQ,
                query_model_name = DPRRetriever.FACEBOOK_DPR_QUERY_NQ,
                collection_name = 'test_dpr', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    _test_dense_index(retriever)

def test_dpr_retrieve():
    retriever = DPRRetriever(
                passage_model_name = DPRRetriever.FACEBOOK_DPR_PASAGE_NQ,
                query_model_name = DPRRetriever.FACEBOOK_DPR_QUERY_NQ,
                collection_name = 'test_dpr', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    _test_dense_retrieve(retriever)

#*************** MedRAG **********************
def test_medrag_encode_passage():
    retriever = MedragFoundationRetriever(
                query_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PUBMEDBERT_BASE,
                passage_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PSG,
                collection_name = 'test_medrag',
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True
            )
    _test_dense_retriever_encode_passage(retriever)


def test_medrag_encode_query():
    retriever = MedragFoundationRetriever(
                query_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PUBMEDBERT_BASE,
                passage_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PSG,
                collection_name = 'test_medrag',
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True
            )
    
    _test_dense_retriever_encode_query(retriever)

def test_medrag_index():
    retriever = MedragFoundationRetriever(
                query_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PUBMEDBERT_BASE,
                passage_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PSG,
                collection_name = 'test_medrag',
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True
            )
    _test_dense_index(retriever)

def test_medrag_retrieve():
    retriever = MedragFoundationRetriever(
                query_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PUBMEDBERT_BASE,
                passage_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PSG,
                collection_name = 'test_medrag',
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True
            )
    _test_dense_retrieve(retriever)


#*************** ModernBERT + ColBERT **********************
def test_syncmc_encode_passage():
    retriever = MCSymmetricRetriever(
        encoder_model_name=MCSymmetricRetriever.MODERNBERT_BASE,
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = 'test_sync_mc',
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)
    
    _test_dense_retriever_encode_passage(retriever)


def test_syncmc_encode_query():
    retriever = MCSymmetricRetriever(
        encoder_model_name=MCSymmetricRetriever.MODERNBERT_BASE,
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = 'test_sync_mc',
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)
    
    _test_dense_retriever_encode_query(retriever)

def test_syncmc_index():
    retriever = MCSymmetricRetriever(
        encoder_model_name=MCSymmetricRetriever.MODERNBERT_BASE,
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = 'test_sync_mc',
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)
    _test_dense_index(retriever)

def test_syncmc_retrieve():
    retriever = MCSymmetricRetriever(
        encoder_model_name=MCSymmetricRetriever.MODERNBERT_BASE,
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = 'test_sync_mc',
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)
    _test_dense_retrieve_reranker(retriever)


#*************** MedGemma **********************
def test_medgemma_encode_passage():
    retriever = MedGemmaRetriever(
                query_model_name = MedGemmaRetriever.MEDGEMA_IT_4B, 
                passage_model_name = MedGemmaRetriever.MEDGEMA_IT_4B,
                collection_name = 'test_medgemma', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    
    _test_dense_retriever_encode_passage(retriever)


def test_medgemma_encode_query():
    retriever = MedGemmaRetriever(
                query_model_name = MedGemmaRetriever.MEDGEMA_IT_4B, 
                passage_model_name = MedGemmaRetriever.MEDGEMA_IT_4B,
                collection_name = 'test_medgemma', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    
    _test_dense_retriever_encode_query(retriever)

def test_medgemma_index():
    retriever = MedGemmaRetriever(
                query_model_name = MedGemmaRetriever.MEDGEMA_IT_4B, 
                passage_model_name = MedGemmaRetriever.MEDGEMA_IT_4B,
                collection_name = 'test_medgemma', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    _test_dense_index(retriever)
    
def test_medgemma_retrieve():
    retriever = MedGemmaRetriever(
                query_model_name = MedGemmaRetriever.MEDGEMA_IT_4B, 
                passage_model_name = MedGemmaRetriever.MEDGEMA_IT_4B,
                collection_name = 'test_medgemma', 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    _test_dense_retrieve(retriever)