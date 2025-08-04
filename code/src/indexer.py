import argparse
import json
import traceback
from qdrant_client import QdrantClient
from rag import indexing
from rag.app_settings import RetrieverType, getRetrieverCollection
from rag.rag_core import RAGSystemFactory
from dataset import data_loader as dl
from datasets import load_from_disk
from rag.retriever import DPRRetriever, MedragFoundationRetriever, MCSymmetricRetriever, MedGemmaRetriever
import pandas as pd
import sys
from datetime import datetime
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

client = QdrantClient(
    host="uhtred.inf.ed.ac.uk",  # Cambia si usas otro host
    port=6333          # Cambia si usas otro puerto
)

QDRANT_HOST = None#"uhtred.inf.ed.ac.uk"
QDRANT_PORT = None#6333
QDRANT_URL = 'https://qdrant-162981050281.us-central1.run.app/'
client = QdrantClient(
    url=QDRANT_URL,
    host=QDRANT_HOST,  # Cambia si usas otro host
    port=QDRANT_PORT          # Cambia si usas otro puerto
)

DATASET = 'MedRAG/pubmed'
FILE_IDS = '/home/s2586627/msc-dissertation/msc-dissertation/train_hf_ids.csv'

def get_dataset(file_ids):
    
    if os.path.exists(DATASET + '_index_cached'):
        filtered_dataset = load_from_disk(DATASET + '_index_cached')
    else:
        dataset = dl.retrieve_data(dataset_name=DATASET, hugging_face=True, verbose=True)

        # Carga los IDs guardados
        df_ids = pd.read_csv(file_ids)
        id_set = set(df_ids["id"].tolist())

        # Filtra el dataset usando esos IDs
        filtered_dataset = dataset.filter(lambda x: x['id'] in id_set)

        
        #full_dataset = load_dataset(dataset_name, split='train[:100000]')
        filtered_dataset.save_to_disk(DATASET + '_index_cached')

    return filtered_dataset

def get_unindex_data(collection_name, dataset):    
    collection_info = client.get_collection(collection_name=collection_name)
    n_datapoints = collection_info.points_count
    print(f"Collection '{collection_name}' has {n_datapoints} points.")
    existing_ids = set()

    offset = 0
    limit = 500

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            with_payload=["id_hf"],
            with_vectors=False,
            offset=offset,
            limit=limit,
            timeout=20
        )
        #print(f"Retrieved {len(points)} points from offset {offset}.")
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            if "id_hf" in payload:
                existing_ids.add(payload["id_hf"])
        
        if next_offset is None:
            break

        offset = next_offset  

    
    print(f"Existing IDs in collection '{collection_name}': {len(existing_ids)}")
    
    unindex_dataset = dataset.filter(lambda datapoint: datapoint["id"] not in existing_ids, num_proc=4)

    print(f"Total datos originales     : {len(dataset)}")
    print(f"Total datos sin procesar   : {len(unindex_dataset)}")

    return unindex_dataset


def run():
    dataset = get_dataset(FILE_IDS)
    

    print('***** DPR (DPR) *****')
    runDenseIndex(dataset, RetrieverType.DPR)

    print('***** MEDRAG (MedRAG) *****')
    runDenseIndex(dataset, RetrieverType.MEDRAG)

    print('***** MED_GEMMA (MedGemma) *****')
    runDenseIndex(dataset, RetrieverType.MED_GEMMA)

    print('***** MODERNBERT (ModernBERT) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_BASE)

    
    print('***** MODERNBERT_CLINICAL (ModernBERTClinical) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_CLINICAL)

    print('***** MODERNBERT_TUNED_10K_COS_IBNS (ModernBERT_10kCosIbns) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS)
    
    print('***** MODERNBERT_TUNED_10K_COS_RAND (ModernBERT_10kCosRand42) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_TUNED_10K_COS_RAND)

    print('***** MODERNBERT_TUNED_10K_COS_BM25 (ModernBERT_10kCosBM25) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_TUNED_10K_COS_BM25)
    

    print('***** MODERNBERT_TUNED_10K_DOT_RAND (ModernBERT_10kDotRand42) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND)

    print('***** MODERNBERT_TUNED_10K_DOT_BM25 (ModernBERT_10kDotBM25) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25)

    print('***** MODERNBERT_TUNED_10K_DOT_IBNS (ModernBERT_10kDotIbns) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS)

    print('***** MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS (ModernBERTClinical_10kCosIbns) *****')
    runDenseIndex(dataset, RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS)



def runDenseIndex(data, retriever_type:RetrieverType):
    start = datetime.now().isoformat()
    retriever = RAGSystemFactory.create_retriever(retriever_type, 
                                                  db_host =QDRANT_HOST,
                                                  db_port=QDRANT_PORT,
                                                  verbose = True)
    dataset_filtrado = get_unindex_data(getRetrieverCollection(retriever_type), data)

    try:
        t_metrics = retriever.index(dataset_filtrado, batch_size=500)
    except Exception as e:
        t_metrics = {'error': ''.join(traceback.format_exception(type(e), e, e.__traceback__))}

    t_metrics['start'] = start
    t_metrics['RetrieverType'] = str(retriever_type)
    t_metrics['qdrant_host'] = QDRANT_HOST
    t_metrics['qdrant_port'] = QDRANT_PORT
    

    with open(f'index_{str(retriever_type)}.jsonl' , 'a', encoding='utf-8') as f:
            f.write(json.dumps(t_metrics, ensure_ascii=False) + '\n')
    
    print("Tuned Indexing completed successfully.")



def runModernColBERT_tuned(data):
    print("Starting indexing process...")
    collection_name = "medical_mc_10k_cos_ibns"

    retriever = MCSymmetricRetriever(
        encoder_model_name='models/ModernBERT-tuned-10k-cos-ibns',
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = collection_name,
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)


    dataset_filtrado = get_unindex_data(collection_name, data)
    retriever.index(dataset_filtrado, batch_size=500)
    print("Tuned Indexing completed successfully.")

def runModernColBERT(data):
    print("Starting indexing process...")
    collection_name = "medical_sync_mc_collection"

    retriever = MCSymmetricRetriever(
        encoder_model_name=MCSymmetricRetriever.MODERNBERT_BASE,
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = collection_name,
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)


    dataset_filtrado = get_unindex_data(collection_name, data)
    retriever.index(dataset_filtrado, batch_size=500)
    print("MedRAG Indexing completed successfully.")

def runClinicalModernColBERT(data):
    print("Starting indexing process...")
    collection_name = "medical_sync_clinic_mc_collection"

    retriever = MCSymmetricRetriever(
        encoder_model_name=MCSymmetricRetriever.MODERNBERT_CLINICAL,
        reranker_model_name=MCSymmetricRetriever.COLBERT_BASE,
        collection_name = collection_name,
        db_host="uhtred.inf.ed.ac.uk",
        db_port=6333)


    dataset_filtrado = get_unindex_data(collection_name, data)
    retriever.index(dataset_filtrado, batch_size=500)
    print("MedRAG Indexing completed successfully.")

def runDPR(data):
    print("Starting indexing process...")
    collection_name = "medical_dpr_collection"

    retriever = DPRRetriever( #model_name = DPRRetriever.MULTI_QA_MPNET_BASE_COS_V1,
                query_model_name = DPRRetriever.FACEBOOK_DPR_QUERY_NQ, 
                passage_model_name = DPRRetriever.FACEBOOK_DPR_PASAGE_NQ,
                collection_name = collection_name, 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    
    dataset_filtrado = get_unindex_data(collection_name, data)
    retriever.index(dataset_filtrado, batch_size=500)
    print("MedRAG Indexing completed successfully.")

def runMedRAG(data):
    print("Starting indexing process...")
    collection_name = "medical_pubmedbert_collection"

    retriever = MedragFoundationRetriever(
                #model_name = MedragFoundationRetriever.BIOMEDNLP_PUBMEDBERT_BASE,
                query_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PUBMEDBERT_BASE,
                passage_model_name = MedragFoundationRetriever.NCBI_MEDRAG_PSG,
                collection_name = collection_name,
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True
            )
    
    dataset_filtrado = get_unindex_data(collection_name, data)
    retriever.index(dataset_filtrado, batch_size=500)
    print("MedRAG Indexing completed successfully.")


def runMedGemma(data):
    print("Starting indexing process...")
    collection_name = "medical_medgemma_collection"

    retriever = MedGemmaRetriever( #model_name = DPRRetriever.MULTI_QA_MPNET_BASE_COS_V1,
                query_model_name = MedGemmaRetriever.MEDGEMA_IT_4B, 
                passage_model_name = MedGemmaRetriever.MEDGEMA_IT_4B,
                collection_name = collection_name, 
                db_host = "uhtred.inf.ed.ac.uk", db_port=6333,
                verbose = True)
    
    dataset_filtrado = get_unindex_data(collection_name, data)
    retriever.index(dataset_filtrado, batch_size=100)
    print("MedRAG Indexing completed successfully.")

#if __name__ == "__main__":
#    run()
#    #runModernColBERT()


def main():
    parser = argparse.ArgumentParser(description="Index data")

    parser.add_argument("retriever_type", type=str, choices=[r.name for r in RetrieverType], help="Type of retriever to use")
    parser.add_argument("ids_filename", type=str, help="Path to the file with IDs")
    parser.add_argument("--dataset_name", type=str, default='MedRAG/pubmed', help="Name of the dataset (HuggingFace) or local path")
    parser.add_argument("--cache_dataset", action='store_true', help="Cache the dataset locally")
    parser.add_argument("--cache_path", type=str, default='data/', help="Path to cache the dataset")
    parser.add_argument("--local_index_filename", type=str, help="Local index filename for sparse retrievers")
    parser.add_argument("--local_corpus_filename", type=str, help="Local corpus filename for sparse retrievers")
    parser.add_argument("--db_host", type=str, help="Qdrant database host")
    parser.add_argument("--db_port", type=int, help="Qdrant database port")
    parser.add_argument("--log_outdir", type=str, default='./', help="Directory path for log files")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    #python code/src/indexer.py DPR train_hf_ids.csv --dataset_name 'MedRAG/pubmed' --db_host 'https://qdrant-162981050281.us-central1.run.app/'  --cache_dataset
    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if not args.local_index_filename and args.retriever_type in [RetrieverType.BM25]:
        raise ValueError("local_index_filename must be provided for BM25 retrievers.")

    if not args.db_host and args.retriever_type not in [RetrieverType.BM25]:
        raise ValueError("db_host must be provided for dense retrievers for remote Qdrant.")


    indexing.runIndex(ids_filename=args.ids_filename,
                        dataset_name=args.dataset_name,
                        retriever_type=RetrieverType[args.retriever_type],
                        cache_dataset=args.cache_dataset,
                        cache_path=args.cache_path,
                        local_index_filename=args.local_index_filename,
                        local_corpus_filenam=args.local_corpus_filename,
                        db_host=args.db_host,
                        db_port=args.db_port,
                        log_dirpath=args.log_outdir,
                        verbose=args.verbose)

def run_lib():
    def run_indexing_proc(retriever:RetrieverType):
        indexing.runIndex(ids_filename='train_hf_ids.csv',
                        dataset_name='MedRAG/pubmed',
                        retriever_type=retriever,
                        cache_dataset=True,
                        cache_path='data/',
                        db_host='https://qdrant-162981050281.us-central1.run.app', #'uhtred.inf.ed.ac.uk',
                        db_port=6333,
                        log_dirpath='data/')

    print('***** DPR (DPR) *****')
    run_indexing_proc(RetrieverType.DPR)

    print('***** MEDRAG (MedRAG) *****')
    run_indexing_proc(RetrieverType.MEDRAG)

    print('***** MED_GEMMA (MedGemma) *****')
    run_indexing_proc(RetrieverType.MED_GEMMA)

    print('***** MODERNBERT (ModernBERT) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_BASE)

    
    print('***** MODERNBERT_CLINICAL (ModernBERTClinical) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_CLINICAL)

    print('***** MODERNBERT_TUNED_10K_COS_IBNS (ModernBERT_10kCosIbns) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS)
    
    print('***** MODERNBERT_TUNED_10K_COS_RAND (ModernBERT_10kCosRand42) *****')
    run_indexing_proc( RetrieverType.MODERNBERT_TUNED_10K_COS_RAND)

    print('***** MODERNBERT_TUNED_10K_COS_BM25 (ModernBERT_10kCosBM25) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_TUNED_10K_COS_BM25)
    

    print('***** MODERNBERT_TUNED_10K_DOT_RAND (ModernBERT_10kDotRand42) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND)

    print('***** MODERNBERT_TUNED_10K_DOT_BM25 (ModernBERT_10kDotBM25) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25)

    print('***** MODERNBERT_TUNED_10K_DOT_IBNS (ModernBERT_10kDotIbns) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS)

    print('***** MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS (ModernBERTClinical_10kCosIbns) *****')
    run_indexing_proc(RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS)


if __name__ == "__main__":
    main()
    #run_lib()
    #runModernColBERT()

