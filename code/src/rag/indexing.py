import json
import traceback
from qdrant_client import QdrantClient
from rag.app_settings import RetrieverType, getRetrieverCollection
from rag.rag_core import RAGSystemFactory
#from dataset import data_loader as dl
from datasets import load_from_disk, load_dataset
from rag.retriever import BM25Retriever, BaseDenseRetriever
import pandas as pd
from datetime import datetime
import os

def get_dataset(file_ids:str, dataset_name:str, cache:bool = True, cache_path:str = 'data/'):
    
    suffix = '_index_cached'
    dataset_cache_path = os.path.join(cache_path, f'{dataset_name}{suffix}')
    if cache and os.path.exists(dataset_cache_path):
        filtered_dataset = load_from_disk(dataset_cache_path)
    else:
        #dataset = dl.retrieve_data(dataset_name=dataset_name, hugging_face=True, verbose=True)
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)

        # Carga los IDs guardados
        df_ids = pd.read_csv(file_ids)
        id_set = set(df_ids["id"].tolist())

        # Filtra el dataset usando esos IDs
        filtered_dataset = dataset.filter(lambda x: x['id'] in id_set)

        #full_dataset = load_dataset(dataset_name, split='train[:100000]')
        if cache:
            filtered_dataset.save_to_disk(dataset_cache_path)

    return filtered_dataset

def get_unindex_data(qclient:QdrantClient, collection_name:str, dataset):    
    collection_info = qclient.get_collection(collection_name=collection_name)
    n_datapoints = collection_info.points_count
    print(f"Collection '{collection_name}' has {n_datapoints} points.")
    existing_ids = set()

    offset = 0
    limit = 500

    while True:
        #print(f'{offset}')
        points, next_offset = qclient.scroll(
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

    
    print(f"Total IDs in collection '{collection_name}': {len(existing_ids)}")
    
    unindex_dataset = dataset.filter(lambda datapoint: datapoint["id"] not in existing_ids, num_proc=4)

    print(f"Total original datapoints             : {len(dataset)}")
    print(f"Total missing datapoints to process   : {len(unindex_dataset)}")

    return unindex_dataset


def runIndex(ids_filename:str,
             dataset_name:str,
             retriever_type:RetrieverType,
             cache_dataset:bool = False,
             cache_path:str = 'data/',
             local_index_filename:str = None, local_corpus_filenam:str = None,
             db_host:str = None, db_port:int = None,
             log_dirpath:str = './',
             verbose:bool = False):
    
    print(f'Getting dataset {dataset_name}')
    dataset = get_dataset(file_ids=ids_filename,
                          dataset_name=dataset_name,
                          cache=cache_dataset,
                          cache_path=cache_path)
    if retriever_type == RetrieverType.BM25:
        t_metrics = runSparseIndex(dataset, retriever_type = retriever_type,
                       local_index_filename = local_index_filename,
                       local_corpus_filename = local_corpus_filenam,
                        )
    else:
        t_metrics = runDenseIndex(dataset, retriever_type = retriever_type,
                      db_host = db_host,
                      db_port = db_port,
                      verbose = verbose)

    with open(os.path.join(log_dirpath, f'logindex_{str(retriever_type)}.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(t_metrics, ensure_ascii=False) + '\n')



def runSparseIndex(data, retriever_type:RetrieverType, local_index_filename:str, local_corpus_filename:str) -> dict:
    t_metrics = {'start': datetime.now().isoformat()}
    print(f'Initializing retriever {retriever_type} ...')
    retriever = RAGSystemFactory.create_retriever(retriever_type, 
                                                  local_index_filename = local_index_filename, 
                                                  local_corpus_filename = local_corpus_filename,
                                                  verbose = True)
    if isinstance(retriever, BM25Retriever):

        try:
            print(f'Indexing data...')
            index_times = retriever.index(data, batch_size=500)
            t_metrics.update(index_times)
        except Exception as e:
            t_metrics['error'] = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

        t_metrics['RetrieverType'] = str(retriever_type)
        t_metrics['local_index_filename'] = local_index_filename
        t_metrics['local_corpus_filename'] = local_corpus_filename
        
        print("Indexing completed successfully.")
    else:
        print(f'Dense retriever not valid {retriever_type}')
    
    t_metrics['end'] = datetime.now().isoformat()
    return t_metrics


def runDenseIndex(data, retriever_type:RetrieverType, db_host:str, db_port:int, verbose:bool = False) -> dict:
    t_metrics = {'start': datetime.now().isoformat()}
    print(f'Initializing retriever {retriever_type} ...')
    retriever = RAGSystemFactory.create_retriever(retriever_type, 
                                                  db_host =db_host,
                                                  db_port=db_port,
                                                  verbose = verbose)
    if isinstance(retriever, BaseDenseRetriever):
        try:
            print(f'Getting unindexed data ...')
            dataset_filtrado = get_unindex_data(retriever.db_client, getRetrieverCollection(retriever_type), data)
            
            print(f'Indexing data ...')
            index_times = retriever.index(dataset_filtrado, batch_size=500)
            t_metrics.update(index_times)
        except Exception as e:
            t_metrics['error'] = ''.join(traceback.format_exception(type(e), e, e.__traceback__))

        t_metrics['RetrieverType'] = str(retriever_type)
        t_metrics['qdrant_host'] = db_host
        t_metrics['qdrant_port'] = db_port
        
        print("Indexing completed successfully.")
    else:
        print(f'Dense retriever not valid {retriever_type}')
    
    t_metrics['end'] = datetime.now().isoformat()
    return t_metrics



