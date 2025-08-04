from tqdm import tqdm
from benchmark.mirage.utils import QADataset, locate_answer4pub_llama
from benchmark.mirage.evaluate import evaluate
from rag.rag_core import RAG, RAGSystemFactory
from rag.app_settings import AppSettings, RetrieverType
import os
import json
import time
import re

QDRANT_HOST = AppSettings.QDRANT_HOST
QDRANT_PORT = AppSettings.QDRANT_PORT
OLLAMA_MODEL = AppSettings.OLLAMA_MODEL
OLLAMA_HOST = AppSettings.OLLAMA_HOST
OLLAMA_PORT = AppSettings.OLLAMA_PORT

INIT_TOP_K = 20
TOP_K = 5

DATASET_NAMES = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']



def test_rag(rag: RAG, bench_data: QADataset, save_dir:str, split='test', resume_process=True):
    len_data = len(bench_data)
    gtstart = time.time()
    tstart = gtstart 


    for q_idx in tqdm(range(len_data), desc=f"Processing {split} data"):
    #for q_idx in range(len_data):
        #'''
        if q_idx % ((len_data // 100)*10) == 0:
            perctg = int((q_idx / len_data) * 100)
            ctime = time.time()
            print(f"Progress: {perctg}% in {ctime - tstart:.2f}s ({ctime - gtstart:.2f}s total)")
            tstart = ctime
        #'''
        fpath = os.path.join(save_dir, split + "_" + bench_data.index[q_idx] + ".json")
        if os.path.exists(fpath) and resume_process and os.path.getsize(fpath) > 0:
            continue
        
        question = bench_data[q_idx].get('question')
        options = bench_data[q_idx].get('options')
        result = rag.query(user_query=question, is_multi_opt=True, options=options,
                           return_context = True,
                           return_init_context = True,
                           return_time_metrics = True)
        raw_answer = result.get('answer')

        try:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump([raw_answer], f, ensure_ascii=False)
        except Exception as e:
            print(f'Error: {e}. Writing Raw Answer [{fpath}]')
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f'[{raw_answer}]')

        save_metric_data(save_dir, split, bench_data.index[q_idx], question, options, result)
        
    print(f"Total time taken: {time.time() - gtstart:.2f}s")
    print(f"Results saved to {save_dir}")
    


def extract_response(response: str):
    '''
    Extracts elements from the response string.
    '''
    res = {}
    if response:
        try:
            res = json.loads(response)
        except Exception as e:
            str_response = str(response) if not isinstance(response, str) else response
            sst = re.search(r'"step_by_step_thinking":.*?"(.*?)"', str_response)
            ac = re.search(r'"answer_choice":.*?([A-Za-z]+?)', str_response)
            rc = re.search(r'"relevant_context":.+?"(.*?)"', str_response)

            if sst:
                sst = sst.group(1)

            if ac:
                ac = ac.group(1)

            if rc:
                rc = rc.group(1)

            res = {
                'step_by_step_thinking': sst,
                'answer_choice': ac,
                'relevant_context': rc
                }
    
    return res

def save_metric_data(save_dir:str, split:str, question_id,
                    question,
                    options,
                    rag_result:dict):
    
    allfile = os.path.join(save_dir, split + "_output.jsonl")
    allfile_context = os.path.join(save_dir, split + "_output_context.jsonl")
    raw_answer = rag_result.get('answer')

    processed_answer = extract_response(raw_answer)
    with open(allfile, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'id': question_id,
            'question': question,
            'options': options,
            'answer': raw_answer,
            'step_by_step_thinking': processed_answer.get('step_by_step_thinking', None),
            'answer_choice': processed_answer.get('answer_choice', None),
            'relevant_context': processed_answer.get('relevant_context', None),
            't_metrics': rag_result.get('time_metrics', None),
        }, ensure_ascii=False) + '\n')
    with open(allfile_context, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'id': question_id,
            'relevant_context': processed_answer.get('relevant_context', None),
            'context': rag_result.get('retrieved_context', None),
            'context': rag_result.get('retrieved_init_context', None),
        }, ensure_ascii=False) + '\n')

def run_benchmark_forward(rag_type: RetrieverType, data_dir, out_dir, corpus_name = 'pubmed'):
    print('*'*100)
    print(f'***** {rag_type} *****')
    print('*'*100)
    print(f'Instanciado modelo {rag_type}')
    if rag_type == RetrieverType.BM25:
        rag = RAGSystemFactory.create_rag_system(retriever_type = RetrieverType.BM25,
                                                ollama_model = OLLAMA_MODEL, 
                                                ollama_host = OLLAMA_HOST, 
                                                ollama_port = OLLAMA_PORT,
                                                top_k = TOP_K,
                                                verbose=False)
        retriever_name = str(RetrieverType.BM25)
    elif rag_type in {RetrieverType.DPR,
                      RetrieverType.MEDRAG, 
                      RetrieverType.MODERNBERT_BASE,
                      RetrieverType.MODERNBERT_CLINICAL,
                      RetrieverType.MED_GEMMA,
                      RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS,
                      RetrieverType.MODERNBERT_TUNED_10K_COS_BM25,
                      RetrieverType.MODERNBERT_TUNED_10K_COS_RAND,
                      RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS,
                      RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND,
                      RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25,
                      RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS,
                      }:
        rag = RAGSystemFactory.create_rag_system(retriever_type = rag_type, 
                                                db_host = QDRANT_HOST, 
                                                db_port = QDRANT_PORT, 
                                                ollama_model = OLLAMA_MODEL, 
                                                ollama_host = OLLAMA_HOST, 
                                                ollama_port = OLLAMA_PORT,
                                                top_k = TOP_K,
                                                verbose=False)
        retriever_name = str(rag_type)
    elif rag_type in {RetrieverType.MC_SYMETRIC_BASE,
                      RetrieverType.MC_SYM_CLINICAL,
                      RetrieverType.MC_SYM_TUNED_10K_COS_IBNS,
                      RetrieverType.MC_SYC_TUNED_10K_COS_BM25,
                      RetrieverType.MC_SYM_TUNED_10K_COS_RAND,
                      RetrieverType.MC_SYM_TUNED_10K_DOT_RAND,
                      RetrieverType.MC_SYM_TUNED_10K_DOT_BM25,
                      RetrieverType.MC_SYM_TUNED_10K_DOT_IBNS,
                      RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS,
                      }:
        rag = RAGSystemFactory.create_rag_system(retriever_type = rag_type,
                                                db_host = QDRANT_HOST, 
                                                db_port = QDRANT_PORT, 
                                                ollama_model = OLLAMA_MODEL, 
                                                ollama_host = OLLAMA_HOST, 
                                                ollama_port = OLLAMA_PORT,
                                                top_k = TOP_K,
                                                reranker_top_k = INIT_TOP_K,
                                                verbose=False
                                                )
        retriever_name = str(rag_type)
    else:
        print('**** No type ****')
        return
    
    llm_name = rag._generator._model_name
    dataset_names = DATASET_NAMES

    print('Iniciando evaluacion...')
    print(f'{rag_type} -> {retriever_name}')
    for ds_name in dataset_names:
        bm_outdir = os.path.join(out_dir, ds_name, "rag_"+str(rag._top_k), llm_name, corpus_name, retriever_name)
        print(f'{rag_type}[{ds_name}]')
        test_rag(rag, QADataset(ds_name, dir=data_dir), bm_outdir, split='test')#'data/benchmark/out/')



def mirage_evaluation(rag_type: RetrieverType, mirage_data_dir: str, out_dir: str, top_k: int, 
                      ollama_model: str, ollama_host: str, ollama_port: int,
                      db_host: str = None, db_port: int = None,
                      local_index_filename:str = None, local_corpus_filename:str = None,
                      init_top_k: int = None,
                      corpus_name: str = 'pubmed',
                      resume_process: bool = True,
                      verbose: bool = False):
    print('*'*100)
    print(f'***** {rag_type} *****')
    print('*'*100)
    print(f'Instanciado modelo {rag_type}')
    if rag_type == RetrieverType.BM25:
        rag = RAGSystemFactory.create_rag_system(retriever_type = RetrieverType.BM25,
                                                ollama_model = ollama_model, 
                                                ollama_host = ollama_host, 
                                                ollama_port = ollama_port,
                                                local_index_filename = local_index_filename,
                                                local_corpus_filename = local_corpus_filename,
                                                top_k = top_k,
                                                verbose=verbose)
        retriever_name = str(RetrieverType.BM25)
    elif rag_type in {RetrieverType.DPR,
                      RetrieverType.MEDRAG, 
                      RetrieverType.MODERNBERT_BASE,
                      RetrieverType.MODERNBERT_CLINICAL,
                      RetrieverType.MED_GEMMA,
                      RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS,
                      RetrieverType.MODERNBERT_TUNED_10K_COS_BM25,
                      RetrieverType.MODERNBERT_TUNED_10K_COS_RAND,
                      RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS,
                      RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND,
                      RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25,
                      RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS,
                      }:
        rag = RAGSystemFactory.create_rag_system(retriever_type = rag_type,
                                                db_host = db_host, 
                                                db_port = db_port,
                                                ollama_model = ollama_model, 
                                                ollama_host = ollama_host, 
                                                ollama_port = ollama_port,
                                                top_k = top_k,
                                                verbose=verbose)
        retriever_name = str(rag_type)
    elif rag_type in {RetrieverType.MC_SYMETRIC_BASE,
                      RetrieverType.MC_SYM_CLINICAL,
                      RetrieverType.MC_SYM_TUNED_10K_COS_IBNS,
                      RetrieverType.MC_SYC_TUNED_10K_COS_BM25,
                      RetrieverType.MC_SYM_TUNED_10K_COS_RAND,
                      RetrieverType.MC_SYM_TUNED_10K_DOT_RAND,
                      RetrieverType.MC_SYM_TUNED_10K_DOT_BM25,
                      RetrieverType.MC_SYM_TUNED_10K_DOT_IBNS,
                      RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS,
                      }:
        rag = RAGSystemFactory.create_rag_system(retriever_type = rag_type,
                                                db_host = db_host, 
                                                db_port = db_port,
                                                ollama_model = ollama_model, 
                                                ollama_host = ollama_host, 
                                                ollama_port = ollama_port,
                                                top_k = top_k,
                                                reranker_top_k = init_top_k,
                                                verbose=verbose
                                                )
        retriever_name = str(rag_type)
    else:
        print('**** No type ****')
        return
    
    llm_name = rag._generator._model_name
    dataset_names = DATASET_NAMES

    print('Iniciando evaluacion...')
    print(f'{rag_type} -> {retriever_name}')
    for ds_name in dataset_names:
        bm_outdir = os.path.join(out_dir, ds_name, "rag_"+str(rag._top_k), llm_name, corpus_name, retriever_name)
        print(f'{rag_type}[{ds_name}]')
        test_rag(rag, QADataset(ds_name, dir=mirage_data_dir), bm_outdir, split='test', resume_process=resume_process)



def eval_mirage_output(rag_type: RetrieverType, data_dir, out_dir, k, corpus_name = 'pubmed', llm_name='llama3'):
    dataset_names = DATASET_NAMES
    datasets = {key:QADataset(key, dir=data_dir) for key in dataset_names}
    retriever_name = str(rag_type)
    all_scores = {}

    scores = []
    for dataset_name in dataset_names:
        print("[{:s}] ".format(dataset_name), end="")
        split = "test"
        
        save_dir = os.path.join(out_dir, dataset_name, "rag_"+str(k), llm_name, corpus_name, retriever_name)
        #print(f'Looking {dataset_name} in {save_dir}')
        if os.path.exists(save_dir):
            #if "pmc_llama" in llm_name.lower():
            #    acc, std, flag = evaluate(datasets[dataset_name], save_dir, split, locate_answer4pub_llama)
            #else:
            #    acc, std, flag = evaluate(datasets[dataset_name], save_dir, split)
            acc, std, flag = evaluate(datasets[dataset_name], save_dir, split, locate_answer4pub_llama)
            all_scores[dataset_name] = {'acc': acc, 'std': std}
            scores.append(acc)
            print("mean acc: {:.4f}; proportion std: {:.4f}".format(acc, std), end="")
            if flag:
                print(" (NOT COMPLETED)")
            else:
                print("")
        else:
            print("NOT STARTED.")
            # scores.append(0)

    if len(scores) > 0:
        print("[Average] mean acc: {:.4f}".format(sum(scores) / len(scores)))
    
    return all_scores
