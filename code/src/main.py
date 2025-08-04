
from rag.app_settings import RetrieverType
from benchmark.evaluation import eval_mirage_output
from training.prepare_data_triplets import generate_triplets
from datasets import load_dataset
from benchmark.evaluation import mirage_evaluation

from rag.app_settings import RetrieverType
import pandas as pd


def test_prepare_data_pubmed_qa():
    print('test_prepare_data_pubmed_qa')
    dataset = load_dataset('bigbio/pubmed_qa', split=f"train[:{1000}]")
    dataset = dataset.filter(lambda example: example["final_decision"] == "yes")
    generate_triplets(
        output_filename='./data/training/test_bigbio_pubmed_qa_triplets_rand_7.jsonl',
        dataset=dataset,
        in_nm_query='QUESTION',
        in_nm_positive='CONTEXTS',
        out_nm_query='query',
        out_nm_positive='positive',
        out_nm_negative='negative',
        method='random',
        seed=7,
        max_samples=100,
        num_process=4
    )
    print("Data preparation completed.")

def test_prepare_data_pubmed():
    print('test_prepare_data_pubmed')
    dataset = load_dataset('MedRAG/pubmed', split=f"train[:{1000}]")
    generate_triplets(
        output_filename='./data/training/test_medrag_pubmed_triplets_rand_7.jsonl',
        dataset=dataset,
        in_nm_query='title',
        in_nm_positive='content',
        out_nm_query='query',
        out_nm_positive='positive',
        out_nm_negative='negative',
        method='random',
        seed=7,
        max_samples=100,
        num_process=4
    )
    print("Data preparation completed.")
    


def test_mirage_evaluations():
    print('test_mirage_evaluations')
    out_dir = 'data/benchmark_v2/out/'
    data_dir = 'data/mirage'
    top_k=5
    init_top_k=20
    run = 2

    def run_benchmark_forward(rag_type: RetrieverType):
        mirage_evaluation(
            rag_type=rag_type,
            mirage_data_dir=data_dir,
            out_dir=out_dir,
            top_k=top_k,
            ollama_model='llama3:8b',
            ollama_host='https://ollama-llama3-162981050281.us-central1.run.app',
            ollama_port=None,
            db_host='http://uhtred.inf.ed.ac.uk:6333/',#'https://qdrant-162981050281.us-central1.run.app',
            db_port=None,
            local_index_filename='bm25_index.pkl',
            local_corpus_filename='bm25_corpus.pkl',
            init_top_k=init_top_k,
            corpus_name='pubmed',
            resume_process=True,
            verbose=False
        )

    run_benchmark_forward(RetrieverType.DPR)



def test_parse_results():
    print('test_parse_results')
    out_dir = 'data/benchmark/out/'
    data_dir = 'data/mirage'
    top_k = 5
    model_name = 'llama3'
    def get_evaluation(rag_type: RetrieverType):
        print(f"******* Running evaluation for {rag_type}...")
        return eval_mirage_output(rag_type=rag_type,
                                  data_dir=data_dir,
                                  out_dir=out_dir,
                                  k=top_k,
                                  corpus_name='pubmed',
                                  llm_name=model_name)
    
    #*************************#
    results = {}
    
    results[str(RetrieverType.DPR)] = get_evaluation(RetrieverType.DPR)
    
    acc_dict = {}
    std_dict = {}

    for row_key, cols in results.items():
        acc_dict[row_key] = {}
        std_dict[row_key] = {}
        for col_key, metrics in cols.items():
            acc_dict[row_key][col_key] = metrics['acc']
            std_dict[row_key][col_key] = metrics['std']

    # Crear los DataFrames
    df_acc = pd.DataFrame.from_dict(acc_dict, orient='index')
    df_std = pd.DataFrame.from_dict(std_dict, orient='index')

    # Mostrar resultados
    print("DataFrame de ACC:\n", df_acc)
    print("\nDataFrame de STD:\n", df_std)

if __name__ == "__main__":
    #test_prepare_data_pubmed_qa()
    ##test_prepare_data_pubmed()
    #test_mirage_evaluations()
    test_parse_results()
    print("All tests completed successfully.")