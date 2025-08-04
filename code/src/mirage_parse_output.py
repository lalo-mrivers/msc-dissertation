from benchmark.evaluation import eval_mirage_output
from rag.app_settings import RetrieverType
import pandas as pd


def get_all_results():
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
    
    for rt in list(RetrieverType):
        print(f"******* {rt} *******")
        results[str(rt)] = get_evaluation(rt)
    
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
    get_all_results()