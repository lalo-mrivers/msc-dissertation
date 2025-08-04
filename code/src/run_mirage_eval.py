from benchmark.evaluation import eval_mirage_output, TOP_K, run_benchmark_forward
from rag.app_settings import RetrieverType

'''
class RetrieverType(str, Enum):
    BM25 = "BM25"
    DPR = "DPR"
    MEDRAG = "MedRAG"
    MC_SYMETRIC = "MCSymmetric"
    MED_GEMMA = "MedGemma"
'''
def main():
    out_dir = 'data/benchmark_v2/out/'
    data_dir = 'data/mirage'
    run = 2

    if run == 1:
        run_benchmark_forward(RetrieverType.BM25, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.DPR, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MEDRAG, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MED_GEMMA, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MODERNBERT_BASE, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MODERNBERT_CLINICAL, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYMETRIC_BASE, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYM_CLINICAL, data_dir, out_dir)
        
        run_benchmark_forward(RetrieverType.MODERNBERT_BASE, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYMETRIC_BASE, data_dir, out_dir)
        
        run_benchmark_forward(RetrieverType.MODERNBERT_CLINICAL, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYM_CLINICAL, data_dir, out_dir)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYM_TUNED_10K_COS_IBNS, data_dir, out_dir)

    elif run == 2:
        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_COS_BM25, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYC_TUNED_10K_COS_BM25, data_dir, out_dir)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_COS_RAND, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYM_TUNED_10K_COS_RAND, data_dir, out_dir)
        
        run_benchmark_forward(RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS, data_dir, out_dir)
        run_benchmark_forward(RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS, data_dir, out_dir)
    elif run == 3:
        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND, data_dir, out_dir)
        ##run_benchmark_forward(RetrieverType.MC_SYMETRIC_TUNED_10K_DOT_RAND, data_dir, out_dir)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25, data_dir, out_dir)
        ##run_benchmark_forward(RetrieverType.MC_SYMETRIC_TUNED_10K_DOT_BM25, data_dir, out_dir)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS, data_dir, out_dir)
        ##run_benchmark_forward(RetrieverType.MC_SYMETRIC_TUNED_10K_DOT_IBNS, data_dir, out_dir)
        #'''
    else:
        pass
    print('end')

if __name__ == "__main__":
    main()