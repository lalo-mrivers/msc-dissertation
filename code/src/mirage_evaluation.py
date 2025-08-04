import argparse
import benchmark.evaluation as eval
from rag.app_settings import RetrieverType


def main():
    parser = argparse.ArgumentParser(description="Index data")
    
    parser.add_argument('rag_type', type=RetrieverType, choices=list(RetrieverType), help="Type of RAG system to use")
    parser.add_argument('--mirage_data_dir', type=str, required=True, help="Directory containing Mirage data")
    parser.add_argument('--out_dir', type=str, required=True, help="Output directory for results")
    parser.add_argument('--top_k', type=int, required=True, help="Top K results to retrieve")
    parser.add_argument('--ollama_model', type=str, required=True, help="Ollama model to use")
    parser.add_argument('--ollama_host', type=str, required=True, help="Ollama host")
    parser.add_argument('--ollama_port', type=int, required=True, help="Ollama port")
    parser.add_argument('--db_host', type=str, help="Qdrant database host")
    parser.add_argument('--db_port', type=int, help="Qdrant database port")
    parser.add_argument('--local_index_filename', type=str, default=None, help="Local index filename for BM25 retriever")
    parser.add_argument('--local_corpus_filename', type=str, default=None, help="Local corpus filename for BM25 retriever")
    parser.add_argument('--init_top_k', type=int, default=None, help="Initial top K for retrieval")
    parser.add_argument('--corpus_name', type=str, default='pubmed', help="Name of the corpus to use")
    parser.add_argument('--resume_process', action='store_true', help="Resume processing if interrupted")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose output")

    args = parser.parse_args()
    
    if args.rag_type not in list(RetrieverType):
        raise ValueError("Invalid RAG type specified.")
    
    if not args.local_index_filename and args.retriever_type in [RetrieverType.BM25]:
        raise ValueError("local_index_filename must be provided for BM25 retrievers.")

    if not args.db_host and args.retriever_type not in [RetrieverType.BM25]:
        raise ValueError("db_host must be provided for dense retrievers for remote Qdrant.")


    eval.mirage_evaluation(rag_type=args.rag_type,
                           mirage_data_dir=args.mirage_data_dir,
                           out_dir=args.out_dir,
                           top_k=args.top_k,
                           ollama_model=args.ollama_model,
                           ollama_host=args.ollama_host,
                           ollama_port=args.ollama_port,
                           db_host=args.db_host,
                           db_port=args.db_port,
                           local_index_filename=args.local_index_filename,
                           local_corpus_filename=args.local_corpus_filename,
                           init_top_k=args.init_top_k,
                           corpus_name=args.corpus_name,
                           resume_process=args.resume_process,
                           verbose=args.verbose
                        )
                        
def run_all_mirage_evaluations():
    out_dir = 'data/benchmark_v2/out/'
    data_dir = 'data/mirage'
    top_k=5
    init_top_k=20
    run = 1

    def run_benchmark_forward(rag_type: RetrieverType):
        eval.mirage_evaluation(
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

    if run == 1:
        run_benchmark_forward(RetrieverType.BM25)
        run_benchmark_forward(RetrieverType.DPR)
        run_benchmark_forward(RetrieverType.MEDRAG)
        run_benchmark_forward(RetrieverType.MED_GEMMA)
        run_benchmark_forward(RetrieverType.MODERNBERT_BASE)
        run_benchmark_forward(RetrieverType.MODERNBERT_CLINICAL)
        run_benchmark_forward(RetrieverType.MC_SYMETRIC_BASE)
        run_benchmark_forward(RetrieverType.MC_SYM_CLINICAL)

        run_benchmark_forward(RetrieverType.MODERNBERT_BASE)
        run_benchmark_forward(RetrieverType.MC_SYMETRIC_BASE)

        run_benchmark_forward(RetrieverType.MODERNBERT_CLINICAL)
        run_benchmark_forward(RetrieverType.MC_SYM_CLINICAL)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS)
        run_benchmark_forward(RetrieverType.MC_SYM_TUNED_10K_COS_IBNS)

    elif run == 2:
        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_COS_BM25)
        run_benchmark_forward(RetrieverType.MC_SYC_TUNED_10K_COS_BM25)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_COS_RAND)
        run_benchmark_forward(RetrieverType.MC_SYM_TUNED_10K_COS_RAND)

        run_benchmark_forward(RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS)
        run_benchmark_forward(RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS)
    elif run == 3:
        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND)
        ##run_benchmark_forward(RetrieverType.MC_SYMETRIC_TUNED_10K_DOT_RAND)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25)
        ##run_benchmark_forward(RetrieverType.MC_SYMETRIC_TUNED_10K_DOT_BM25)

        run_benchmark_forward(RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS)
        ##run_benchmark_forward(RetrieverType.MC_SYMETRIC_TUNED_10K_DOT_IBNS)
        #'''
    else:
        pass
    print('end')

if __name__ == "__main__":
    #main()
    run_all_mirage_evaluations()