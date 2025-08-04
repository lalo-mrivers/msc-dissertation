from enum import Enum



class DataSource(str, Enum):
    PUBMED = "MedRAG/pubmed"
    CLINICAL_TRIALS = "MedRAG/clinical_trials"
    COVID_19 = "MedRAG/covid_19"
    ALL = "MedRAG/all"
    
    def __str__(self):
        return self.value
    
class RetrieverType(str, Enum):
    BM25 = "BM25"
    DPR = "DPR"
    MEDRAG = "MedRAG"
    MED_GEMMA = "MedGemma"
    
    MODERNBERT_BASE = "ModernBERT"
    MC_SYMETRIC_BASE = "MCSymmetric"

    MODERNBERT_CLINICAL = "ModernBERTClinical"
    MC_SYM_CLINICAL = "MCSymClinical"
    
    MODERNBERT_TUNED_10K_COS_IBNS = "ModernBERT_10kCosIbns"
    MC_SYM_TUNED_10K_COS_IBNS = "MCSymetric_10kCosIbns"
    
    MODERNBERT_TUNED_10K_COS_RAND = "ModernBERT_10kCosRand42"
    MC_SYM_TUNED_10K_COS_RAND = "MCSymetric_10kCosRand42"

    MODERNBERT_TUNED_10K_COS_BM25 = "ModernBERT_10kCosBM25"
    MC_SYC_TUNED_10K_COS_BM25 = "MCSymetric_10kCosBM25"
    
    MODERNBERT_TUNED_10K_DOT_RAND = "ModernBERT_10kDotRand42"
    MC_SYM_TUNED_10K_DOT_RAND = "MCSymetric_10kDotRand42"

    MODERNBERT_TUNED_10K_DOT_BM25 = "ModernBERT_10kDotBM25"
    MC_SYM_TUNED_10K_DOT_BM25 = "MCSymetric_10kDotBM25"

    MODERNBERT_TUNED_10K_DOT_IBNS = "ModernBERT_10kDotIbns"
    MC_SYM_TUNED_10K_DOT_IBNS = "MCSymetric_10kDotIbns"

    MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS = "ModernBERTClinical_10kCosIbns"
    MC_SYM_CLINICAL_TUNED_10K_COS_IBNS = "MCClinical_10kCosIbns"
    
    def __str__(self):
        return self.value
    
def getRetrieverCollection(retriever: RetrieverType):
    ret_map = {
        RetrieverType.DPR: "medical_dpr_collection",
        RetrieverType.MEDRAG: "medical_pubmedbert_collection",
        RetrieverType.MED_GEMMA: "medical_medgemma_collection",
        
        RetrieverType.MODERNBERT_BASE: "medical_sync_mc_collection", #<---------- Same ModernBERT
        RetrieverType.MC_SYMETRIC_BASE: "medical_sync_mc_collection", #<---------- Same ModernBERT

        RetrieverType.MODERNBERT_CLINICAL: "medical_sync_clinic_mc_collection", #<---------- Same ModernBERT Clinical
        RetrieverType.MC_SYM_CLINICAL: "medical_sync_clinic_mc_collection", #<---------- Same ModernBERT Clinical
        
        RetrieverType.MODERNBERT_TUNED_10K_COS_IBNS: "medical_mc_10k_cos_ibns",
        RetrieverType.MC_SYM_TUNED_10K_COS_IBNS: "medical_mc_10k_cos_ibns",

        RetrieverType.MODERNBERT_TUNED_10K_COS_RAND: "medical_mc_10k_cos_rand42",
        RetrieverType.MC_SYM_TUNED_10K_COS_RAND: "medical_mc_10k_cos_rand42",

        RetrieverType.MODERNBERT_TUNED_10K_COS_BM25: "medical_mc_10k_cos_bm25",
        RetrieverType.MC_SYC_TUNED_10K_COS_BM25: "medical_mc_10k_cos_bm25",

        RetrieverType.MODERNBERT_TUNED_10K_DOT_RAND: "medical_mc_10k_dot_rand42",
        RetrieverType.MC_SYM_TUNED_10K_DOT_RAND: "medical_mc_10k_dot_rand42",

        RetrieverType.MODERNBERT_TUNED_10K_DOT_BM25: "medical_mc_10k_dot_bm25",
        RetrieverType.MC_SYM_TUNED_10K_DOT_BM25: "medical_mc_10k_dot_bm25",

        RetrieverType.MODERNBERT_TUNED_10K_DOT_IBNS: "medical_mc_10k_dot_ibns",
        RetrieverType.MC_SYM_TUNED_10K_DOT_IBNS: "medical_mc_10k_dot_ibns",

        RetrieverType.MODERNBERT_CLINICAL_TUNED_10K_COS_IBNS: "medical_clinical_mc_10k_cos_ibns",
        RetrieverType.MC_SYM_CLINICAL_TUNED_10K_COS_IBNS: "medical_clinical_mc_10k_cos_ibns",
    }

    return ret_map.get(retriever)

    
class AppSettings:
    """
    Class to hold application settings for the RAG system.
    This includes data sources, retriever types, and other configurations.
    """

    QDRANT_HOST = "uhtred.inf.ed.ac.uk"
    QDRANT_PORT = 6333
    #OLLAMA_MODEL = "llama3"
    #OLLAMA_HOST = "crannog06.inf.ed.ac.uk" #** "crannog06.inf.ed.ac.uk" #"crannog02.inf.ed.ac.uk"
    #OLLAMA_PORT = 11435 #** 11435 #11434
    OLLAMA_MODEL = "llama3:8b"
    OLLAMA_HOST = "ollama-llama3-162981050281.us-central1.run.app"
    OLLAMA_PORT = None

    RETRIEVER_TOP_K = 5
    RERANKER_TOP_K = 20

    VERBOSE = True

    def __init__(self, 
                 db_host: str = QDRANT_HOST,
                 db_port: int = QDRANT_PORT,
                 ollama_model: str = OLLAMA_MODEL,
                 ollama_host: str = OLLAMA_HOST,
                 ollama_port: int = OLLAMA_PORT,
                 retriever_top_k: int = RETRIEVER_TOP_K,
                 reranker_top_k: int = RERANKER_TOP_K):
        self.db_host = db_host
        self.db_port = db_port
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k = reranker_top_k
    