import colbert_fine_tuning_parallel as trainer

def run_fine_tune():
    CONFIG = {
        "base_model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "output_dir": "models/test_ColBERT-10k-cos-rand42",
        "run_name": "colbert-pubmedqa-finetune",
        # Dataset parasm
        "dataset_path": "data/training/bigbio_pubmed_qa_triplets_rand_42.jsonl",
        "test_size": 0.1,
        "max_datasize": 1112,
        "seed": 42,
        # Training params
        "batch_size": 20,
        "num_train_epochs": 10,
        "learning_rate": 3e-6,
        "num_checkpoints": 5,
        "save_nsteps": 500,
        "load_best_one": True,
    }
    trainer.main_run(CONFIG)

if __name__ == "__main__":
    run_fine_tune()

    