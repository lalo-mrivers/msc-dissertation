import modernbert_fine_tuning_parallel as trainer
from sentence_transformers import util

def run_fine_tune():
    outname = 'Clinical_ModernBERT-tuned-10k-cos-ibns'
    CONFIG = {
        "model_name": 'Simonlee711/Clinical_ModernBERT',
        "training_data": 'data/training/bigbio_pubmed_qa_triplets_rand_42.jsonl',
        "negative_sampler": 'ibn',
        "output_path": f'./models/{outname}',
        "epochs": 100,
        "batch_size": 32,
        "optimizer_params" : {"lr": 1e-5},
        "warmup_steps": 100,
        "max_samples": 10000, # 
        "sim_fun": util.cos_sim,
        "checkpoint_path": f'./checkpoints/{outname}_chkpt', # New checkpoint path
        "evaluation_steps": 1000,
        "checkpoint_save_steps": 1000,
        "checkpoint_save_total_limit":5
    }
    trainer.main_run(CONFIG)


if __name__ == "__main__":
    run_fine_tune()
    