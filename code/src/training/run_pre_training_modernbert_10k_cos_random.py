import modernbert_pre_trainer_parallel as trainer
from sentence_transformers import util

def run_pre_training():
    outname = 'ModernBERT-10k-cos-rand42'
    CONFIG = {
        "model_name": 'answerdotai/ModernBERT-base',
        "training_data": 'data/training/MedRAG_pubmed_triplets_rand_42_20k.jsonl',
        "negative_sampler": 'triplet',
        "output_path": f'./models/{outname}',
        "epochs": 100,
        "batch_size": 40,
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
    run_pre_training()