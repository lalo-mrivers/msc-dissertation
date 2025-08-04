# Este script es el bueno
import os
from typing import Literal, Union
import torch
import torch.distributed as dist
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import re
from datasets import load_from_disk



import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    '''Initialize the group of processes for DPP and returns the range and world size'''
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    return dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    '''Cleans the processes group'''
    dist.destroy_process_group()


def load_pre_train_medrag_pubmed(max_samples: int,
                                 dataset_name: str = 'MedRAG/pubmed',
                                 cached_dataset_path: str = None,
                                 save_cache: bool = False,
                                 subset_samples: int = 0,
                                 is_main_process: bool = True) -> list:
    """
    Loads the dataset, filters it, and prepares it as a list of InoutExample with positive pairs.
    It takes title of the medical text as the query and the 'content' as a positive passage.
    It only uses one positive pair per medical text.
    """
    import time
    if is_main_process:
        print(f"Loading dataset {dataset_name}...")

    subset_samples = subset_samples if subset_samples > max_samples else max_samples
    st = time.time()
    if cached_dataset_path and os.path.exists(cached_dataset_path):
        if is_main_process:
            print('Loading data from cache local')
        full_dataset = load_from_disk(cached_dataset_path)
    else:
        if is_main_process:
            print('Loading data from huggingface')
        full_dataset = load_dataset(dataset_name, split=f'train[:{subset_samples}]')

        if save_cache:
            full_dataset.save_to_disk(cached_dataset_path)

    if is_main_process:
        print(f'Dataset loaded. {time.time() - st:.2f}s')

    samples = []
    titles = set()
    for i in range(len(full_dataset)):
        data_point = full_dataset[i]
        if is_main_process and len(samples) % (max_samples // 10) == 0:
            print(f'{len(samples) / (max_samples // 100)}%')
        
        if len(samples) >= max_samples:
            break
        
        query = re.sub(r'^[@$!%&/=\?\[\]]+', '', data_point.get('title'), count=1)
        query = re.sub(r'](?=[^\]]*$)', '', query, count=1)

        positive_passage = data_point.get('content')

        if len(query) > 0 and not query[0].isalnum():
            continue
        
        if len(positive_passage) > 0 and not positive_passage[0].isalnum():
            continue
        
        if query and positive_passage and query not in titles:
            samples.append(InputExample(texts=[query, positive_passage]))
            titles.add(query)

    return samples


def load_training_data_triplets(file_name: str, max_samples: int, is_main_process:bool = True):
    '''
    '''

    full_dataset = load_dataset('json', data_files={'train': file_name}, split='train')
    examples = []
    for i, row in enumerate(full_dataset):
        if max_samples and len(examples) >= max_samples:
            break
        query = str(row["query"]).strip()
        positive = str(row["positive"]).strip()
        negative = str(row["negative"]).strip()
        if query and positive:
            if negative:
                examples.append(InputExample(texts=[query, positive, negative]))
            else:
                examples.append(InputExample(texts=[query, positive]))
    
    if is_main_process:
        print(f"Loaded {len(examples)} examples for training")
    
    return examples

def load_positive_pairs(training_filename:str,
                        max_samples: int,
                        is_main_process: bool = True):
    """
    Loads the dataset, filters it, and prepares it as a list of InoutExample with positive pairs.
    It takes title of the medical text as the query and the 'content' as a positive passage.
    It only uses one positive pair per medical text.
    """
    if is_main_process:
        print("Loading dataset")
    train = load_training_data_triplets(file_name = training_filename,
                                    max_samples = max_samples,
                                    is_main_process = is_main_process)
    #train, val, test, (queries, relevant_docs, corpus)

    queries = {}
    corpus = {}
    relevant_docs = {}

    if is_main_process:
        for idx, example in enumerate(train):
            qid = f"q_{idx}"
            docid = f"doc_{idx}"
            queries[qid] = example.texts[0]
            corpus[docid] = example.texts[1]
            relevant_docs[qid] = {docid}
    
    return train, [], [], (queries, relevant_docs, corpus)

# ------------------------------------------------------------------------------------

def train_contrastive_model(train_samples: list[InputExample], eval_data: tuple,
                            model_name: str,
                            sim_fun: callable,
                            neg_sampler_method: Literal['triplet', 'ibn'],
                            batch_size: int,
                            checkpoint_path: str,
                            epochs: int,
                            warmup_steps: int,
                            optimizer_params: dict,
                            output_path: str,
                            rank: int, world_size: int,
                            evaluation_steps: int = 1000,
                            checkpoint_save_steps: int = 1000,
                            checkpoint_save_total_limit: int = 5):
    """
    In-batch contrastive training with InfoNCELoss.
    """
    is_main_process = (rank == 0)
    if is_main_process:
        print("\n--- 2. Setting up and running contrastive training ---")

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   
    model = SentenceTransformer(model_name)
    
    # importtant to set shuffle=True
    train_sampler = DistributedSampler(train_samples, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(train_samples, sampler=train_sampler, batch_size=batch_size)
    
    if neg_sampler_method == 'ibn':
        # In Batch Contrastive Learning
        train_loss = losses.MultipleNegativesRankingLoss(model=model, similarity_fct=sim_fun)
    elif neg_sampler_method == 'triplet':
        train_loss = losses.TripletLoss(model=model, distance_metric=sim_fun, triplet_margin=1) 
    else:
        raise Exception('No valid negative sampling method')
    
    evaluator = None
    if is_main_process:
        queries, relevant_docs, corpus = eval_data
        evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name='dev')
    
    if is_main_process:
        #print(f"Using DDP with rank {dist.get_rank()}. Training model: {config['model_name']}")
        print(f"Model '{model_name}' ready.")
        print(f"Train Loss '{train_loss}' ready.")
        print(f"Resuming from '{checkpoint_path}'.")
        print(f"Batch size '{batch_size}'.")
        print(f"Loss function: InfoNCELoss")
    
    model.fit(
        train_objectives = [(train_dataloader, train_loss)],
        epochs = epochs,
        warmup_steps = warmup_steps,
        evaluator = evaluator,
        evaluation_steps = evaluation_steps if is_main_process else 0,
        optimizer_params = optimizer_params,#{'lr': config["learning_rate"]},
        output_path = output_path,
        show_progress_bar = is_main_process,
        save_best_model = True, # Saves the model with the best evaluation score
        checkpoint_path = checkpoint_path,
        checkpoint_save_steps = checkpoint_save_steps,
        checkpoint_save_total_limit = checkpoint_save_total_limit,
        resume_from_checkpoint = True,
        use_amp = True,
        #**{"resume_from_checkpoint": True}
    )

    if is_main_process:
        print(f"\n Process completed! Model saved to: {output_path}")


#if __name__ == "__main__":
def main_run(config: dict):
    rank, world_size = setup_ddp()
    is_main_process = (rank == 0)

    if is_main_process:
        for k,v in config.items():
            print(f'{k}: {v}')

    try:
        train, eval, test, eval_data = load_positive_pairs(training_filename=config['training_data'],
                                                            max_samples = config["max_samples"],
                                                            #dataset_name = config["dataset_name"],
                                                            #cached_dataset_path = config["cached_dataset_path"],
                                                            #save_cache = config["save_cache"],
                                                            #subset_samples = config["subset_samples"],
                                                            is_main_process = is_main_process)
        train_contrastive_model(train_samples = train,
                                eval_data = eval_data,
                                model_name = config['model_name'],
                                sim_fun = config['sim_fun'],
                                neg_sampler_method = config['negative_sampler'],
                                batch_size = config['batch_size'],
                                checkpoint_path = config['checkpoint_path'],
                                epochs = config['epochs'],
                                warmup_steps = config['warmup_steps'],
                                optimizer_params = config['optimizer_params'],
                                output_path = config['output_path'],
                                rank = rank, world_size = world_size,
                                evaluation_steps = config['evaluation_steps'],
                                checkpoint_save_steps = config['checkpoint_save_steps'],
                                checkpoint_save_total_limit = config['checkpoint_save_total_limit'])
        if is_main_process:
            print(f"\n✅ Process completed: {config['output_path']}")
    finally:
        cleanup_ddp() # Ensure processes are cleaned up correctly
    

if __name__ == "__main__":
    CONFIG = {
        "model_name": 'answerdotai/ModernBERT-base',
        "training_data": 'data/training/MedRAG_pubmed_triplets_rand_42_20k.jsonl',
        "negative_sampler": 'triplet',
        #"dataset_name": "MedRAG/pubmed",
        #"save_cache": True,
        #"cached_dataset_path": 'MedRAG/pubmed_100k_cached',
        #"subset_samples": 100000,
        "output_path": './models/rtx_ModernBERT-10k-cos-ibnl',
        "epochs": 100,
        "batch_size": 50,#40,  
        "optimizer_params" : {"lr": 1e-5},
        "warmup_steps": 100,
        "max_samples": 12500, 
        "sim_fun": util.cos_sim,
        "checkpoint_path": './checkpoints/rtx_ModernBERT-10k-cos-ibnl_chkpt', # New checkpoint path
        "evaluation_steps": 1000,
        "checkpoint_save_steps": 1000,
        "checkpoint_save_total_limit":5
    }
    main_run(CONFIG)