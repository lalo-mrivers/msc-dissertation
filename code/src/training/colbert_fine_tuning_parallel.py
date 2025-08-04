import torch
import os
from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from transformers.trainer_utils import get_last_checkpoint


from pylate import evaluation, losses, models, utils

def load_training_data(dataset_path: str, test_size: float, max_datasize: int = None, seed: int = 42):
    dataset_split_slice = f'train[:{max_datasize}]' if max_datasize else 'train'
    full_dataset = load_dataset('json', data_files={'train': dataset_path}, split=dataset_split_slice)
    splits = full_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    print(f"Dataset loaded: {len(train_dataset)} train datapoints, {len(eval_dataset)} eval datapoints")

    return splits

def train_colbert_model(base_model_name: str,
                        output_dir: str,
                        run_name: str,
                        dataset: DatasetDict,
                        batch_size: int,
                        num_train_epochs: int,
                        learning_rate: float,
                        save_num_checkpoints: int = 5,
                        save_every_n_steps: int = 2000,
                        load_best_model_at_end: bool = True,):
    """
    Function to set up and train a ColBERT model.

    Args:
        base_model_name (str): Name of the base Hugging Face model.
        output_dir (str): Directory to save the model and checkpoints.
        run_name (str): Name for the training run (used in logs).
        dataset_path (str): Path to the JSONL triplets dataset file.
        dataset_split_slice (str): String to select a portion of the dataset (e.g., 'train[:1000]').
        test_size (float): Proportion of the dataset to use for evaluation.
        batch_size (int): Batch size for training and evaluation.
        num_train_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    """
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]


    print("--- 2. Defining the Model and Training Components ---")
    
    last_checkpoint = get_last_checkpoint(output_dir) if os.path.exists(output_dir) else None
    
    if last_checkpoint:
        print(f"Checkpoint found in {last_checkpoint}. Resuming training.")
        model = models.ColBERT(model_name_or_path=last_checkpoint)
    else:
        print(f"No checkpoint found. Starting new training with base: {base_model_name}")
        model = models.ColBERT(model_name_or_path=base_model_name)

    
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        print("Compiling the model with torch.compile() for speed.")
        model = torch.compile(model)

    
    train_loss = losses.Contrastive(model=model)
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name=run_name,
    )

    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        save_total_limit= save_num_checkpoints,
        save_steps= save_every_n_steps,
        load_best_model_at_end= load_best_model_at_end, # Loads the best model at the end of training
    )

    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    print("Components ready. Trainer initialized.")
    print("-" * 30)

    
    print(" Starting the training process...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print(f" Training complete. Final model saved in: {output_dir}")

def main_run(config):
    dataset = load_training_data(dataset_path = config['dataset_path'],
                                 test_size = config['test_size'],
                                 max_datasize = config['max_datasize'],
                                 seed = config['seed'])
    
    train_colbert_model(base_model_name = config['base_model_name'],
                        output_dir = config['output_dir'],
                        run_name = config['run_name'],
                        dataset = dataset,
                        batch_size = config['batch_size'],
                        num_train_epochs = config['num_train_epochs'],
                        learning_rate = config['learning_rate'], 
                        save_num_checkpoints = config['num_checkpoints'],
                        save_every_n_steps = config['save_nsteps'],
                        load_best_model_at_end = config['load_best_one'])