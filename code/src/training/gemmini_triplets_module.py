import os
import time
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


CONFIG = {
    "model_name": 'answerdotai/ModernBERT-base',
    "dataset_name": "pubmed_qa",
    "dataset_config": "pqa_labeled",
    "output_path": './ModernBERT-base-fine-tuned-triplets',
    "epochs": 1,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "max_triplets": 20000
}


def load_and_prepare_data(dataset_name: str, dataset_config: str) -> tuple[list, list]:
    """
    Loads the dataset, filters it, and prepares the positive pairs and the corpus.
    
    Returns:
        tuple[list, list]: A tuple containing the list of positive pairs and the list of the corpus.
    """
    print("--- 1. Loading and preparing data ---")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("Installing rank_bm25...")
        os.system("pip install rank_bm25")

    full_dataset = load_dataset(dataset_name, dataset_config, split='train')
    filtered_dataset = full_dataset.filter(lambda ex: ex['final_decision'] == 'yes')
    
    positive_pairs = []
    corpus = {}
    for example in filtered_dataset:
        question = example['question']
        positive_passage = example['long_answer']
        if question and positive_passage:
            positive_pairs.append({"query": question, "positive": positive_passage})
            if positive_passage not in corpus:
                corpus[positive_passage] = len(corpus)

    corpus_list = list(corpus.keys())
    print(f"Data prepared: {len(positive_pairs)} positive pairs, {len(corpus_list)} documents in the corpus.")
    return positive_pairs, corpus_list

# ------------------------------------------------------------------------------------

def mine_hard_negatives(positive_pairs: list, corpus: list) -> list:
    """
    Uses BM25 to find a "hard negative" for each positive pair.

    Returns:
        list: A list of dictionaries, where each is a triplet.
    """
    print("\n--- 2. Mining 'Hard Negatives' with BM25 ---")
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    training_triplets = []
    start_time = time.time()
    
    for pair in positive_pairs:
        query = pair['query']
        positive_passage = pair['positive']
        
        tokenized_query = query.split(" ")
        top_docs = bm25.get_top_n(tokenized_query, corpus, n=5)
        
        hard_negative_passage = None
        for doc in top_docs:
            if doc != positive_passage:
                hard_negative_passage = doc
                break
        
        if hard_negative_passage:
            training_triplets.append({
                "query": query,
                "positive": positive_passage,
                "negative": hard_negative_passage
            })
            
    end_time = time.time()
    print(f"Mining completed in {end_time - start_time:.2f}s. {len(training_triplets)} triplets created.")
    return training_triplets

# ------------------------------------------------------------------------------------

def train_retriever_model(triplets: list, config: dict):
    """
    Sets up and runs the SentenceTransformer training loop.
    """
    print("\n--- 3. Setting up and running the training ---")
    os.environ["WANDB_DISABLED"] = "true"
    
    model = SentenceTransformer(config["model_name"])
    
    #
    train_samples = []
    for triplet in triplets:
        if len(train_samples) >= config["max_triplets"]:
            break
        train_samples.append(InputExample(
            texts=[triplet['query'], triplet['positive'], triplet['negative']]
        ))

    #
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=config["batch_size"])
    train_loss = losses.TripletLoss(model=model)
    
    print(f"Model '{config['model_name']}' ready. Training with {len(train_samples)} triplets.")
    
    #
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config["epochs"],
        warmup_steps=config["warmup_steps"],
        optimizer_params={'lr': config["learning_rate"]},
        output_path=config["output_path"],
        show_progress_bar=True
    )


if __name__ == "__main__":
    
    positive_pairs, corpus = load_and_prepare_data(
        CONFIG["dataset_name"], 
        CONFIG["dataset_config"]
    )
    
    triplets = mine_hard_negatives(positive_pairs, corpus)
    
    train_retriever_model(triplets, CONFIG)
    
    print(f"Model saved to: {CONFIG['output_path']}")