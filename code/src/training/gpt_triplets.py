import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import random

# Configuration
MODEL_NAME = "answerdotai/ModernBERT-base"
BATCH_SIZE = 8
MAX_LEN = 128
EPOCHS = 2
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)

# MedQA dataset in English with 4 options
raw = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="train")

# Triplets: (query, positive, negative)
def make_triplet(example):
    q = example["question"]
    correct = example["options"][example["answer_idx"]]
    incorrect = random.choice([
        o for i, o in enumerate(example["options"]) if i != example["answer_idx"]
    ])
    return {"query": q, "positive": correct, "negative": incorrect}

triplets = raw.map(make_triplet)

# PyTorch Dataset
class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer, max_len):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anchor = self.tokenizer(triplet["query"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        positive = self.tokenizer(triplet["positive"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        negative = self.tokenizer(triplet["negative"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return anchor, positive, negative

# Mean pooling for embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Loader
dataset = TripletDataset(triplets, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.TripletMarginLoss(margin=1.0)

# Fine-tuning loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in loader:
        anchor, pos, neg = batch
        for x in [anchor, pos, neg]:
            for k in x:
                x[k] = x[k].squeeze(1).to(DEVICE)

        emb_anchor = mean_pooling(model(**anchor), anchor["attention_mask"])
        emb_pos = mean_pooling(model(**pos), pos["attention_mask"])
        emb_neg = mean_pooling(model(**neg), neg["attention_mask"])

        loss = loss_fn(emb_anchor, emb_pos, emb_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

# Save the fine-tuned model
model.save_pretrained("modernbert-finetuned")
tokenizer.save_pretrained("modernbert-finetuned")