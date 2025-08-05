# ModernBERT + ColBERT for Clinical RAG Enhancement

## Project Overview

This project aims to improve clinical Retrieval-Augmented Generation (RAG) for knowledge-intensive Question Answering (QA) tasks by combining a state-of-the-art bi-encoder (**ModernBERT**) with a late-interaction re-ranker (**ColBERT**).

---

## External Components & Prerequisites

This project relies on several external components. Please ensure they are properly installed and running before proceeding.

* **QDrant**: A QDrant service instance is required for vector storage and retrieval. [https://github.com/qdrant/qdrant](https://github.com/qdrant/qdrant)
* **Ollama**: The project uses the Ollama service for the generator (LLM) implementation. Experiments were conducted using the `llama3:8B` model. [https://ollama.com/download/linux](https://ollama.com/download/linux)
* **PyLate**: This library is used to simplify the fine-tuning of ColBERT V2.
    * **Repository**: [https://github.com/lightonai/pylate](https://github.com/lightonai/pylate)

* **MIRAGE**: Benchmark toolkit [https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/](https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/)

---

## File Structure

The project is organized as follows:
```
Project/
├── data/
│   ├── benchmark/out/
│   ├── mirage/
│   └── training/
└── models/
```
---

##  Core Tasks & Workflow

The project is divided into four main tasks, from data preparation to final evaluation.

### 1. Data Preparation

This module prepares the training data into a triplet format (`query`, `positive passage`, `negative passage`) to ensure consistency across all experiments.

* **Script**: `code/src/training/prepare_data_triplets.py`

### 2. Model Training

This directory contains all the experiments related to the pre-training and fine-tuning of the ModernBERT and ColBERT models.

To train the models we define tthe following scripts

* Pre-train modernBERT:
```
torchrun --nproc_per_node=1 code/src/training/test_run_pre_training_modernbert_10k_cos_random.py
```


* Fine-tune modernBERT:
```
torchrun --nproc_per_node=1 code/src/training/test_run_fine_tune_modernbert_10k_cos_random.py
```


* Fine-tune colBERT:
```
torchrun --nproc_per_node=1 code/src/training/test_run_fine_tune_colbert_10k_random.py
```

* **Location**: `code/src/training/`

### 3. Corpus Indexing

This script indexes the corpus using the corresponding trained models. The `train_hf_ids.csv` file contains the document IDs from the MedRag/pubmed dataset that need to be indexed.

* **Script**: `code/src/indexer.py`

### 4. Testing with MIRAGE Benchmark

This final stage executes a comprehensive evaluation of all models using the MIRAGE benchmark toolkit.

* **Evaluation Execution**: `code/src/mirage_evaluation.py`
    * This script runs the evaluation for all defined models, strictly adhering to the MIRAGE toolkit's standards.
* **Results Parsing**: `code/src/mirage_parse_output.py`
    * This script collects the output from the MIRAGE benchmark and presents the final, parsed results.


A testing script to run 1. Data preparation and 4. Testing with MIRAGE is in `code\src\main.py`