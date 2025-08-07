import multiprocessing
import os
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time
import torch
from transformers import AutoModel, AutoTokenizer


DATASET_NAME = 'data/training/bigbio_pubmed_qa_triplets_rand_42.jsonl'#"bigbio/pubmed_qa"  # Dataset to index
TEXT_COLUMN = "positive"      # Field to index
MODEL_NAME = "models/ModernBERT-tuned-10k-cos-ibns" # Base model
FAISS_INDEX_PATH = "data/training/ModernBERT-tuned-10k-cos-ibns.bin"

NUM_PROC = os.cpu_count() // 2 or 1
TEXT_DATA_PATH = "data/training/indexed_texts.txt" 

def create_faiss_index(dataset_name: str, text_column: str, model_name: str):
    """
    Carga un dataset, genera embeddings, crea un índice FAISS y guarda los textos originales.
    """
    print(f"1. Cargando el modelo de embeddings '{model_name}'...")
    model = SentenceTransformer(model_name, device="cpu")
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"\n2. Cargando el dataset '{dataset_name}'...")
    #dataset = load_dataset(dataset_name, split="train")
    dataset = load_dataset('json', data_files={'train': dataset_name}, split='train')

    print(f"\n3. Generando embeddings en paralelo usando {NUM_PROC} procesos...")
    
    def generate_embeddings_batch(batch):
        embeddings = model.encode(batch[text_column], convert_to_numpy=True, show_progress_bar=False)
        return {"embeddings": embeddings}

    start_time = time.time()
    dataset_with_embeddings = dataset.map(
        generate_embeddings_batch,
        batched=True,
        batch_size=256,
        num_proc=NUM_PROC
    )
    end_time = time.time()
    print(f"   -> Embeddings generados en {end_time - start_time:.2f} segundos.")

    print("\n4. Creando y poblando el índice FAISS...")
    all_embeddings = np.array(dataset_with_embeddings["embeddings"]).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    
    print(f"   - Indexando {len(all_embeddings)} vectores de dimensión {embedding_dim}...")
    index.add(all_embeddings)
    
    print(f"   - ¡Indexación completa! Total de vectores en el índice: {index.ntotal}")

    print(f"\n5. Guardando el índice en '{FAISS_INDEX_PATH}'...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"\n6. Guardando los textos originales en '{TEXT_DATA_PATH}'...")
    
    text_corpus = dataset[text_column]
    
    with open(TEXT_DATA_PATH, 'w', encoding='utf-8') as f:
        for text_line in text_corpus:
            f.write(text_line.replace('\n', ' ') + '\n')
    print(f"   -> Se han guardado {len(text_corpus)} textos.")

    return index, text_corpus


def search_and_retrieve(query_text: str, model, index, text_data, k: int = 5):
    """
    """
    print(f"\nRetrieving {k} documents for: '{query_text}'")
    
    query_embedding = model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    print("\nResults...:")
    for i, idx in enumerate(indices[0]):
        retrieved_text = text_data[int(idx)]
        print(f"  {i+1}. (index: {idx}, score: {distances[0][i]:.4f}) - '{retrieved_text[:120]}...'")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    print("Indexing..")
    
    faiss_index, original_texts = create_faiss_index(
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        model_name=MODEL_NAME
    )
    
    print('Finished')
    
    query_model = SentenceTransformer(MODEL_NAME)
    
    search_and_retrieve(
        query_text="What are the latest developments in space exploration?",
        model=query_model,
        index=faiss_index,
        text_data=original_texts,
        k=3
    )