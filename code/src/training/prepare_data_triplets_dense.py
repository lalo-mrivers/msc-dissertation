import multiprocessing
import os
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import time
import torch
from transformers import AutoModel, AutoTokenizer

# --- Parámetros de Configuración ---
DATASET_NAME = 'data/training/bigbio_pubmed_qa_triplets_rand_42.jsonl'#"bigbio/pubmed_qa"  # Un dataset de ejemplo con noticias.
TEXT_COLUMN = "positive"      # El campo que queremos indexar.
MODEL_NAME = "models/ModernBERT-tuned-10k-cos-ibns" # Modelo eficiente y multilingüe.
FAISS_INDEX_PATH = "data/training/ModernBERT-tuned-10k-cos-ibns.bin" # Dónde guardar el índice.

# Número de procesadores a usar. Usa os.cpu_count() para el máximo disponible.
NUM_PROC = os.cpu_count() // 2 or 1 # Usar la mitad de los cores disponibles o 1 si no se puede determinar.
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

    ## ----------------------------------------------------------------
    ## NUEVO: Paso 6 - Guardar los textos originales en orden
    ## ----------------------------------------------------------------
    print(f"\n6. Guardando los textos originales en '{TEXT_DATA_PATH}'...")
    # Extraemos solo la columna de texto del dataset original.
    # El orden se preserva automáticamente.
    text_corpus = dataset[text_column]
    
    with open(TEXT_DATA_PATH, 'w', encoding='utf-8') as f:
        for text_line in text_corpus:
            # Escribimos cada texto en una nueva línea.
            # Reemplazamos los saltos de línea dentro del texto para evitar archivos corruptos.
            f.write(text_line.replace('\n', ' ') + '\n')
    print(f"   -> Se han guardado {len(text_corpus)} textos.")
    ## ----------------------------------------------------------------
    
    # Devolvemos el corpus de texto cargado en memoria para el ejemplo de búsqueda.
    return index, text_corpus

# --- Función de Búsqueda Modificada ---
def search_and_retrieve(query_text: str, model, index, text_data, k: int = 5):
    """
    Toma un texto de consulta, encuentra los textos más similares y los recupera
    del archivo de texto o de la lista en memoria.
    """
    print(f"\nBuscando los {k} textos más similares para: '{query_text}'")
    
    query_embedding = model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    
    print("\nResultados encontrados:")
    for i, idx in enumerate(indices[0]):
        # Ahora recuperamos el texto de nuestra lista 'text_data'
        retrieved_text = text_data[int(idx)]
        print(f"  {i+1}. (Índice: {idx}, Distancia: {distances[0][i]:.4f}) - '{retrieved_text[:120]}...'")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    print("--- Inicio del Proceso de Indexación ---")
    
    faiss_index, original_texts = create_faiss_index(
        dataset_name=DATASET_NAME,
        text_column=TEXT_COLUMN,
        model_name=MODEL_NAME
    )
    
    print("\n--- Proceso Finalizado Exitosamente ---")
    
    # --- Ejemplo de Búsqueda ---
    # En un script real, cargarías el índice y el archivo de texto aquí.
    # print("\nCargando recursos para la búsqueda...")
    # faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    # with open(TEXT_DATA_PATH, 'r', encoding='utf-8') as f:
    #     original_texts = f.readlines()
    
    query_model = SentenceTransformer(MODEL_NAME)
    
    search_and_retrieve(
        query_text="What are the latest developments in space exploration?",
        model=query_model,
        index=faiss_index,
        text_data=original_texts, # Usamos la lista de textos
        k=3
    )