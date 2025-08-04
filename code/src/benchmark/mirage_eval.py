# run_evaluation.py
import requests
import pandas as pd
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import time
import os

# 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 
API_URL = "http://127.0.0.1:8001/{retriever_type}/query"

# 
RETRIEVER_TYPES = ["BM25"]

def query_rag_system(question: str, options: dict, retriever_type: str) -> dict:
    """Envía una única pregunta a nuestra API RAG y devuelve la respuesta."""
    api_url = API_URL.format(retriever_type=retriever_type)
    payload = {
        "query": question,
        "opts": options,
        "is_multi_opt": True
    }
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API '{retriever_type}': {e}")
        return {"answer": "Error", "retrieved_context": []}

def gather_evaluation_data(benchmark_dataset):
    """
    """
    results = []
    total_questions = len(benchmark_dataset)
    
    for i, record in enumerate(benchmark_dataset):
        question = record['question']
        options = record['options']
        short_answer = record['answer']

        if short_answer in options:
            ground_truth_answer = options[short_answer]
        else:
            ground_truth_answer = short_answer
        
        print(f"\nProcesando Pregunta #{i+1}/{total_questions}: '{question}' --> Grund Truth: '{ground_truth_answer}'")
        
        for retriever in RETRIEVER_TYPES:
            print(f"  -> Probando retriever: '{retriever}'...")
            start_time = time.time()
            rag_response = query_rag_system(question, options, retriever)
            end_time = time.time()
            
            results.append({
                "retriever_type": retriever,
                "question": question,
                "answer": rag_response.get('answer', ''),
                "contexts": rag_response.get('retrieved_context', []),
                "ground_truth": ground_truth_answer,
                "response_time_secs": end_time - start_time
            })
    
    print("\n--- Recopilación de Datos Finalizada ---")
    return pd.DataFrame(results)

def run_ragas_evaluation(results_df):
    """
    Toma los resultados recopilados y los evalúa usando Ragas.
    """
    print("\n--- Iniciando Evaluación con Ragas ---")
    
    # Ragas espera un objeto Dataset de Hugging Face
    evaluation_dataset = Dataset.from_pandas(results_df)

    # Definimos las métricas que queremos calcular
    metrics = [
        faithfulness,       # ¿La respuesta se basa en el contexto? (Mide alucinaciones)
        answer_relevancy,   # ¿La respuesta es relevante para la pregunta?
        context_precision,  # ¿Qué tan relevante es el contexto recuperado? (Mide al retriever)
        context_recall,     # ¿El contexto recuperado contiene toda la info para responder?
    ]
    
    # Ejecutamos la evaluación
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
    )

    print("--- Evaluación con Ragas Finalizada ---")
    return result

# --- SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    # 1. Cargar el dataset de benchmark (usaremos un subset para una prueba rápida)
    # Nota: Reemplaza "your-org/mirage-benchmark" con el nombre correcto del dataset en Hugging Face
    print("Cargando dataset de benchmark MIRAGE...")
    try:
        mirage_dataset = load_dataset("MedRAG/MIRAGE", split='all').select(range(10)) # Usamos solo 10 preguntas para la demo
        print(f"Benchmark cargado. Se evaluarán {len(mirage_dataset)} preguntas.")
    except Exception as e:
        print(f"!! No se pudo cargar el dataset MIRAGE. Verifica el nombre o la conexión. Error: {e}")
        exit()

    # 2. Recopilar respuestas de nuestro sistema RAG
    results_df = gather_evaluation_data(mirage_dataset)
    print("\nResultados Recopilados:")
    print(results_df[['retriever_type', 'question', 'response_time_secs']])

    # 3. Evaluar los resultados con Ragas
    evaluation_results = run_ragas_evaluation(results_df)
    
    print("\nResultados de la Evaluación (Scores):")
    # Imprimimos los resultados en un formato legible
    print(evaluation_results.to_pandas())