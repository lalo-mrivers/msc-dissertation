import os
import pandas as pd
#import psycopg2
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import datasets as hfds
import torch

DS_NAME='fmenol/acr-appro-3opts-v2'

Q_CLIENT = QdrantClient(
        host="uhtred.inf.ed.ac.uk",
        port=6333
    )

def load_dataset():
    dataset = hfds.load_dataset(DS_NAME, split='all', trust_remote_code=True)
    return dataset.to_pandas()


def load_st_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def create_collection(collection_name, vector_dim):
    
    Q_CLIENT.recreate_collection(
            collection_name = collection_name,
            vectors_config = models.VectorParams(size = vector_dim,
                                                 distance = models.Distance.COSINE 
            )
        )
    list_collection()

def list_collection():
    collections = Q_CLIENT.get_collections().collections
    print("Colections:")
    for col in collections:
        print(f"- {col.name}")


def delete_collection(collection_name):
    try:
        Q_CLIENT.delete_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' deleted.")
    except UnexpectedResponse as e:
        print(f"Error deleting the collection '{collection_name}': {e}")


def embed_st_data(st_model, collection_base_name):
    
    model = load_st_model(st_model)

    #*** Type one ***
    embed_type_name = collection_base_name+'_v1'
    delete_collection(embed_type_name)
    create_collection(embed_type_name, model.get_sentence_embedding_dimension())

    df = load_dataset()[['Condition', 'Variant', 'Procedure', 'Appropriateness Category']].drop_duplicates()
    print(f'** Procesando: {embed_type_name}')
    batch_size = 500
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        print(f'Batch [{start} - {end} / {len(df)}]')

        batch_df = df.iloc[start:end].copy()
        
        batch_ids = batch_df.index.tolist()

        batch_df['text'] = batch_df['Variant']
        batch_record = batch_df[['Condition', 'Variant', 'Procedure', 'Appropriateness Category', 'text']]\
            .rename(columns={'Appropriateness Category': 'approp'})\
            .to_dict(orient='records')

        embeddins = model.encode(batch_df['text'].tolist(), show_progress_bar=True, batch_size=128)

        Q_CLIENT.upsert(
                    collection_name=embed_type_name,
                    points=models.Batch(
                        #ids=list(range(start_batch, end_batch)), 
                        ids=batch_ids, 
                        vectors=embeddins.tolist(),
                        payloads=batch_record
                    ),
                    wait=True
                )
    """
    for idx, row in df.iterrows():
        # Embed the combined condition and variant text
        text_to_encode = f"{row['Variant']}"
        
        embedding = model.encode(text_to_encode, batch_size=128)

        Q_CLIENT.upsert(
                        collection_name=embed_type_name,
                        points=[
                            models.PointStruct(
                                id=idx,
                                vector=embedding.tolist(),
                                payload={"Condition": row['Condition'],
                                         "Variant": row['Variant'],
                                         "Procedure": row['Procedure'],
                                         "approp": row['Appropriateness Category'],
                                         "text": text_to_encode}
                            )
                        ],
                        wait=True
                    )
    """
    
    #*** Type TWO ***
    embed_type_name = collection_base_name+'_v2'
    delete_collection(embed_type_name)
    create_collection(embed_type_name, model.get_sentence_embedding_dimension())
    
    print(f'** Procesando: {embed_type_name}')
    batch_size = 500
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        print(f'Batch [{start} - {end} / {len(df)}]')

        batch_df = df.iloc[start:end].copy()
        
        batch_ids = batch_df.index.tolist()

        batch_df['text'] = "Condition: " + batch_df['Condition'] + " | Clinical Scenario: " + batch_df['Variant']
        batch_record = batch_df[['Condition', 'Variant', 'Procedure', 'Appropriateness Category', 'text']]\
            .rename(columns={'Appropriateness Category': 'approp'})\
            .to_dict(orient='records')

        embeddins = model.encode(batch_df['text'].tolist(), show_progress_bar=True, batch_size=128)

        Q_CLIENT.upsert(
                    collection_name=embed_type_name,
                    points=models.Batch(
                        #ids=list(range(start_batch, end_batch)), 
                        ids=batch_ids, 
                        vectors=embeddins.tolist(),
                        payloads=batch_record
                    ),
                    wait=True
                )
    """
    for idx, row in df.iterrows():
        # Embed the combined condition and variant text
        text_to_encode =  f"Condition: {row['Condition']} | Clinical Scenario: {row['Variant']}"
        
        embedding = model.encode(text_to_encode, batch_size=128)

        Q_CLIENT.upsert(
                        collection_name=embed_type_name,
                        points=[
                            models.PointStruct(
                                id=idx,
                                vector=embedding.tolist(),
                                payload={"Condition": row['Condition'],
                                         "Variant": row['Condition'],
                                         "Procedure": row['Procedure'],
                                         "approp": row['Appropriateness Category'],
                                         "text": text_to_encode}
                            )
                        ],
                        wait=True
                    )
    """
        
    
def evaluate_exact_match_retrieval(st_model, collection_name):
    """
    Evaluates the embedding system's ability to retrieve exact matches
    for condition-variant pairs through vector similarity search.
    """
    print("Initializing embedding evaluation...")
    model = load_st_model(st_model)
    # Load reference dataset

    df = load_dataset()

    unique_scenarios = df[['Condition', 'Variant']].drop_duplicates().reset_index(drop=True)
    
    print(f"Evaluating {len(unique_scenarios)} unique condition-variant pairs...")
    
    # Initialize results storage
    evaluation_results = []
    
    for idx, row in unique_scenarios.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(unique_scenarios)} ({idx/len(unique_scenarios)*100:.1f}%)")
        
        # Query using only the variant (clinical scenario)
        query_text = row['Variant']
        
        # Perform vector similarity search
        query_embedding = model.encode(query_text, batch_size=128)
        
        search_result = Q_CLIENT.search(collection_name=collection_name,
                                              query_vector=query_embedding,
                                              limit=3,
                                              with_payload = True)
        result = search_result[0].payload
        distance = search_result[0].score
        if result:
            retrieved_condition = result['Condition']
            retrieved_variant = result['Variant']
            
            # Determine exact match
            exact_match = (retrieved_condition == row['Condition'] and 
                            retrieved_variant == row['Variant'])
            
            evaluation_results.append({
                'query_id': idx + 1,
                'true_condition': row['Condition'],
                'true_variant': row['Variant'],
                'retrieved_condition': retrieved_condition,
                'retrieved_variant': retrieved_variant,
                'exact_match': 'Yes' if exact_match else 'No',
                'euclidean_distance': round(distance, 6)
            })
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(evaluation_results)
    
    # Generate evaluation metrics
    total_queries = len(results_df)
    exact_matches = len(results_df[results_df['exact_match'] == 'Yes'])
    accuracy = exact_matches / total_queries if total_queries > 0 else 0
    
    # Create summary statistics
    summary_stats = {
        'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_queries': total_queries,
        'exact_matches': exact_matches,
        'accuracy': round(accuracy, 4),
        'mean_distance_exact_matches': round(results_df[results_df['exact_match'] == 'Yes']['euclidean_distance'].mean(), 6) if exact_matches > 0 else None,
        'mean_distance_all': round(results_df['euclidean_distance'].mean(), 6)
    }
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'embedding_evaluation_results_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    
    # Save summary
    summary_filename = f'embedding_evaluation_summary_{timestamp}.csv'
    pd.DataFrame([summary_stats]).to_csv(summary_filename, index=False)
    
    # Print formal evaluation report
    _print_evaluation_report(summary_stats, results_filename, summary_filename)
    
    return results_df, summary_stats

def _print_evaluation_report(stats, results_file, summary_file):
    """Print formal evaluation report."""
    print("\n" + "="*60)
    print("EMBEDDING SYSTEM EVALUATION REPORT")
    print("="*60)
    print(f"Evaluation Date: {stats['evaluation_timestamp']}")
    print(f"Model: neuml/pubmedbert-base-embeddings")
    print(f"Embedding Method: Combined Condition + Variant")
    print(f"Query Method: Variant Only")
    print(f"Distance Metric: Euclidean Distance (L2)")
    print(f"Evaluation Method: Cross-Modal Exact Match Retrieval")
    print()
    print("PERFORMANCE METRICS:")
    print(f"  Total Test Cases: {stats['total_queries']:,}")
    print(f"  Exact Matches: {stats['exact_matches']:,}")
    print(f"  Accuracy (Top-1): {stats['accuracy']:.2%}")
    print(f"  Mean Distance (Exact): {stats['mean_distance_exact_matches']}")
    print(f"  Mean Distance (All): {stats['mean_distance_all']}")
    print()
    print("OUTPUT FILES:")
    print(f"  Detailed Results: {results_file}")
    print(f"  Summary Statistics: {summary_file}")
    print()
    
    if stats['accuracy'] == 1.0:
        print("ASSESSMENT: Perfect retrieval performance.")
    elif stats['accuracy'] >= 0.95:
        print("ASSESSMENT: Excellent retrieval performance.")
    elif stats['accuracy'] >= 0.90:
        print("ASSESSMENT: Good retrieval performance.")
    elif stats['accuracy'] >= 0.80:
        print("ASSESSMENT: Acceptable retrieval performance.")
    else:
        print("ASSESSMENT: Sub-optimal retrieval performance requires investigation.")
    
    print("="*60)



def main():
    embed_st_data('neuml/pubmedbert-base-embeddings', 'acr_pumbmedbert')

    #evaluate_exact_match_retrieval('neuml/pubmedbert-base-embeddings', 'acr_pumbmedbert_v2')


if __name__ == "__main__":
    main()
    

