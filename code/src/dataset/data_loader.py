from typing import Union
import datasets as hfds
from sentence_transformers import InputExample
from utils.vprint import vprint
import re
import random
from datasets import load_from_disk

def retrieve_huggingface_dataset(dataset_name, verbose=True):
    """
    Loads a dataset from Hugging Face and returns it.
    
    Args:
        dataset_name (str): Dataset name.
        
    Returns:
        dataset: Loaded dataset, None otherwise.
    """
    vprint(verbose, f"-> Loading Hugging Face dataset {dataset_name}...")
    
    try:
        dataset = hfds.load_dataset(dataset_name, split='all', trust_remote_code=True)
    except Exception as e:
        print(f"!! Error loading dataset: {e}")
        return None
    
    vprint(verbose, f"--> Dataset '{dataset_name}' successfully loaded. Total of documents: {len(dataset)}")
    return dataset


def retrieve_data(dataset_name, hugging_face=True, verbose=True):
    """
    Loads and preprocesses a dataset. This function is a wrapper for other dataset loading functions.
    Args:
        dataset_name (str): Name of the dataset to load.
        hugging_face (bool): If True, uses Hugging Face datasets. If False, uses a custom dataset loader.
        verbose (bool): if True, prints loasing messages.
    Returns:
        dataset: Preprocessed dataset with 'text' field.
    """
    if hugging_face:
        dataset = retrieve_huggingface_dataset(dataset_name, verbose=verbose)

    if dataset is None:
        return None
    
    #dataset = dataset.map(preprocess_data, num_proc=4)
    
    vprint(verbose, f"--> MedRAG/pubmed dataset preprocessed. Total documents: {len(dataset)}")

    return dataset


def preprocess_data(example, verbose=True):
        ''' 
        Append the title and abstract to a single text field
        '''
        if verbose:
            print('**'*100)
            print([k for k in example.keys()])
            print('**'*100)
        example['text'] = f"Title: {example['title']}\n\nAbstract: {example['abstract']}"
        return example
        

def load_pre_train_medrag_pubmed(max_samples: int,
                                 test_size:Union[float, int, None] = 0.2,
                                 val_size:Union[float, int, None] = 0.2,
                                 seed:int = 42,
                                 dataset_name: str = 'MedRAG/pubmed') -> list:
    """
    Loads the dataset, filters it, and prepares it as a list of InoutExample with positive pairs.
    It takes title of the medical text as the query and the 'content' as a positive passage.
    It only uses one positive pair per medical text.
    """
    import time
    print(f"Loading dataset [test_size: {test_size}; val_size: {val_size}]...")
    st = time.time()
    #full_dataset = hfds.load_dataset(dataset_name, split='train[:10000]')
    #full_dataset.save_to_disk(dataset_name+'_cached')
    full_dataset = load_from_disk(dataset_name+'_cached')
    print(f'Dataset loaded. {time.time() - st:.2f}s')

    samples = []
    titles = set()
    for i in range(len(full_dataset)):
        data_point = full_dataset[i]
        if len(samples) % (max_samples // 10) == 0:
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

    if test_size:
        if seed:
            random.seed(seed)
        
        random.shuffle(samples)

        n = len(samples)
        train_size_idx = int((1 - test_size) * n)
        train = samples[:train_size_idx]
        test = samples[train_size_idx:]

        val = []
        queries = {}
        corpus = {}
        relevant_docs = {}
        if val_size:
            val_size_idx = int((1 - val_size) * train_size_idx)
            print(f'train_size_idx: {train_size_idx}, val_size_idx: {val_size_idx}')
            for idx in range(train_size_idx):
                if idx >= val_size_idx:
                    val.append(train[idx])
                    queries[f"q_{idx}"] =  train[idx].texts[0]
                    relevant_docs[f"q_{idx}"] = {f"doc_{idx}"}
                corpus[f"doc_{idx}"] = train[idx].texts[1] 
        
    else:
        train = samples
        val = []
        test = []


    return train, val, test, (queries, relevant_docs, corpus)

