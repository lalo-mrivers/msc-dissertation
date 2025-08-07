import abc
import argparse
import json
import os
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import multiprocessing
import random
from typing import Literal


# Number of subprocess. os.cpu_count() we use all cpus available.
DEFAULT_NUM_PROCESSES = os.cpu_count() // 2


#**********************************************************************#
class NegativeSelector(abc.ABC):

    def __init__(self, corpus):
        self.corpus = corpus

    @abc.abstractmethod
    def getNegative(self, query, positive):
        pass

class RandomNegativeSelector(NegativeSelector):

    def __init__(self, corpus, seed):
        super().__init__(corpus)
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def getNegative(self, query, positive):
        top_passages = self._rng.sample(self.corpus, k=10)
        # Takes the first passage that is NOT the same as the positive passage
        negative_passage = None
        for passage in top_passages:
            if passage != positive:
                negative_passage = passage
                break

        return negative_passage
    
class BM25NegativeSelector(NegativeSelector):
    def __init__(self, corpus):
        super().__init__(corpus)
        print("Indexing with BM25...")
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index ready.")

    def getNegative(self,  query, positive):
        tokenized_query = query.split(" ")
        top_passages = self.bm25.get_top_n(tokenized_query, self.corpus, n=10)
        # Takes the first passage that is NOT the same as the positive passage
        negative_passage = None
        for passage in top_passages:
            if passage != positive:
                negative_passage = passage
                break

        return negative_passage
#*************


_selector = None
_out_nm_query = None
_out_nm_positive = None
_out_nm_negative = None

def init_worker(#method_name:str, corpus, seed,
                selector,
                      out_nm_query: str,
                      out_nm_positive: str,
                      out_nm_negative: str
                      ):
    global _selector, _out_nm_query, _out_nm_positive, _out_nm_negative
    _selector = selector
    _out_nm_query = out_nm_query
    _out_nm_positive = out_nm_positive
    _out_nm_negative = out_nm_negative
    '''
    if method_name == 'random':
        _selector = RandomNegativeSelector(corpus=corpus, seed=seed)
    elif method_name == 'bm25':
        _selector = BM25NegativeSelector(corpus=corpus)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    '''
    


def find_negative_wrapper(single_pair):
    return find_negative(pair = single_pair,
                            method = _selector, 
                            out_nm_query = _out_nm_query,
                            out_nm_positive = _out_nm_positive,
                            out_nm_negative = _out_nm_negative)
    


def find_negative(pair, method: NegativeSelector, 
                      out_nm_query: str = 'query',
                      out_nm_positive: str = 'positive',
                      out_nm_negative: str = 'negative'):
    question = pair[out_nm_query]
    positive_passage = pair[out_nm_positive]

    negative_passage = method.getNegative(question, positive_passage)    
    if negative_passage:
        return {
            out_nm_query: question,
            out_nm_positive: positive_passage,
            out_nm_negative: negative_passage
        }
    return None

def build_possitive_pairs(dataset:(DatasetDict | Dataset | IterableDatasetDict | IterableDataset),
                      in_nm_query: str,
                      in_nm_positive: str,
                      out_nm_query: str = 'query',
                      out_nm_positive: str = 'positive') -> list[dict]:
    original_pairs = []
    for example in dataset:
        query = example[in_nm_query]
        positive_passage = example[in_nm_positive]
        if query and positive_passage:
            if not isinstance(positive_passage, str):
                if  isinstance(positive_passage, list):
                    positive_passage = ' '.join(positive_passage)
                else:
                    continue

            original_pairs.append({
                out_nm_query: query,
                out_nm_positive: positive_passage
            })

    return original_pairs


def generate_triplets(output_filename: str,
                      dataset:(DatasetDict | Dataset | IterableDatasetDict | IterableDataset),
                      in_nm_query: str,
                      in_nm_positive: str,
                      out_nm_query: str = 'query',
                      out_nm_positive: str = 'positive',
                      out_nm_negative: str = 'negative',
                      method: Literal['random', 'bm25', 'dense'] = 'random',
                      seed: int=None,
                      max_samples: int=None,
                      num_process: int = DEFAULT_NUM_PROCESSES):
    
    #Load existing data if any
    processed_queries = set()
    if os.path.exists(output_filename):
        print(f"File '{output_filename}' already exists. Loading processed datapoints.")
        # Process line by line in order to let the process append lines
        with open(output_filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_queries.add(data[out_nm_query])
                except json.JSONDecodeError:
                    continue
        print(f"{len(processed_queries)} triplets found. Continuing process.")

    original_pairs = build_possitive_pairs(dataset =dataset,
                                           in_nm_positive =in_nm_positive,
                                           in_nm_query = in_nm_query,
                                           out_nm_query = out_nm_query,
                                           out_nm_positive = out_nm_positive
                                           )
    
    corpus = list(set(pair[out_nm_positive] for pair in original_pairs))
    

    # Filter pairs already processed
    pairs_to_process = [pair for pair in original_pairs if pair[out_nm_query] not in processed_queries]

    if max_samples is not None:
        pairs_to_process = pairs_to_process[:max_samples]

    if not pairs_to_process:
        print("No more datapoints to process.")
        return

    if method == 'random':
        selector = RandomNegativeSelector(corpus=corpus, seed=seed)
    elif method == 'bm25':
        selector = BM25NegativeSelector(corpus=corpus)
    else:
        print(f'Not valid method of negative mining {method}')
        return

    print(f'Selector: {selector}')
    ## Processing
    print(f"Processing {len(pairs_to_process)} pairs.")
    print(f"Starting mining 'hard negatives' with {num_process} parallel processes...")

    with open(output_filename, 'a', encoding='utf-8') as f:
        #creating subprocess
        with multiprocessing.Pool(processes=num_process,
                                  initializer=init_worker,
                                  #initargs=(method, corpus, seed, out_nm_query, out_nm_positive, out_nm_negative)
                                  initargs=(selector, out_nm_query, out_nm_positive, out_nm_negative)
                        ) as pool:
            #launch process unordered
            results_iterator = pool.imap_unordered(find_negative_wrapper, pairs_to_process)
            
            for triplet in tqdm(results_iterator, total=len(pairs_to_process), desc="Mining hard negatives"):
                if triplet:
                    f.write(json.dumps(triplet) + '\n')

    print(f" Process finished. Output file: '{output_filename}'")

    

def main():
    parser = argparse.ArgumentParser(description="Generate triplets for training")

    parser.add_argument("output_filename", type=str, help="Name of the output file")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (HuggingFace) or local path")
    parser.add_argument("in_nm_query", type=str, help="Name of the query field")
    parser.add_argument("in_nm_positive", type=str, help="Name of the positive field")

    parser.add_argument("--out_nm_query", type=str, default="query", help="Output field name for query")
    parser.add_argument("--out_nm_positive", type=str, default="positive", help="Output field name for positive")
    parser.add_argument("--out_nm_negative", type=str, default="negative", help="Output field name for negative")
    parser.add_argument("--method", type=str, choices=["random", "bm25", "dense"], default="random", help="Negative selection method")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--dataset_slice_size", type=int, default=None, help="Subset of the dataset to speed up the process")
    parser.add_argument("--num_process", type=int, default=None, help="Number of subprocess")

    args = parser.parse_args()

    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Loads dataset
    dataset_name = args.dataset_name

    if args.dataset_slice_size or args.max_samples:
        subset_size = args.dataset_slice_size if args.dataset_slice_size and args.dataset_slice_size > args.max_samples else args.max_samples * 2
        dataset = load_dataset(dataset_name, split=f"train[:{subset_size}]")
    else:
        dataset = load_dataset(dataset_name, split=f"train")

    if dataset_name == 'bigbio/pubmed_qa':
        print(f'filtering dataset {dataset_name}')
        dataset = dataset.filter(lambda example: example["final_decision"] == "yes")

    generate_triplets(
        output_filename=args.output_filename,
        dataset=dataset,
        in_nm_query=args.in_nm_query,
        in_nm_positive=args.in_nm_positive,
        out_nm_query=args.out_nm_query,
        out_nm_positive=args.out_nm_positive,
        out_nm_negative=args.out_nm_negative,
        method=args.method,
        seed=args.seed,
        max_samples=args.max_samples,
        num_process=args.num_process if args.num_process else DEFAULT_NUM_PROCESSES
    )

if __name__ == "__main__":
    main()
