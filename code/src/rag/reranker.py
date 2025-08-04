from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from utils.vprint import vprint

class ColBERTReranker:
    def __init__(self, model_name: str ='colbert-ir/colbertv2.0', device=None, verbose: bool= True):
        """
        Initializes the ColBERT reranker model and tokenizer.
        """
        print(verbose, f"--> [Retriever {type(self).__name__}] Loading model from '{model_name}' ...")
        self._verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if model_name != 'colbert-ir/colbertv2.0':
            self.model_wrapper = SentenceTransformer(model_name_or_path=model_name, device=self.device)
            
            self.model = self.model_wrapper[0].auto_model
            self.tokenizer = self.model_wrapper.tokenizer
        
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Set eval mode to inference 
        self.model.eval()


    def encode(self, texts):
        """
        Encodes a list of texts into token embeddings using the ColBERT model.
        Returns a dictionary containing embeddings and attention masks.
        """
        encoded_inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        # Move inputs to the designated device (GPU/CPU)
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        
        with torch.no_grad():
            # Pass the encoded inputs through the model to get the embeddings
            output = self.model(**encoded_inputs)
            
        # Return the last hidden states and the attention mask
        return {
            'embeddings': output.last_hidden_state,  # shape: (batch_size, seq_len, hidden_dim)
            'attention_mask': encoded_inputs['attention_mask'] # shape: (batch_size, seq_len)
        }

    def score(self, query_emb_dict, doc_emb_dict):
        """
        Computes ColBERT-style MaxSim similarity between the query and the doc embedding.
        
        Args:
            query_emb_dict (dict): Dictionary with 'embeddings' and 'attention_mask' for the query.
                                   Shape of 'embeddings' is [1, Q_len, H].
            doc_emb_dict (dict): Dictionary with 'embeddings' and 'attention_mask' for the documents.
                                 Shape of 'embeddings' is [N, D_len, H].
        Returns:
            torch.Tensor: A tensor of scores for each document in the batch. Shape is [N].
        """
        # Extract embeddings and masks from the input dictionaries
        query_emb = query_emb_dict['embeddings']
        query_mask = query_emb_dict['attention_mask']
        doc_emb = doc_emb_dict['embeddings']
        doc_mask = doc_emb_dict['attention_mask']
        
        # Normalize embeddings to unit vectors for cosine similarity
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        
        # Compute the similarity matrix for all query-doc token pairs. Shape: [batch_size (N), q_len, d_len]
        #ToDo: 
        query_emb_expanded = query_emb.expand(doc_emb.size(0), -1, -1) 
        sim_matrix = torch.matmul(query_emb_expanded, doc_emb.transpose(1, 2)) # [batch, q_len, d_len]
        
        # Apply the attention masks to the similarity matrix
        query_mask_expanded = query_mask.unsqueeze(2).expand_as(sim_matrix) # [N, Q_len, 1] -> [N, Q_len, D_len]
        doc_mask_expanded = doc_mask.unsqueeze(1).expand_as(sim_matrix)     # [N, 1, D_len] -> [N, Q_len, D_len]
        
        # The combined mask is 0 if either query or doc token is padding
        mask = query_mask_expanded * doc_mask_expanded
        
        # Use a high negative number or -inf to mask out padding tokens
        sim_matrix_masked = sim_matrix.masked_fill(mask == 0, float('-inf'))
        
        # Compute MaxSim over document tokens for each query token
        max_sim_per_query_token, _ = sim_matrix_masked.max(dim=2) # [N, Q_len]
        
        # Sum the maximum similarities over the query tokens
        scores = max_sim_per_query_token.sum(dim=1) # Shape: [N]
        
        return scores

    def rerank(self, query, documents, top_k=5):
        """
        Reranks a list of documents for a given query using the ColBERT scoring.
        """
        vprint(self._verbose, "--> [Re-ranker] Encoding query...")
        query_emb_dict = self.encode([query])  # Query encoded as a batch of 1
        
        vprint(self._verbose, f"--> [Re-ranker] Encoding {len(documents)} documents in a batch...")
        doc_emb_dict = self.encode(documents)  # Documents encoded in one large batch
        
        vprint(self._verbose, "Scoring and reranking documents...")
        # Compute scores for the entire batch of documents in one vectorized operation
        scores = self.score(query_emb_dict, doc_emb_dict).cpu().numpy()
        
        # Combine documents with their scores and sort
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]

