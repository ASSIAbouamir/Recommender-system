"""
Additional Baselines: BERT4Rec, GRU4Rec, KNN+Embeddings, GPT-based RAG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from tqdm import tqdm

from .embedder import ProductEmbedder
from .recommender import LLMClient


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Bidirectional Encoder Representations from Transformers for Sequential Recommendation
    """
    def __init__(self, num_items, embedding_dim=64, max_len=50, num_blocks=2, num_heads=2, dropout=0.2):
        super(BERT4Rec, self).__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        
        # Item embeddings (index 0 is padding, 1 is mask token)
        self.item_emb = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Blocks (bidirectional, no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_seqs, mask_positions=None):
        # input_seqs: (batch_size, max_len)
        seqs = self.item_emb(input_seqs)
        positions = torch.arange(input_seqs.size(1), device=input_seqs.device).unsqueeze(0).expand(input_seqs.size(0), -1)
        pos = self.pos_emb(positions)
        
        seqs += pos
        seqs = self.dropout(seqs)
        
        # No causal mask (bidirectional)
        key_padding_mask = (input_seqs == 0)
        output = self.transformer_encoder(seqs, src_key_padding_mask=key_padding_mask)
        output = self.norm(output)
        
        return output
    
    def predict(self, user_seqs, candidate_items):
        log_feats = self.forward(user_seqs)
        # Use mean pooling over sequence (bidirectional)
        final_feat = log_feats.mean(dim=1)  # (B, D)
        
        item_embs = self.item_emb(candidate_items)  # (B, K, D)
        scores = torch.bmm(item_embs, final_feat.unsqueeze(2)).squeeze(2)
        return scores


class GRU4Rec(nn.Module):
    """
    GRU4Rec: Gated Recurrent Unit for Sequential Recommendation
    """
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.2):
        super(GRU4Rec, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.item_emb = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_seqs):
        # input_seqs: (batch_size, max_len)
        seqs = self.item_emb(input_seqs)
        seqs = self.dropout(seqs)
        
        # Pack padded sequences
        lengths = (input_seqs != 0).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(seqs, lengths, batch_first=True, enforce_sorted=False)
        
        output, hidden = self.gru(packed)
        # hidden: (num_layers, batch_size, hidden_dim)
        final_hidden = hidden[-1]  # Last layer
        
        return final_hidden
    
    def predict(self, user_seqs, candidate_items):
        final_feat = self.forward(user_seqs)  # (B, hidden_dim)
        
        item_embs = self.item_emb(candidate_items)  # (B, K, embedding_dim)
        
        # Project final_feat to embedding_dim if needed
        if final_feat.size(1) != self.embedding_dim:
            # Use linear projection (simplified)
            scores = torch.bmm(item_embs, final_feat.unsqueeze(2).expand(-1, -1, self.embedding_dim)[:, :, :self.embedding_dim]).squeeze(2)
        else:
            scores = torch.bmm(item_embs, final_feat.unsqueeze(2)).squeeze(2)
        
        return scores


class KNNEmbeddingsRecommender:
    """
    KNN + Embeddings: Dense vector retrieval baseline using embeddings
    """
    def __init__(self, embeddings: np.ndarray, product_ids: np.ndarray, k: int = 10):
        """
        Args:
            embeddings: Product embeddings (n_products, embedding_dim)
            product_ids: Product ASINs corresponding to embeddings
            k: Number of neighbors for KNN
        """
        self.embeddings = embeddings
        self.product_ids = product_ids
        self.k = k
        
        # Build KNN index
        print("Building KNN index...")
        self.knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
        self.knn.fit(embeddings)
    
    def recommend(self, query_embedding: np.ndarray, exclude_asins: Optional[List[str]] = None, k: int = 10) -> List[Dict]:
        """
        Recommend products based on embedding similarity
        
        Args:
            query_embedding: Query embedding (embedding_dim,)
            exclude_asins: List of ASINs to exclude
            k: Number of recommendations
        
        Returns:
            List of recommendation dicts
        """
        query = query_embedding.reshape(1, -1)
        distances, indices = self.knn.kneighbors(query, n_neighbors=min(k * 2, len(self.embeddings)))
        
        recommendations = []
        seen = set(exclude_asins) if exclude_asins else set()
        
        for dist, idx in zip(distances[0], indices[0]):
            asin = self.product_ids[idx]
            if asin not in seen:
                recommendations.append({
                    'asin': asin,
                    'score': float(1 - dist)  # Convert cosine distance to similarity
                })
                seen.add(asin)
                if len(recommendations) >= k:
                    break
        
        return recommendations


class GPTBasedRAGRecommender:
    """
    GPT-based RAG Recommend: LLM-only zero-shot re-ranker
    Uses LLM to re-rank products without dense retrieval
    """
    def __init__(self, products_df: pd.DataFrame, llm_client: LLMClient):
        """
        Args:
            products_df: DataFrame with product information
            llm_client: LLM client for re-ranking
        """
        self.products_df = products_df
        self.llm_client = llm_client
        self.product_map = {}
        for _, row in products_df.iterrows():
            self.product_map[row['asin']] = row.to_dict()
    
    def recommend(
        self, 
        user_history: Optional[List[str]] = None,
        query: Optional[str] = None,
        k: int = 10,
        candidate_pool_size: int = 100
    ) -> List[Dict]:
        """
        Recommend using LLM-only re-ranking
        
        Args:
            user_history: List of ASINs user has purchased
            query: Natural language query
            k: Number of recommendations
            candidate_pool_size: Number of random candidates to consider
        
        Returns:
            List of recommendation dicts
        """
        # Sample random candidates (no retrieval step)
        candidate_asins = np.random.choice(
            self.products_df['asin'].values, 
            size=min(candidate_pool_size, len(self.products_df)),
            replace=False
        )
        
        # Prepare candidate products
        candidate_products = []
        for asin in candidate_asins:
            if asin in self.product_map:
                product = self.product_map[asin].copy()
                candidate_products.append(product)
        
        # Create prompt for LLM
        history_text = ""
        if user_history:
            history_text = f"User history: {', '.join(user_history[:5])}"
        
        products_text = ""
        for i, product in enumerate(candidate_products, 1):
            products_text += f"{i}. {product.get('title', 'Unknown')} | {product.get('description', '')[:100]}\n"
        
        prompt = f"""You are a product recommendation assistant.

{history_text}
{f"User query: {query}" if query else ""}

Here are {len(candidate_products)} candidate products:
{products_text}

Select the TOP-{k} most relevant products and rank them. Return JSON:
{{
  "recommendations": [
    {{"rank": 1, "asin": "B00XXXXX", "explanation": "Why this product..."}},
    ...
  ]
}}
"""
        
        # Call LLM
        response = self.llm_client.generate(prompt)
        
        # Parse response
        import json
        import re
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                llm_recs = data.get('recommendations', [])
                
                recommendations = []
                for rec in llm_recs[:k]:
                    asin = rec.get('asin', '')
                    if asin in self.product_map:
                        product = self.product_map[asin].copy()
                        recommendations.append({
                            'asin': asin,
                            'title': product.get('title', 'Unknown'),
                            'score': 1.0 - (rec.get('rank', 1) - 1) * 0.1,  # Simple score based on rank
                            'explanation': rec.get('explanation', '')
                        })
                
                return recommendations
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
        
        # Fallback: return random products
        return [{'asin': asin, 'title': self.product_map[asin].get('title', 'Unknown'), 'score': 0.5} 
                for asin in candidate_asins[:k]]

