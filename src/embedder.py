"""
Embedding generation using state-of-the-art models
Supports: BGE-large-en-v1.5, Contriever-MS-MARCO
"""

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from pathlib import Path
import pickle
from tqdm import tqdm
import os


class ProductEmbedder:
    """Generate dense embeddings for products"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name or path
                       Options: "BAAI/bge-large-en-v1.5" (default, best quality)
                                "facebook/contriever-msmarco"
            device: 'cuda', 'cpu', or None (auto-detect)
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading embedding model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading {model_name}, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.model_name = 'all-MiniLM-L6-v2'
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_products(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embeddings for product texts
        
        Args:
            texts: List of product text representations
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            cache_key: Optional cache key to save/load embeddings
        
        Returns:
            numpy array of shape (n_products, embedding_dim)
        """
        # Check cache
        if cache_key:
            cache_path = self.cache_dir / f"{cache_key}_embeddings.pkl"
            if cache_path.exists():
                print(f"Loading embeddings from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        print(f"Generating embeddings for {len(texts)} products...")
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        # Save to cache
        if cache_key:
            print(f"Saving embeddings to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query
        
        Args:
            query: Query text
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    def embed_user_history(
        self,
        user_history_texts: List[str],
        aggregation: str = "mean"
    ) -> np.ndarray:
        """
        Generate embedding for user purchase history
        
        Args:
            user_history_texts: List of text representations for purchased products
            aggregation: How to aggregate ('mean', 'max', 'weighted_mean')
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        if len(user_history_texts) == 0:
            return np.zeros(self.embedding_dim)
        
        # Encode all products in history
        embeddings = self.model.encode(
            user_history_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Aggregate
        if aggregation == "mean":
            return np.mean(embeddings, axis=0)
        elif aggregation == "max":
            return np.max(embeddings, axis=0)
        elif aggregation == "weighted_mean":
            # Could weight by rating or recency here
            weights = np.ones(len(embeddings)) / len(embeddings)
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def create_query_embedding(
        self,
        user_history_texts: List[str],
        natural_query: Optional[str] = None
    ) -> np.ndarray:
        """
        Create rich query embedding from user history + optional natural query
        
        Args:
            user_history_texts: List of product texts from user history
            natural_query: Optional natural language query (e.g., "sustainable products")
        
        Returns:
            Combined query embedding
        """
        # Embed user history
        history_emb = self.embed_user_history(user_history_texts)
        
        if natural_query:
            # Embed natural query
            query_emb = self.embed_query(natural_query)
            # Combine (weighted average)
            combined = 0.7 * history_emb + 0.3 * query_emb
            # Renormalize
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            return combined
        else:
            # Just use history
            return history_emb


def create_rich_query_text(
    user_history: pd.DataFrame,
    natural_query: Optional[str] = None
) -> str:
    """
    Create a rich natural language query from user history
    
    Args:
        user_history: DataFrame with columns: title, description, category
        natural_query: Optional additional query text
    
    Returns:
        Natural language query string
    """
    parts = []
    
    if len(user_history) > 0:
        # Extract key information
        titles = user_history['title'].dropna().unique()[:5].tolist()
        categories = user_history['category'].dropna().unique()[:3].tolist()
        
        if titles:
            parts.append(f"Products similar to: {', '.join(titles[:3])}")
        
        if categories:
            parts.append(f"in categories: {', '.join(categories)}")
        
        parts.append("products that I would like based on my purchase history")
    
    if natural_query:
        parts.append(natural_query)
    
    if not parts:
        return "recommend popular products"
    
    return ". ".join(parts) + "."


if __name__ == "__main__":
    # Test embedding generation
    embedder = ProductEmbedder()
    
    test_texts = [
        "Title: Wireless Bluetooth Headphones | Category: Electronics | Description: High-quality wireless headphones with noise cancellation",
        "Title: Organic Cotton T-Shirt | Category: Clothing | Description: Sustainable t-shirt made from organic cotton",
    ]
    
    embeddings = embedder.embed_products(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

