"""
FAISS index creation and management for efficient similarity search
Supports Flat, IVF, and HNSW indexes based on dataset size
"""

import faiss
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import pickle


class FAISSIndexer:
    """Build and manage FAISS index for product retrieval"""
    
    def __init__(self, embedding_dim: int, index_type: str = "auto"):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_type: 'flat', 'ivf', 'hnsw', or 'auto' (choose based on size)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.product_ids = None  # Map index position to product ASIN
        
    def build_index(
        self,
        embeddings: np.ndarray,
        product_ids: np.ndarray,
        n_clusters: int = 100
    ):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of shape (n_products, embedding_dim)
            product_ids: array of product ASINs corresponding to embeddings
            n_clusters: Number of clusters for IVF index (ignored for flat)
        """
        n_products, dim = embeddings.shape
        assert dim == self.embedding_dim, f"Dimension mismatch: {dim} != {self.embedding_dim}"
        
        print(f"Building FAISS index for {n_products} products...")
        
        # Choose index type
        if self.index_type == "auto":
            if n_products < 50000:
                index_type = "flat"
            elif n_products < 500000:
                index_type = "ivf"
            else:
                index_type = "ivf"  # Can switch to HNSW for very large datasets
        else:
            index_type = self.index_type
        
        # Build appropriate index
        if index_type == "flat":
            print("Using FlatIndex (exact search, slower for large datasets)")
            self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity (normalized embeddings)
        elif index_type == "ivf":
            print(f"Using IVF index with {n_clusters} clusters (approximate, faster)")
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, min(n_clusters, n_products // 10))
            
            # Train the index
            print("Training IVF index...")
            self.index.train(embeddings.astype('float32'))
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        print("Adding embeddings to index...")
        self.index.add(embeddings.astype('float32'))
        
        # Store product IDs mapping
        self.product_ids = product_ids
        
        print(f"Index built successfully! Index size: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
        filter_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar products
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            k: Number of results to return
            filter_ids: Optional array of ASINs to filter out (e.g., already purchased)
        
        Returns:
            (distances, indices) where:
            - distances: similarity scores (higher is better for IP)
            - indices: positions in product_ids array
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Reshape query to (1, dim)
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Filter out excluded products if specified
        if filter_ids is not None and len(filter_ids) > 0:
            mask = np.isin(self.product_ids[indices[0]], filter_ids, invert=True)
            indices = indices[0][mask]
            distances = distances[0][mask]
        else:
            indices = indices[0]
            distances = distances[0]
        
        return distances, indices
    
    def get_product_ids(self, indices: np.ndarray) -> np.ndarray:
        """Convert index positions to product ASINs"""
        if self.product_ids is None:
            raise ValueError("Product IDs not set")
        return self.product_ids[indices]
    
    def save(self, index_path: str):
        """Save index and metadata to disk"""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path.with_suffix('.index')))
        
        # Save metadata
        metadata_path = index_path.with_suffix('.meta')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'product_ids': self.product_ids
            }, f)
        
        print(f"Index saved to {index_path.with_suffix('.index')}")
        print(f"Metadata saved to {metadata_path}")
    
    def load(self, index_path: str):
        """Load index and metadata from disk"""
        index_path = Path(index_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path.with_suffix('.index')))
        
        # Load metadata
        metadata_path = index_path.with_suffix('.meta')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.embedding_dim = metadata['embedding_dim']
            self.index_type = metadata.get('index_type', 'auto')
            self.product_ids = metadata['product_ids']
        
        print(f"Index loaded from {index_path.with_suffix('.index')}")
        print(f"Index size: {self.index.ntotal}")


def create_index_from_embeddings(
    embeddings: np.ndarray,
    product_ids: np.ndarray,
    embedding_dim: int,
    output_path: str,
    index_type: str = "auto"
) -> FAISSIndexer:
    """
    Convenience function to create and save index
    
    Args:
        embeddings: Product embeddings
        product_ids: Product ASINs
        embedding_dim: Embedding dimension
        output_path: Path to save index
        index_type: Index type ('auto', 'flat', 'ivf')
    
    Returns:
        FAISSIndexer instance
    """
    indexer = FAISSIndexer(embedding_dim=embedding_dim, index_type=index_type)
    indexer.build_index(embeddings, product_ids)
    indexer.save(output_path)
    return indexer

