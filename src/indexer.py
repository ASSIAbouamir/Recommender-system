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
    
    def __init__(
        self, 
        embedding_dim: int, 
        index_type: str = "auto",
        nprobe: int = 64,
        efSearch: int = 128
    ):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_type: 'flat', 'ivf', 'hnsw', or 'auto' (choose based on size)
            nprobe: Number of clusters to probe for IVF indexes (paper: 64)
            efSearch: HNSW search parameter (paper: 128)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nprobe = nprobe
        self.efSearch = efSearch
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
        # Validate embeddings shape
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array, got shape {embeddings.shape}. "
                f"This usually means the product dataset is empty. "
                f"Please check your data loading and preprocessing steps."
            )
        
        if embeddings.shape[0] == 0:
            raise ValueError(
                "Cannot build index with 0 products. "
                "Please check your data loading and preprocessing steps."
            )
        
        n_products, dim = embeddings.shape
        assert dim == self.embedding_dim, f"Dimension mismatch: {dim} != {self.embedding_dim}"
        
        print(f"Building FAISS index for {n_products} products...")
        
        # Choose index type
        if self.index_type == "auto":
            if n_products < 50000:
                index_type_str = "Flat"
            else:
                index_type_str = "IVF100,Flat" # Default simple IVF
        else:
            index_type_str = self.index_type
        
        print(f"Building index type: {index_type_str}")
        
        # Build index using factory
        try:
            # Check if we need to measure training time or use GPU
            # For this simplified version, we use CPU and factory
            
            # Special handling for metric (Inner Product vs L2)
            # BGE embeddings are normalized, so IP == Cosine Similarity
            metric = faiss.METRIC_INNER_PRODUCT
            
            self.index = faiss.index_factory(self.embedding_dim, index_type_str, metric)
            
            # Set parameters if available (e.g. nprobe)
            # Note: These are usually set at search time, but can be set on index
        except Exception as e:
             print(f"Error creating index with factory: {e}. Falling back to Flat.")
             self.index = faiss.IndexFlatIP(dim)

        # Train if necessary
        if not self.index.is_trained:
            print(f"Training index with {min(50000, len(embeddings))} samples...")
            # Use a random subset for training if dataset is huge
            if len(embeddings) > 50000:
                indices = np.random.choice(len(embeddings), 50000, replace=False)
                train_data = embeddings[indices]
            else:
                train_data = embeddings
            
            self.index.train(train_data.astype('float32'))
        
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
        
        # Debug: Check dimension mismatch
        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Query embedding dimension mismatch! "
                f"Query has {query.shape[1]} dimensions, but index expects {self.index.d}. "
                f"This usually means the embedding model changed. "
                f"Please delete the index files and rebuild."
            )
        
        # Set search parameters as per paper
        # For IVF indexes, set nprobe
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # For HNSW indexes, set efSearch
        if hasattr(self.index, 'hnsw') and hasattr(self.index.hnsw, 'efSearch'):
            self.index.hnsw.efSearch = self.efSearch
        
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

