"""
Test script to check app initialization step by step
"""
import sys
from pathlib import Path

print("=" * 60)
print("Testing RecLM-RAG Initialization")
print("=" * 60)

# Step 1: Test imports
print("\n1. Testing imports...")
try:
    from src.data_loader import AmazonDataLoader
    from src.embedder import ProductEmbedder
    from src.indexer import FAISSIndexer
    from src.recommender import RecLM_RAG, LLMClient
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Step 2: Load data
print("\n2. Loading data...")
try:
    data_loader = AmazonDataLoader(data_path="data/sample")
    data_loader.load_processed("data/sample/products_sample.parquet")
    
    # Try to load reviews
    reviews_file = Path("data/sample/reviews_sample.jsonl")
    if reviews_file.exists():
        print(f"Loading reviews from {reviews_file}")
        data_loader.load_reviews(str(reviews_file))
    
    print(f"✓ Loaded {len(data_loader.products_df)} products")
    if data_loader.reviews_df is not None:
        print(f"✓ Loaded {len(data_loader.reviews_df)} reviews")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Initialize embedder
print("\n3. Initializing embedder...")
try:
    embedder = ProductEmbedder(model_name="BAAI/bge-base-en-v1.5")
    print(f"✓ Embedder initialized (dim: {embedder.embedding_dim})")
except Exception as e:
    print(f"✗ Embedder error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Check/create index
print("\n4. Checking index...")
try:
    indexer = FAISSIndexer(embedding_dim=embedder.embedding_dim)
    index_path = Path("index/faiss_index")
    
    if index_path.with_suffix('.index').exists():
        print(f"✓ Index exists at {index_path}")
        indexer.load(str(index_path))
    else:
        print("⚠ Index not found - will be created on first run")
        print("  (This will take several minutes to generate embeddings)")
except Exception as e:
    print(f"✗ Index error: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Initialize LLM client (optional)
print("\n5. Testing LLM client...")
try:
    llm_client = LLMClient(provider="groq", model="llama-3.1-70b-versatile")
    print("✓ LLM client initialized (API key check will happen on first use)")
except Exception as e:
    print(f"⚠ LLM client warning: {e}")
    print("  (This is OK - API key will be checked when needed)")

print("\n" + "=" * 60)
print("Initialization test complete!")
print("=" * 60)
print("\nYou can now run: python app.py --data_path data/sample --processed_path data/sample/products_sample.parquet --index_path index/faiss_index --model BAAI/bge-base-en-v1.5 --port 7860")

