"""
Quick verification script for RecLM-RAG system
Tests core functionality without launching the full Gradio interface
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required packages are importable"""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("gradio", "Gradio"),
        ("groq", "Groq API"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            return False
    
    return True

def test_data_loading():
    """Test data loading"""
    print("\n" + "=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    try:
        from src.data_loader import AmazonDataLoader
        
        data_path = "data/sample"
        processed_path = "data/sample/products_sample.parquet"
        
        if not Path(processed_path).exists():
            print(f"❌ Processed data not found: {processed_path}")
            return False
        
        loader = AmazonDataLoader(data_path=data_path)
        loader.load_processed(processed_path)
        
        print(f"✅ Loaded {len(loader.products_df)} products")
        print(f"   Sample product: {loader.products_df.iloc[0]['title'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        traceback.print_exc()
        return False

def test_embedder():
    """Test embedding generation"""
    print("\n" + "=" * 60)
    print("Testing Embedder")
    print("=" * 60)
    
    try:
        from src.embedder import ProductEmbedder
        
        print("Initializing embedder (this may take a moment)...")
        embedder = ProductEmbedder(model_name="BAAI/bge-base-en-v1.5")
        
        # Test embedding a simple query
        test_text = "sustainable organic cotton t-shirt"
        embedding = embedder.embed_query(test_text)
        
        print(f"✅ Generated embedding with dimension: {len(embedding)}")
        print(f"   Test query: '{test_text}'")
        
        return True
    except Exception as e:
        print(f"❌ Error with embedder: {e}")
        traceback.print_exc()
        return False

def test_indexer():
    """Test FAISS indexer"""
    print("\n" + "=" * 60)
    print("Testing FAISS Indexer")
    print("=" * 60)
    
    try:
        from src.indexer import FAISSIndexer
        import numpy as np
        
        # Create a simple test index
        indexer = FAISSIndexer(embedding_dim=768)
        
        # Generate some random embeddings for testing
        test_embeddings = np.random.rand(100, 768).astype('float32')
        test_ids = [f"ASIN_{i}" for i in range(100)]
        
        indexer.build_index(test_embeddings, test_ids)
        
        # Test search
        query_embedding = np.random.rand(768).astype('float32')
        distances, indices = indexer.search(query_embedding, k=5)
        
        print(f"✅ Built index with 100 products")
        print(f"   Retrieved top-5 results")
        print(f"   Top result distance: {distances[0]:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error with indexer: {e}")
        traceback.print_exc()
        return False

def test_llm_client():
    """Test LLM client initialization"""
    print("\n" + "=" * 60)
    print("Testing LLM Client")
    print("=" * 60)
    
    try:
        from src.recommender import LLMClient
        import os
        
        # Check if API key is set
        if not os.getenv("GROQ_API_KEY"):
            print("⚠️  GROQ_API_KEY not set - LLM features will be limited")
            print("   (This is OK for testing basic functionality)")
            return True
        
        llm_client = LLMClient(provider="groq", model="llama-3.1-70b-versatile")
        
        if llm_client.client:
            print("✅ LLM client initialized successfully")
            print("   Provider: Groq")
            print("   Model: llama-3.1-70b-versatile")
        else:
            print("⚠️  LLM client initialized but no API connection")
        
        return True
    except Exception as e:
        print(f"❌ Error with LLM client: {e}")
        traceback.print_exc()
        return False

def test_recommender():
    """Test basic recommendation generation"""
    print("\n" + "=" * 60)
    print("Testing Recommender System")
    print("=" * 60)
    
    try:
        from src.data_loader import AmazonDataLoader
        from src.embedder import ProductEmbedder
        from src.indexer import FAISSIndexer
        from src.recommender import RecLM_RAG, LLMClient
        
        # Load data
        data_path = "data/sample"
        processed_path = "data/sample/products_sample.parquet"
        
        loader = AmazonDataLoader(data_path=data_path)
        loader.load_processed(processed_path)
        products_df = loader.products_df
        
        # Initialize components
        print("Initializing embedder...")
        embedder = ProductEmbedder(model_name="BAAI/bge-base-en-v1.5")
        
        print("Generating embeddings...")
        embeddings = embedder.embed_products(
            products_df['text_for_embedding'].tolist()[:100],  # Limit for speed
            cache_key="test_products"
        )
        
        print("Building index...")
        indexer = FAISSIndexer(embedding_dim=embedder.embedding_dim)
        indexer.build_index(
            embeddings,
            products_df['asin'].values[:100]
        )
        
        # Initialize recommender (without LLM for speed)
        llm_client = LLMClient(provider="groq", model="llama-3.1-70b-versatile")
        
        recommender = RecLM_RAG(
            data_loader=loader,
            embedder=embedder,
            indexer=indexer,
            products_df=products_df.head(100),
            llm_client=llm_client
        )
        
        # Test zero-shot recommendation
        print("\nGenerating recommendations for query: 'sustainable organic products'")
        recommendations = recommender.recommend(
            natural_query="sustainable organic products",
            k=5,
            use_llm_rerank=False  # Skip LLM for speed
        )
        
        print(f"\n✅ Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['title'][:60]}... (Score: {rec['score']:.3f})")
        
        return True
    except Exception as e:
        print(f"❌ Error with recommender: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RecLM-RAG System Verification")
    print("=" * 60 + "\n")
    
    results = {
        "Package Imports": test_imports(),
        "Data Loading": test_data_loading(),
        "Embedder": test_embedder(),
        "FAISS Indexer": test_indexer(),
        "LLM Client": test_llm_client(),
        "Recommender System": test_recommender(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! RecLM-RAG is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python app.py' to launch the Gradio interface")
        print("2. Visit http://localhost:7860 in your browser")
        print("3. Try generating recommendations!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
