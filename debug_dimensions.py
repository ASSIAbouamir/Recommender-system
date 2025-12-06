from app import initialize_system, reclm_rag
import traceback

print("Initializing system...")
initialize_system(
    data_path="data",
    processed_data_path="data/products_processed.parquet",
    index_path="index/faiss_index"
)

print("\nChecking dimensions...")
print(f"Embedder dimension: {reclm_rag.embedder.embedding_dim}")
print(f"Index dimension: {reclm_rag.indexer.index.d}")
print(f"Index metadata embedding_dim: {reclm_rag.indexer.embedding_dim}")

print("\nCalling get_recommendations...")
try:
    from app import get_recommendations
    res = get_recommendations(
        user_id="A000000173",
        natural_query="Standard Plastic Toy",
        k=10,
        use_llm=True,
        sustainability=False
    )
    print("Success!")
except Exception:
    traceback.print_exc()
