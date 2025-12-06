from app import initialize_system, get_recommendations
import traceback

print("Initializing system...")
# using defaults matching app.py
initialize_system(
    data_path="data",
    processed_data_path="data/products_processed.parquet",
    index_path="index/faiss_index"
)

print("Calling get_recommendations...")
try:
    # User input from prompt
    user_id = "A000000173"
    query = "Standard Plastic Toy"
    
    res = get_recommendations(
        user_id=user_id,
        natural_query=query,
        k=10,
        use_llm=True,
        sustainability=True 
    )
    print("Result (first 100 chars):", str(res)[:100])
except Exception:
    traceback.print_exc()
