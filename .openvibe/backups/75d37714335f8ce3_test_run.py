"""Test script to run the app and see errors"""
import sys
import traceback

try:
    print("Starting app initialization...")
    from app import initialize_system, create_interface
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sample")
    parser.add_argument("--processed_path", type=str, default="data/sample/products_sample.parquet")
    parser.add_argument("--index_path", type=str, default="index/faiss_index")
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--llm_provider", type=str, default="groq")
    parser.add_argument("--llm_model", type=str, default="llama-3.1-70b-versatile")
    parser.add_argument("--port", type=int, default=7860)
    
    args = parser.parse_args()
    
    print("Initializing system...")
    initialize_system(
        data_path=args.data_path,
        processed_data_path=args.processed_path,
        index_path=args.index_path,
        model_name=args.model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
    
    print("Creating interface...")
    app = create_interface()
    
    print(f"Launching on port {args.port}...")
    app.launch(server_port=args.port, share=False, server_name="127.0.0.1")
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

