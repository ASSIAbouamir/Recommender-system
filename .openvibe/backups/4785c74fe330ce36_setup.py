"""
Setup script for RecLM-RAG
"""

from pathlib import Path
import os

# Create necessary directories
dirs = ["data", "embeddings", "index", "logs"]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {d}")

# Create .env file if it doesn't exist
if not Path(".env").exists():
    env_content = """# LLM API Keys (choose one provider)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Data paths
DATA_PATH=data
PROCESSED_DATA_PATH=data/products_processed.parquet
INDEX_PATH=index/faiss_index

# Model settings
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
"""
    with open(".env", "w") as f:
        f.write(env_content)
    print("Created .env file - please add your API keys!")

print("\nSetup complete!")
print("\nNext steps:")
print("1. Add your API key to .env file")
print("2. Download Amazon Reviews dataset to data/ directory")
print("3. Run: python app.py --data_path data/")

