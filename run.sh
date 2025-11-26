#!/bin/bash
# Quick start script for RecLM-RAG

echo "🚀 RecLM-RAG - Starting up..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    python setup.py
fi

# Set default values
DATA_PATH=${DATA_PATH:-"data"}
PROCESSED_PATH=${PROCESSED_PATH:-"data/products_processed.parquet"}
INDEX_PATH=${INDEX_PATH:-"index/faiss_index"}
MODEL=${MODEL:-"BAAI/bge-large-en-v1.5"}
LLM_PROVIDER=${LLM_PROVIDER:-"groq"}
LLM_MODEL=${LLM_MODEL:-"llama-3.1-70b-versatile"}
PORT=${PORT:-7860}

echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Model: $MODEL"
echo "  LLM Provider: $LLM_PROVIDER"
echo "  LLM Model: $LLM_MODEL"
echo "  Port: $PORT"
echo ""

# Run the application
python app.py \
    --data_path "$DATA_PATH" \
    --processed_path "$PROCESSED_PATH" \
    --index_path "$INDEX_PATH" \
    --model "$MODEL" \
    --llm_provider "$LLM_PROVIDER" \
    --llm_model "$LLM_MODEL" \
    --port "$PORT" \
    "$@"


