@echo off
REM Quick start script for RecLM-RAG (Windows)

echo RecLM-RAG - Starting up...

REM Check if .env exists
if not exist .env (
    echo Creating .env file...
    python setup.py
)

REM Set default values
set DATA_PATH=data
set PROCESSED_PATH=data/products_processed.parquet
set INDEX_PATH=index/faiss_index
set MODEL=BAAI/bge-large-en-v1.5
set LLM_PROVIDER=groq
set LLM_MODEL=llama-3.1-70b-versatile
set PORT=7860

echo Configuration:
echo   Data path: %DATA_PATH%
echo   Model: %MODEL%
echo   LLM Provider: %LLM_PROVIDER%
echo   Port: %PORT%
echo.

REM Run the application
python app.py --data_path "%DATA_PATH%" --processed_path "%PROCESSED_PATH%" --index_path "%INDEX_PATH%" --model "%MODEL%" --llm_provider "%LLM_PROVIDER%" --llm_model "%LLM_MODEL%" --port %PORT%

pause


