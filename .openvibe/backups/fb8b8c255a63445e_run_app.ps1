# PowerShell script to run RecLM-RAG
Write-Host "Starting RecLM-RAG application..." -ForegroundColor Green
Write-Host ""

$env:PYTHONUNBUFFERED = "1"

python app.py `
    --data_path data/sample `
    --processed_path data/sample/products_sample.parquet `
    --index_path index/faiss_index `
    --model BAAI/bge-base-en-v1.5 `
    --port 7860

Write-Host ""
Write-Host "Application stopped." -ForegroundColor Yellow

