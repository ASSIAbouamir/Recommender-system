# RecLM-RAG: Zero-Shot, Explainable and Sustainability-Aware Recommendation with Retrieval-Augmented LLMs

Official code for the paper **"An Adaptive Cold-Aware Recommendation System for Sparse E-Commerce Data"**, presented orally at the **IEEE International Conference on Circuit, Systems and Communication (ICCSC 2026)**, Fès, Morocco.

📄 **Paper (IEEE Xplore):** https://ieeexplore.ieee.org/abstract/document/11650150

**Authors:** Assia Bouamir, Assoumana Souley Hadiza, Yassine Afoudi — Laboratory of Computer Science and Smart Systems (LISI), Faculty of Sciences Semlalia, Cadi Ayyad University, Marrakech, Morocco

---

## Overview

Collaborative filtering (SVD, ALS) still powers most industrial recommenders, but it struggles with four problems: users with no history (cold start), opaque decisions, no support for natural-language queries, and no sustainability signals.

**RecLM-RAG** is a **training-free, zero-shot** Retrieval-Augmented Generation pipeline that turns product recommendation into a transparent, conversational and sustainability-aware process, with no collaborative-filtering backbone and no fine-tuning.

## Pipeline (6 steps, no training)

| Step | Component | Details |
|---|---|---|
| 1 | Input encoding | Query or user history → BGE-Large embedding (1024-d) |
| 2 | Vector search | FAISS index → top-100 candidates (< 15 ms) |
| 3 | Sustainability score | Rule-based scorer (keywords + regression, 0–100) |
| 4 | Prompt construction | Query + history + candidates + sustainability objective |
| 5 | LLM re-ranking | Llama-3.1-8B (or Mixtral), zero-shot JSON output |
| 6 | Final output | Top-10 recommendations with natural-language explanations and sustainability scores |

End-to-end latency ≈ 1.1 s on a single A100 GPU.

## Results

**Datasets:** Amazon Reviews 2018 — Electronics (192,403 users, 63,001 items, 99.94% sparsity) and Clothing, Shoes & Jewelry (278,154 users, 85,547 items, 99.88% sparsity).
**Protocol:** temporal leave-one-out split, plus a cold-start test set of 10,000 users with zero interaction history.

| Method | Recall@10 | NDCG@10 | Diversity@10 | Long-tail (%) | Latency (ms) |
|---|---|---|---|---|---|
| Popularity | 0.189 | 0.312 | 0.21 | 4.2 | 8 |
| SVD | 0.256 | 0.378 | 0.29 | 8.7 | 12 |
| LightGCN | 0.341 | 0.456 | 0.38 | 14.3 | 45 |
| SASRec | 0.367 | 0.471 | 0.41 | 16.8 | 68 |
| P5 (few-shot) | 0.398 | 0.489 | 0.47 | 19.4 | 890 |
| **RecLM-RAG (Llama)** | **0.498** | **0.562** | **0.71** | **38.4** | 1,100 |
| **RecLM-RAG (Mixtral)** | **0.514** | **0.579** | **0.73** | **41.2** | 1,340 |

Compared with the best baseline (P5): **+25.1% Recall@10, +14.9% NDCG@10, +51.1% Diversity@10, +98% long-tail coverage** (significant at p < 0.01).

**Cold start:** RecLM-RAG keeps **92.6%** of its warm-start Recall@10 for users with no history (0.461 vs 0.498 on Electronics) and 90.1% on Clothing, while traditional and deep CF models drop to popularity level (R@10 ≈ 0.19).

**Ablation (Electronics, Llama-3.1-8B):** removing LLM re-ranking costs −16.0% NDCG@10; replacing BGE-Large with BGE-Base costs −8.9%; removing the sustainability module costs −2.5% NDCG and −4.2% diversity.

**Limitations:** 1.1–1.3 s latency (the LLM accounts for ~91% of it), sustainability lexicon validated only on Electronics/Clothing, occasional hallucinations (< 5%), no use of product images yet.

---

##  Installation

### Prerequisites

- Python 3.10+
- 32GB+ RAM (for full dataset)
- GPU recommended (but works on CPU with smaller datasets)
- LLM API key (Groq/OpenAI/OpenRouter)

### Quick Start

```bash
# Clone repository
git clone https://github.com/ASSIAbouamir/Recommender-system.git
cd Recommender-system

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# Download dataset (see Data section below)
# Place in data/ directory

# Initialize system (will process data and create index)
python app.py --data_path data/
```

##  Project Structure

```
Recommender-system/
├── data/                  # Dataset directory
│   ├── reviews_*.json     # Review files
│   └── meta_*.json        # Product metadata
├── embeddings/            # Cached embeddings
├── index/                 # FAISS indexes
├── src/
│   ├── data_loader.py     # Data loading & preprocessing
│   ├── embedder.py        # Embedding generation
│   ├── indexer.py         # FAISS index management
│   ├── recommender.py     # Main RAG system
│   └── utils.py           # Utility functions
├── baseline_cf.py         # Baseline methods
├── app.py                 # Gradio interface
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

##  Data

### Amazon Reviews Dataset

Recommended datasets:
- **Full dataset**: [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
- **Kaggle subset**: [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)

**Data format expected:**
- `reviews_*.json` or `reviews_*.jsonl`: Each line is a JSON object with `reviewerID`, `asin`, `overall`, `reviewText`, etc.
- `meta_*.json` or `meta_*.jsonl`: Each line is a JSON object with `asin`, `title`, `description`, `category`, `price`, etc.

### Quick Data Setup

```python
from src.data_loader import AmazonDataLoader

loader = AmazonDataLoader(data_path="data", category="Electronics")
loader.load_reviews("data/reviews_Electronics.json")
loader.load_metadata("data/meta_Electronics.json")
products_df = loader.merge_and_preprocess(min_reviews=2, min_rating=3.0)
loader.save_processed("data/products_processed.parquet")
```

##  Usage

### Command Line

```bash
# Launch Gradio interface
python app.py \
    --data_path data/ \
    --processed_path data/products_processed.parquet \
    --index_path index/faiss_index \
    --model BAAI/bge-large-en-v1.5 \
    --llm_provider groq \
    --llm_model llama-3.1-70b-versatile \
    --port 7860
```

### Python API

```python
from src.data_loader import AmazonDataLoader
from src.embedder import ProductEmbedder
from src.indexer import FAISSIndexer
from src.recommender import RecLM_RAG, LLMClient

# Load data
loader = AmazonDataLoader("data")
loader.load_processed("data/products_processed.parquet")

# Initialize components
embedder = ProductEmbedder(model_name="BAAI/bge-large-en-v1.5")
indexer = FAISSIndexer(embedding_dim=1024)
indexer.load("index/faiss_index")

llm_client = LLMClient(provider="groq", model="llama-3.1-70b-versatile")

# Create recommender
recommender = RecLM_RAG(
    data_loader=loader,
    embedder=embedder,
    indexer=indexer,
    products_df=loader.products_df,
    llm_client=llm_client
)

# Get recommendations
recommendations = recommender.recommend(
    user_id="A123456789",
    k=10,
    use_llm_rerank=True,
    sustainability_mode=False
)

# Zero-shot query
recommendations = recommender.recommend(
    natural_query="sustainable organic products",
    k=10
)
```

##  Configuration

### Embedding Models

- **Recommended**: `BAAI/bge-large-en-v1.5` (best quality, 1024 dim)
- **Fast alternative**: `BAAI/bge-base-en-v1.5` (768 dim, faster)
- **Legacy**: `facebook/contriever-msmarco` (768 dim)

### LLM Providers

1. **Groq** (Recommended - Fast & Free)
   ```bash
   export GROQ_API_KEY=your_key
   ```
   Models: `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`

2. **OpenRouter** (Many models)
   ```bash
   export OPENROUTER_API_KEY=your_key
   ```
   Models: `qwen/qwen-2.5-72b-instruct`, `mistralai/mixtral-8x22b-instruct`

3. **OpenAI** (Expensive but reliable)
   ```bash
   export OPENAI_API_KEY=your_key
   ```
   Models: `gpt-4-turbo-preview`, `gpt-3.5-turbo`

##  Examples

### 1. History-Based Recommendations

```python
recommendations = recommender.recommend(
    user_id="A123456789",
    k=10,
    use_llm_rerank=True
)
```

### 2. Zero-Shot Natural Language Query

```python
recommendations = recommender.recommend(
    natural_query="vegan leather handbags under $50",
    k=10
)
```

### 3. Sustainability Mode

```python
recommendations = recommender.recommend(
    user_id="A123456789",
    natural_query="products similar to my purchases",
    sustainability_mode=True,  # Prioritize eco-friendly
    k=10
)
```

##  Evaluation

Compare with baselines:

```python
from baseline_cf import PopularityRecommender, CollaborativeFilteringRecommender

# Popularity baseline
popularity = PopularityRecommender(products_df)
pop_recs = popularity.recommend(user_id="A123456789", k=10)

# CF baseline
cf = CollaborativeFilteringRecommender(reviews_df, products_df)
cf_recs = cf.recommend(user_id="A123456789", k=10)

# Compare
from baseline_cf import compare_recommendations
comparison = compare_recommendations(recommendations, pop_recs)
```

##  Docker

```bash
# Build image
docker build -t reclm-rag .

# Run container
docker run -p 7860:7860 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/index:/app/index \
    -e GROQ_API_KEY=your_key \
    reclm-rag
```

##  Testing

```bash
# Run with sample data (no real dataset needed)
python -c "from src.data_loader import load_sample_data; load_sample_data('data/sample', 1000)"

# Then run app with sample data
python app.py --data_path data/sample
```


## Citation

If you use this work, please cite:

```bibtex
@inproceedings{bouamir2026reclmrag,
  title     = {An Adaptive Cold-Aware Recommendation System for Sparse E-Commerce Data},
  author    = {Bouamir, Assia and Souley Hadiza, Assoumana and Afoudi, Yassine},
  booktitle = {IEEE International Conference on Circuit, Systems and Communication (ICCSC)},
  year      = {2026},
  url       = {https://ieeexplore.ieee.org/abstract/document/11650150}
}
```

## Acknowledgments

- Amazon Reviews dataset by Julian McAuley
- BGE embeddings by BAAI
- FAISS by Meta AI Research
- Gradio for the user interface

## Further analysis

[CRITICAL_ANALYSIS.md](CRITICAL_ANALYSIS.md) covers latency optimization (quantization, caching, distillation), FAISS scalability and a comparison with recent RAG architectures (Self-RAG, REPLUG, RAG-Fusion, ColBERT).

## Future work

- Multi-modal recommendations (image + text)
- Real-time learning from user feedback
- Lower latency (speculative decoding, smaller models)
- Self-RAG critique mechanism and RAG-Fusion for recall
