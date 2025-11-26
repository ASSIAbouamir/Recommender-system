# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your API Key

**Option A: Groq (Recommended - Free & Fast)**
1. Sign up at https://console.groq.com/
2. Create an API key
3. Add to `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```

**Option B: OpenAI**
1. Get key from https://platform.openai.com/
2. Add to `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

### 3. Download Dataset

**Small test (1000 products):**
```python
from src.data_loader import load_sample_data
load_sample_data("data/sample", 1000)
```

**Full Amazon dataset:**
1. Download from https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
2. Extract to `data/` directory
3. Files needed:
   - `reviews_Electronics.json` (or your category)
   - `meta_Electronics.json`

### 4. Run the Application

**Windows:**
```cmd
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Or directly:**
```bash
python app.py --data_path data/
```

### 5. Open Browser

Navigate to: http://localhost:7860

## 🎯 First Recommendations

1. **With sample data:**
   - Leave user ID empty
   - Enter query: "recommend me products"
   - Click "Get Recommendations"

2. **With real data:**
   - Select a user from dropdown
   - Click "Get Recommendations"
   - Or enter a natural query like "sustainable products"

## 🐛 Troubleshooting

**"System not initialized":**
- Check that data files exist in `data/` directory
- Check console for error messages

**"API key not set":**
- Make sure `.env` file exists
- Check that API key is correct
- For Groq, verify at https://console.groq.com/

**"Out of memory":**
- Use smaller dataset (limit products)
- Use smaller embedding model: `--model BAAI/bge-base-en-v1.5`
- Reduce batch size in embedder

**"Index not found":**
- First run will create the index (takes 10-30 min for full dataset)
- Be patient, it's processing embeddings

## 📝 Example Queries

- "sustainable products similar to my purchases"
- "vegan leather handbags under $50"
- "products for home office setup"
- "eco-friendly electronics"

## 🎓 Next Steps

- Read full README.md for advanced usage
- Check out baseline comparisons
- Experiment with different LLM models
- Try sustainability mode


