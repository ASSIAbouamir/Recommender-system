"""
Utility functions for RecLM-RAG
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import requests
from io import BytesIO
from PIL import Image
import re
from pathlib import Path


def download_image(url: str, max_size: int = 224) -> Optional[Image.Image]:
    """Download and resize product image"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.thumbnail((max_size, max_size))
            return img
    except Exception as e:
        pass
    return None


def create_product_card_html(
    asin: str,
    title: str,
    score: float,
    explanation: str,
    image_url: Optional[str] = None,
    price: Optional[float] = None
) -> str:
    """Create HTML card for product display"""
    image_html = ""
    if image_url:
        image_html = f'<img src="{image_url}" style="max-width:150px;max-height:150px;" />'
    else:
        image_html = '<div style="width:150px;height:150px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;">No Image</div>'
    
    price_html = f"${price:.2f}" if price else "N/A"
    
    html = f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:15px;margin:10px;background:white;">
        <div style="display:flex;gap:15px;">
            <div>{image_html}</div>
            <div style="flex:1;">
                <h3 style="margin:0 0 10px 0;font-size:16px;">{title[:80]}</h3>
                <p style="color:#666;font-size:12px;margin:5px 0;">ASIN: {asin}</p>
                <p style="color:#28a745;font-size:14px;font-weight:bold;margin:5px 0;">Score: {score:.3f} | Price: {price_html}</p>
                <p style="font-size:13px;color:#555;margin-top:10px;">{explanation}</p>
            </div>
        </div>
    </div>
    """
    return html


def format_user_history(history_df: pd.DataFrame) -> str:
    """Format user history for display"""
    if len(history_df) == 0:
        return "No purchase history"
    
    items = []
    for _, row in history_df.iterrows():
        title = row.get('title', 'Unknown Product')
        rating = row.get('rating', 'N/A')
        items.append(f"• {title} (Rating: {rating}/5)")
    
    return "\n".join(items[:10])  # Show top 10


def extract_sustainability_keywords(text: str) -> List[str]:
    """Extract sustainability-related keywords"""
    keywords = [
        'organic', 'recycled', 'sustainable', 'eco-friendly', 'vegan',
        'biodegradable', 'compostable', 'carbon neutral', 'fair trade',
        'green', 'environmentally friendly', 'renewable', 'ethical'
    ]
    
    found = []
    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            found.append(keyword)
    
    return found


def calculate_ndcg(recommended_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Calculate NDCG@k metric"""
    if len(relevant_ids) == 0:
        return 0.0
    
    # Take top k
    recommended_ids = recommended_ids[:k]
    relevant_set = set(relevant_ids)
    
    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(recommended_ids):
        if item_id in relevant_set:
            rel = 1.0
            dcg += rel / np.log2(i + 2)  # i+2 because index starts at 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    num_relevant = min(len(relevant_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def clean_text(text: str) -> str:
    """Clean text for display"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def create_rag_prompt(
    user_history_summary: str,
    candidate_products: List[Dict],
    query: Optional[str] = None,
    sustainability_mode: bool = False
) -> str:
    """
    Create RAG prompt for LLM
    
    Args:
        user_history_summary: Summary of user's purchase history
        candidate_products: List of dicts with product info (top 100 from retrieval)
        query: Optional natural language query
        sustainability_mode: Whether to prioritize sustainable products
    
    Returns:
        Formatted prompt string
    """
    # Format candidate products
    products_text = ""
    for i, product in enumerate(candidate_products[:100], 1):  # Top 100
        products_text += f"\n{i}. ASIN: {product['asin']}\n"
        products_text += f"   Title: {product['title']}\n"
        products_text += f"   Category: {product.get('category', 'N/A')}\n"
        products_text += f"   Description: {product.get('description', '')[:200]}\n"
        products_text += f"   Retrieval Score: {product.get('score', 0):.4f}\n"
    
    sustainability_note = ""
    if sustainability_mode:
        sustainability_note = """
IMPORTANT: Prioritize products that are sustainable, eco-friendly, organic, recycled, or vegan.
Look for keywords like: organic, recycled, sustainable, eco-friendly, vegan, biodegradable, fair trade.
"""
    
    prompt = f"""You are an expert product recommendation assistant for an e-commerce platform.

User's Purchase History:
{user_history_summary}

{f"User Query: {query}" if query else ""}

Candidate Products (retrieved from similarity search):
{products_text}

Your task:
1. Analyze the user's purchase history and preferences
2. From the candidate products above, select the TOP 10 most relevant products
3. Re-rank them in order of relevance (1 = most relevant)
4. For each product, provide a natural, convincing explanation (2-3 sentences) explaining why you recommend it

{sustainability_note}

Output your recommendations in the following JSON format:
{{
  "recommendations": [
    {{
      "rank": 1,
      "asin": "B00XXXXX",
      "explanation": "Why this product is perfect for the user..."
    }},
    ...
  ]
}}

Be specific and reference the user's past purchases when relevant. Make the explanations engaging and personalized.
"""
    
    return prompt


def parse_llm_response(response_text: str) -> List[Dict]:
    """Parse LLM JSON response"""
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        data = json.loads(response_text)
        return data.get('recommendations', [])
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response text: {response_text[:500]}")
        return []

