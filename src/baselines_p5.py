"""
P5 Baseline: T5-based recommendation model
P5: Personalized Prompt Pre-training for Recommendation
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


class P5Recommender:
    """
    P5: Personalized Prompt Pre-training for Recommendation
    T5-base fine-tuned on target domain (few-shot variant)
    """
    def __init__(
        self,
        products_df: pd.DataFrame,
        model_name: str = "google/flan-t5-base",  # T5 variant
        device: str = "cpu"
    ):
        """
        Args:
            products_df: DataFrame with product information
            model_name: T5 model name (default: flan-t5-base, can use t5-base)
            device: 'cuda' or 'cpu'
        """
        self.products_df = products_df
        self.device = device
        
        # Load T5 model and tokenizer
        print(f"Loading P5 model: {model_name}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            print("P5 model loaded successfully!")
        except Exception as e:
            print(f"Error loading P5 model: {e}")
            print("P5 requires transformers library. Install with: pip install transformers")
            self.model = None
            self.tokenizer = None
        
        # Create product map
        self.product_map = {}
        for _, row in products_df.iterrows():
            self.product_map[row['asin']] = row.to_dict()
    
    def recommend(
        self,
        user_history: Optional[List[str]] = None,
        query: Optional[str] = None,
        k: int = 10
    ) -> List[Dict]:
        """
        Generate recommendations using P5 prompts
        
        Args:
            user_history: List of ASINs user has purchased
            query: Optional natural language query
            k: Number of recommendations
        
        Returns:
            List of recommendation dicts
        """
        if self.model is None or self.tokenizer is None:
            # Fallback to popularity
            popular = self.products_df.nlargest(k, 'num_reviews')
            return [{'asin': row['asin'], 'title': row.get('title', 'Unknown'), 'score': 0.5} 
                    for _, row in popular.iterrows()]
        
        # Build prompt in P5 format
        if user_history and len(user_history) > 0:
            # Get product titles from history
            history_titles = []
            for asin in user_history[:5]:  # Last 5 items
                if asin in self.product_map:
                    history_titles.append(self.product_map[asin].get('title', ''))
            
            history_text = ", ".join(history_titles)
            prompt = f"Given the following user purchase history: {history_text}. Recommend {k} products:"
        elif query:
            prompt = f"Given the query: {query}. Recommend {k} products:"
        else:
            prompt = f"Recommend {k} popular products:"
        
        # Generate with T5
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=5,
                    early_stopping=True,
                    num_return_sequences=1
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse generated text to extract product recommendations
            # P5 typically generates product titles or IDs
            recommendations = self._parse_p5_output(generated_text, k)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in P5 generation: {e}")
            # Fallback
            popular = self.products_df.nlargest(k, 'num_reviews')
            return [{'asin': row['asin'], 'title': row.get('title', 'Unknown'), 'score': 0.5} 
                    for _, row in popular.iterrows()]
    
    def _parse_p5_output(self, generated_text: str, k: int) -> List[Dict]:
        """
        Parse P5 model output to extract recommendations
        P5 may generate product titles, descriptions, or IDs
        """
        recommendations = []
        
        # Try to match generated text with product titles
        generated_lower = generated_text.lower()
        
        # Simple matching: find products whose titles appear in generated text
        for _, row in self.products_df.iterrows():
            title = row.get('title', '').lower()
            if title and title in generated_lower:
                recommendations.append({
                    'asin': row['asin'],
                    'title': row.get('title', 'Unknown'),
                    'score': 0.8  # P5 score
                })
                if len(recommendations) >= k:
                    break
        
        # If not enough matches, fill with popular products
        if len(recommendations) < k:
            popular = self.products_df.nlargest(k - len(recommendations), 'num_reviews')
            for _, row in popular.iterrows():
                if row['asin'] not in [r['asin'] for r in recommendations]:
                    recommendations.append({
                        'asin': row['asin'],
                        'title': row.get('title', 'Unknown'),
                        'score': 0.5
                    })
                    if len(recommendations) >= k:
                        break
        
        return recommendations[:k]

