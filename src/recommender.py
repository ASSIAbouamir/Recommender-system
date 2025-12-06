"""
Main RecLM-RAG recommendation system
Combines dense retrieval (FAISS) with LLM-based re-ranking and explanation generation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import os

from .data_loader import AmazonDataLoader
from .embedder import ProductEmbedder, create_rich_query_text
from .indexer import FAISSIndexer
from .utils import (
    create_rag_prompt,
    parse_llm_response,
    extract_sustainability_keywords,
    generate_search_criteria,
    evaluate_sustainability_score
)


class LLMClient:
    """Interface for LLM API calls"""
    
    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None
    ):
        """
        Args:
            provider: 'groq', 'openai', or 'openrouter'
            model: Model name
            api_key: API key (or set via environment variable)
        """
        self.provider = provider
        self.model = model
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var)
            if not self.api_key:
                print(f"Warning: {env_var} not set. LLM features will not work.")
        
        # Initialize client based on provider
        if provider == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key) if self.api_key else None
            except ImportError:
                print("groq package not installed. Install with: pip install groq")
                self.client = None
        elif provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            except ImportError:
                print("openai package not installed. Install with: pip install openai")
                self.client = None
        elif provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.api_key
                ) if self.api_key else None
            except ImportError:
                print("openai package not installed. Install with: pip install openai")
                self.client = None
        else:
            self.client = None
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """Generate response from LLM"""
        if not self.client:
            return '{"recommendations": []}'  # Return empty if no client
        
        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful product recommendation assistant. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider in ["openai", "openrouter"]:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful product recommendation assistant. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return '{"recommendations": []}'


class RecLM_RAG:
    """Main recommendation system combining retrieval and generation"""
    
    def __init__(
        self,
        data_loader: AmazonDataLoader,
        embedder: ProductEmbedder,
        indexer: FAISSIndexer,
        products_df: pd.DataFrame,
        llm_client: Optional[LLMClient] = None
    ):
        """
        Args:
            data_loader: DataLoader instance
            embedder: ProductEmbedder instance
            indexer: FAISSIndexer instance
            products_df: DataFrame with product information
            llm_client: Optional LLMClient for re-ranking and explanations
        """
        self.data_loader = data_loader
        self.embedder = embedder
        self.indexer = indexer
        self.products_df = products_df
        self.llm_client = llm_client
        
        # Ensure image_url column exists
        if 'image_url' not in products_df.columns:
            products_df['image_url'] = None
        
        # Create ASIN to product mapping for fast lookup
        self.product_map = {}
        for _, row in products_df.iterrows():
            self.product_map[row['asin']] = row.to_dict()
    
    def recommend(
        self,
        user_id: Optional[str] = None,
        user_history_asins: Optional[List[str]] = None,
        natural_query: Optional[str] = None,
        k: int = 10,
        retrieval_k: int = 100,
        sustainability_mode: bool = False,
        use_llm_rerank: bool = True,
        use_history_in_prompt: bool = True,
        return_latency_breakdown: bool = False
    ) -> List[Dict]:
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID (if None, use user_history_asins)
            user_history_asins: List of ASINs user has purchased
            natural_query: Optional natural language query (e.g., "sustainable products")
            k: Number of final recommendations
            retrieval_k: Number of candidates to retrieve before LLM re-ranking
            sustainability_mode: Prioritize sustainable products
            use_llm_rerank: Whether to use LLM for re-ranking (False = just return top-k from retrieval)
            return_latency_breakdown: If True, return latency breakdown dict
        
        Returns:
            List of recommendation dicts with keys: asin, title, score, explanation, etc.
            If return_latency_breakdown=True, last item is dict with latency breakdown
        """
        import time
        latency = {
            'embedding': 0.0,
            'faiss_retrieval': 0.0,
            'sustainability_scoring': 0.0,
            'prompt_construction': 0.0,
            'llm_inference': 0.0,
            'json_parsing': 0.0,
            'total': 0.0
        }
        start_total = time.time()
        # Get user history
        if user_id:
            user_history = self.data_loader.get_user_history(user_id, top_k=20)
            user_history_asins = user_history['asin'].tolist()
        elif user_history_asins:
            # Get full product info for provided ASINs
            user_history = self.products_df[
                self.products_df['asin'].isin(user_history_asins)
            ].copy()
        else:
            # No history - zero-shot recommendation
            user_history = pd.DataFrame()
            user_history_asins = []
        
        # Create query embedding (as per paper: Input Encoding section)
        start_emb = time.time()
        if natural_query:
            # Query mode: use query directly
            query_embedding = self.embedder.embed_query(natural_query)
            query_summary = natural_query
        elif len(user_history_asins) > 0:
            # History mode: "User bought: " + concat(titles) as per paper
            titles = []
            for asin in user_history_asins:
                if asin in self.product_map:
                    product = self.product_map[asin]
                    title = product.get('title', '')
                    if title:
                        titles.append(title)
            
            if titles:
                # Paper format: "User bought: " + concat(titles)
                history_input = "User bought: " + ", ".join(titles)
                query_embedding = self.embedder.embed_query(history_input)
                query_summary = history_input
            else:
                # Fallback: use text_for_embedding
                history_texts = []
                for asin in user_history_asins:
                    if asin in self.product_map:
                        product = self.product_map[asin]
                        history_texts.append(product.get('text_for_embedding', ''))
                query_embedding = self.embedder.create_query_embedding(history_texts)
                query_summary = create_rich_query_text(user_history, natural_query)
        else:
            # No history and no query: return popular products
            return self._get_popular_recommendations(k)
        
        latency['embedding'] = (time.time() - start_emb) * 1000  # ms
        
        # Retrieve candidates
        start_faiss = time.time()
        distances, indices = self.indexer.search(
            query_embedding,
            k=retrieval_k,
            filter_ids=np.array(user_history_asins) if user_history_asins else None
        )
        latency['faiss_retrieval'] = (time.time() - start_faiss) * 1000  # ms
        
        # Get product IDs
        candidate_asins = self.indexer.get_product_ids(indices)
        
        # Prepare candidate products for LLM
        if len(candidate_asins) > 0:
            print(f"DEBUG: Retrieval found {len(candidate_asins)} candidates. Top 5: {candidate_asins[:5]}")
        else:
            print("DEBUG: Retrieval found 0 candidates!")

        start_sust = time.time()

        candidate_products = []
        for asin, score in zip(candidate_asins, distances):
            if asin in self.product_map:
                product = self.product_map[asin].copy()
                product['score'] = float(score)
                
                # Add sustainability score (always computed as per algorithm)
                # Combine title + description for keywords
                full_text = f"{product.get('title', '')} {product.get('description', '')}"
                sust_eval = evaluate_sustainability_score(
                    title=product.get('title', ''),
                    description=product.get('description', ''),
                    brand=product.get('brand', ''),
                    features=str(product.get('category', '')),
                    keywords=extract_sustainability_keywords(full_text)
                )
                product['sustainability_score'] = sust_eval['sustainability_score']
                product['sustainability_reason'] = sust_eval['short_reason']
                
                candidate_products.append(product)
        latency['sustainability_scoring'] = (time.time() - start_sust) * 1000  # ms
        
        # Re-rank with LLM if enabled
        if use_llm_rerank and self.llm_client:
            # Ablation Logic: Mask history if disabled
            rerank_history = user_history if use_history_in_prompt else pd.DataFrame()
            rerank_query = query_summary
            if not use_history_in_prompt and not natural_query:
                rerank_query = "Recommend best matching products"

            result = self._llm_rerank(
                rerank_history,
                candidate_products,
                rerank_query,
                sustainability_mode,
                k,
                return_latency=True
            )
            if isinstance(result, tuple):
                recommendations, llm_latency = result
                latency.update(llm_latency)
            else:
                recommendations = result
        else:
            # Just return top-k from retrieval
            recommendations = []
            for product in candidate_products[:k]:
                # Evaluate sustainability if in sustainability mode
                sustainability_info = {}
                if sustainability_mode:
                    sustainability_info = evaluate_sustainability_score(
                        title=product.get('title', ''),
                        description=product.get('description', ''),
                        brand=product.get('brand', ''),
                        features=str(product.get('category', '')),
                        keywords=extract_sustainability_keywords(
                            f"{product.get('title', '')} {product.get('description', '')}"
                        ),
                        llm_client=self.llm_client if self.llm_client else None
                    )
                
                recommendations.append({
                    'asin': product['asin'],
                    'title': product.get('title', 'Unknown'),
                    'score': product['score'],
                    'explanation': f"Similarity score: {product['score']:.3f}",
                    'category': product.get('category', ''),
                    'price': product.get('price', 0),
                    'image_url': product.get('image_url'),
                    **sustainability_info,  # Add sustainability_score and short_reason
                    **product
                })
        
        latency['total'] = (time.time() - start_total) * 1000  # ms
        
        if return_latency_breakdown:
            # Add latency breakdown as metadata to last recommendation
            if recommendations:
                recommendations[-1]['_latency_breakdown'] = latency
            else:
                recommendations.append({'_latency_breakdown': latency})
        
        return recommendations
    
    def _llm_rerank(
        self,
        user_history: pd.DataFrame,
        candidate_products: List[Dict],
        query: str,
        sustainability_mode: bool,
        k: int,
        return_latency: bool = False
    ) -> tuple:
        """Use LLM to re-rank candidates and generate explanations"""
        import time
        import json
        import re
        latency = {
            'prompt_construction': 0.0,
            'llm_inference': 0.0,
            'json_parsing': 0.0
        }
        # Create user history summary
        if len(user_history) > 0:
            history_summary = f"User has purchased {len(user_history)} products:\n"
            for _, row in user_history.head(10).iterrows():
                history_summary += f"- {row.get('title', 'Unknown')} (Rating: {row.get('rating', 'N/A')}/5)\n"
        else:
            history_summary = "No purchase history available."
        
        # Generate precise search criteria from query and history
        search_criteria = generate_search_criteria(
            user_query=query,
            user_history=user_history if len(user_history) > 0 else None,
            sustainability_mode=sustainability_mode,
            llm_client=self.llm_client if self.llm_client else None
        )
        
        # Create RAG prompt with search criteria
        start_prompt = time.time()
        prompt = create_rag_prompt(
            history_summary,
            candidate_products,
            query=query,
            sustainability_mode=sustainability_mode,
            search_criteria=search_criteria
        )
        latency['prompt_construction'] = (time.time() - start_prompt) * 1000  # ms
        
        # Call LLM
        print("Calling LLM for re-ranking and explanation generation...")
        start_llm = time.time()
        response = self.llm_client.generate(prompt)
        latency['llm_inference'] = (time.time() - start_llm) * 1000  # ms
        
        # Parse response
        start_parse = time.time()
        llm_recommendations = parse_llm_response(response)
        latency['json_parsing'] = (time.time() - start_parse) * 1000  # ms
        
        # Map LLM recommendations back to full product info
        final_recommendations = []
        seen_asins = set()
        
        # First, add LLM-recommended products
        for rec in llm_recommendations[:k]:
            asin = rec.get('asin', '')
            if asin in self.product_map and asin not in seen_asins:
                product = self.product_map[asin].copy()
                
                # Evaluate sustainability if in sustainability mode
                sustainability_info = {}
                if sustainability_mode:
                    sustainability_info = evaluate_sustainability_score(
                        title=product.get('title', ''),
                        description=product.get('description', ''),
                        brand=product.get('brand', ''),
                        features=str(product.get('category', '')),
                        keywords=extract_sustainability_keywords(
                            f"{product.get('title', '')} {product.get('description', '')}"
                        ),
                        llm_client=self.llm_client if self.llm_client else None
                    )
                
                final_recommendations.append({
                    'asin': asin,
                    'title': product.get('title', 'Unknown'),
                    'score': product.get('score', 0.0),
                    'explanation': rec.get('explanation', 'No explanation provided.'),
                    'category': product.get('category', ''),
                    'price': product.get('price', 0),
                    'image_url': product.get('image_url'),
                    **sustainability_info,  # Add sustainability_score and short_reason
                    **product
                })
                seen_asins.add(asin)
        
        # Fill remaining slots with top retrieval results if needed
        if len(final_recommendations) < k:
            for product in candidate_products:
                if product['asin'] not in seen_asins:
                    # Evaluate sustainability if in sustainability mode
                    sustainability_info = {}
                    if sustainability_mode:
                        sustainability_info = evaluate_sustainability_score(
                            title=product.get('title', ''),
                            description=product.get('description', ''),
                            brand=product.get('brand', ''),
                            features=str(product.get('category', '')),
                            keywords=extract_sustainability_keywords(
                                f"{product.get('title', '')} {product.get('description', '')}"
                            ),
                            llm_client=self.llm_client if self.llm_client else None
                        )
                    
                    final_recommendations.append({
                        'asin': product['asin'],
                        'title': product.get('title', 'Unknown'),
                        'score': product['score'],
                        'explanation': f"High similarity to your preferences (score: {product['score']:.3f})",
                        'category': product.get('category', ''),
                        'price': product.get('price', 0),
                        'image_url': product.get('image_url'),
                        **sustainability_info,  # Add sustainability_score and short_reason
                        **product
                    })
                    seen_asins.add(product['asin'])
                    if len(final_recommendations) >= k:
                        break
        
        if return_latency:
            return final_recommendations[:k], latency
        return final_recommendations[:k]
    
    def _get_popular_recommendations(self, k: int) -> List[Dict]:
        """Fallback: return popular products"""
        popular = self.products_df.nlargest(k, 'num_reviews')
        recommendations = []
        for _, row in popular.iterrows():
            recommendations.append({
                'asin': row['asin'],
                'title': row.get('title', 'Unknown'),
                'score': float(row.get('avg_rating', 0)),
                'explanation': f"Popular product with {int(row.get('num_reviews', 0))} reviews",
                'category': row.get('category', ''),
                'price': row.get('price', 0),
                'image_url': row.get('image_url'),
            })
        return recommendations

