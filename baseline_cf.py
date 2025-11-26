"""
Baseline recommendation systems for comparison
1. Popularity-based recommendations
2. Collaborative Filtering (Item-Item similarity)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import pickle


class PopularityRecommender:
    """Simple popularity-based recommender"""
    
    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df
        self.popular_products = None
        self._compute_popularity()
    
    def _compute_popularity(self):
        """Compute popularity scores"""
        # Simple score: average rating * log(num_reviews + 1)
        self.products_df = self.products_df.copy()
        self.products_df['popularity_score'] = (
            self.products_df['avg_rating'] * 
            np.log1p(self.products_df['num_reviews'])
        )
        self.popular_products = self.products_df.nlargest(1000, 'popularity_score')
    
    def recommend(self, user_id: str = None, k: int = 10, exclude_asins: List[str] = None) -> List[Dict]:
        """Recommend popular products"""
        if exclude_asins is None:
            exclude_asins = []
        
        candidates = self.popular_products[
            ~self.popular_products['asin'].isin(exclude_asins)
        ]
        
        recommendations = []
        for _, row in candidates.head(k).iterrows():
            recommendations.append({
                'asin': row['asin'],
                'title': row.get('title', 'Unknown'),
                'score': float(row['popularity_score']),
                'explanation': f"Popular product: {int(row.get('num_reviews', 0))} reviews, {row.get('avg_rating', 0):.1f}★",
                'category': row.get('category', ''),
                'price': row.get('price', 0),
                'image_url': row.get('image_url'),
            })
        
        return recommendations


class CollaborativeFilteringRecommender:
    """
    Item-based Collaborative Filtering using SVD
    Based on the old Rudrendu Paul approach but improved
    """
    
    def __init__(self, reviews_df: pd.DataFrame, products_df: pd.DataFrame, n_components: int = 50):
        """
        Args:
            reviews_df: DataFrame with columns: reviewerID, asin, rating
            products_df: DataFrame with product metadata
            n_components: Number of components for SVD
        """
        self.reviews_df = reviews_df
        self.products_df = products_df
        self.n_components = n_components
        self.utility_matrix = None
        self.product_ids = None
        self.user_ids = None
        self.svd_model = None
        self.decomposed_matrix = None
        self.correlation_matrix = None
        self._build_model()
    
    def _build_model(self):
        """Build collaborative filtering model"""
        print("Building collaborative filtering model...")
        
        # Create utility matrix (users x products)
        self.utility_matrix = self.reviews_df.pivot_table(
            values='rating',
            index='reviewerID',
            columns='asin',
            fill_value=0
        )
        
        self.user_ids = self.utility_matrix.index.values
        self.product_ids = self.utility_matrix.columns.values
        
        print(f"Utility matrix shape: {self.utility_matrix.shape}")
        
        # Transpose for item-based CF (products x users)
        X = self.utility_matrix.T.values
        
        # Apply SVD
        print(f"Applying SVD with {self.n_components} components...")
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.decomposed_matrix = self.svd_model.fit_transform(X)
        
        # Compute correlation matrix
        print("Computing correlation matrix...")
        self.correlation_matrix = np.corrcoef(self.decomposed_matrix)
        
        print("Model built successfully!")
    
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        min_correlation: float = 0.5
    ) -> List[Dict]:
        """
        Recommend products for a user based on their purchase history
        
        Args:
            user_id: User ID
            k: Number of recommendations
            min_correlation: Minimum correlation threshold
        """
        if user_id not in self.user_ids:
            return []
        
        # Get user's purchased products
        user_idx = np.where(self.user_ids == user_id)[0][0]
        user_ratings = self.utility_matrix.iloc[user_idx]
        purchased_asins = user_ratings[user_ratings > 0].index.tolist()
        
        if len(purchased_asins) == 0:
            return []
        
        # Find similar products
        recommendations = {}
        
        for asin in purchased_asins:
            if asin not in self.product_ids:
                continue
            
            product_idx = np.where(self.product_ids == asin)[0][0]
            correlations = self.correlation_matrix[product_idx]
            
            # Get top correlated products
            top_indices = np.argsort(correlations)[::-1][1:101]  # Exclude self
            top_correlations = correlations[top_indices]
            
            for idx, corr in zip(top_indices, top_correlations):
                if corr >= min_correlation:
                    similar_asin = self.product_ids[idx]
                    
                    # Skip if already purchased
                    if similar_asin in purchased_asins:
                        continue
                    
                    # Aggregate scores if product appears multiple times
                    if similar_asin not in recommendations:
                        recommendations[similar_asin] = []
                    recommendations[similar_asin].append(float(corr))
        
        # Average correlations and sort
        final_recommendations = []
        for asin, corrs in recommendations.items():
            avg_corr = np.mean(corrs)
            
            # Get product info
            product_info = self.products_df[
                self.products_df['asin'] == asin
            ].iloc[0].to_dict() if len(self.products_df[
                self.products_df['asin'] == asin
            ]) > 0 else {}
            
            final_recommendations.append({
                'asin': asin,
                'title': product_info.get('title', 'Unknown'),
                'score': avg_corr,
                'explanation': f"High correlation ({avg_corr:.3f}) with products you've purchased",
                'category': product_info.get('category', ''),
                'price': product_info.get('price', 0),
                'image_url': product_info.get('image_url'),
            })
        
        # Sort by score and return top-k
        final_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return final_recommendations[:k]
    
    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'svd_model': self.svd_model,
                'decomposed_matrix': self.decomposed_matrix,
                'correlation_matrix': self.correlation_matrix,
                'product_ids': self.product_ids,
                'user_ids': self.user_ids,
                'utility_matrix': self.utility_matrix
            }, f)
    
    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.svd_model = data['svd_model']
            self.decomposed_matrix = data['decomposed_matrix']
            self.correlation_matrix = data['correlation_matrix']
            self.product_ids = data['product_ids']
            self.user_ids = data['user_ids']
            self.utility_matrix = data['utility_matrix']


def compare_recommendations(
    rec_lm_rag_results: List[Dict],
    baseline_results: List[Dict],
    metric: str = "overlap"
) -> Dict:
    """
    Compare recommendation results
    
    Args:
        rec_lm_rag_results: Results from RecLM-RAG
        baseline_results: Results from baseline
        metric: 'overlap' or 'ndcg'
    
    Returns:
        Comparison metrics
    """
    rec_asins = {r['asin'] for r in rec_lm_rag_results}
    baseline_asins = {r['asin'] for r in baseline_results}
    
    overlap = len(rec_asins & baseline_asins)
    overlap_ratio = overlap / len(rec_asins) if len(rec_asins) > 0 else 0
    
    return {
        'overlap': overlap,
        'overlap_ratio': overlap_ratio,
        'rec_lm_rag_unique': len(rec_asins - baseline_asins),
        'baseline_unique': len(baseline_asins - rec_asins)
    }

