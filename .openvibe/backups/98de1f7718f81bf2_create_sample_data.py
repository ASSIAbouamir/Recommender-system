"""
Script to create complete sample data with reviews for testing
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_complete_sample_data(output_dir: str = "data/sample", num_products: int = 1000, num_reviews: int = 5000):
    """Create complete sample dataset with products and reviews"""
    print("Creating complete sample dataset...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Create products
    products = []
    for i in range(num_products):
        asin = f"B00{i:06d}"
        products.append({
            'asin': asin,
            'title': f"Sample Product {i+1}",
            'description': f"This is a description for sample product {i+1}. It has great features and quality.",
            'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports']),
            'price': round(np.random.uniform(10, 500), 2),
            'avg_rating': round(np.random.uniform(3.5, 5.0), 1),
            'num_reviews': np.random.randint(10, 1000)
        })
    
    products_df = pd.DataFrame(products)
    products_df['text_for_embedding'] = products_df.apply(
        lambda row: f"Title: {row['title']} | Category: {row['category']} | Description: {row['description']}",
        axis=1
    )
    
    # Create reviews
    reviews = []
    user_ids = [f"A{i:09d}" for i in range(100, 100 + num_products // 10)]  # ~100 users
    
    for i in range(num_reviews):
        asin = np.random.choice(products_df['asin'].values)
        reviewer_id = np.random.choice(user_ids)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.3, 0.4])
        
        reviews.append({
            'reviewerID': reviewer_id,
            'asin': asin,
            'rating': float(rating),
            'reviewText': f"This is a review for product {asin}. I found it {'excellent' if rating >= 4 else 'good' if rating >= 3 else 'okay'}.",
            'summary': f"{'Great' if rating >= 4 else 'Good' if rating >= 3 else 'Okay'} product",
            'unixReviewTime': 1609459200 + i * 86400  # Spread over time
        })
    
    reviews_df = pd.DataFrame(reviews)
    
    # Save products as parquet
    products_path = output_dir / "products_sample.parquet"
    products_df.to_parquet(products_path)
    print(f"Created {len(products_df)} products at {products_path}")
    
    # Save reviews as JSONL
    reviews_path = output_dir / "reviews_sample.jsonl"
    with open(reviews_path, 'w', encoding='utf-8') as f:
        for _, review in reviews_df.iterrows():
            json.dump(review.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    print(f"Created {len(reviews_df)} reviews at {reviews_path}")
    
    return products_df, reviews_df

if __name__ == "__main__":
    create_complete_sample_data()

