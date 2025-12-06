"""
Data loading and preprocessing for Amazon Reviews dataset
Supports multiple formats: JSON, JSONL, CSV
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os


class AmazonDataLoader:
    """Load and preprocess Amazon Reviews dataset"""
    
    def __init__(self, data_path: str, category: str = "Electronics"):
        """
        Args:
            data_path: Path to dataset directory or file
            category: Product category (e.g., "Electronics", "Clothing_Shoes_and_Jewelry")
        """
        self.data_path = Path(data_path)
        self.category = category
        self.reviews_df = None
        self.metadata_df = None
        self.products_df = None
        
    def load_reviews(self, file_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load reviews from JSON/JSONL file
        Format: Each line is a JSON object with keys: reviewerID, asin, reviewText, overall, etc.
        """
        import gzip
        
        if file_path is None:
            # Try to find reviews file
            possible_names = [
                f"reviews_{self.category}.json",
                f"reviews_{self.category}.json.gz",
                f"reviews_{self.category}.jsonl",
                "reviews.json",
                "reviews.jsonl",
                f"{self.category}.json",
                f"{self.category}.json.gz"
            ]
            
            # Check direct paths and nested directories
            search_paths = [self.data_path, self.data_path / f"{self.category}.json"]
            
            for base_path in search_paths:
                if not base_path.exists(): 
                    continue
                    
                for name in possible_names:
                    # Check if base_path is a directory
                    if base_path.is_dir():
                        path = base_path / name
                    # Check if base_path itself matches (if it's a file)
                    elif base_path.name == name:
                        path = base_path
                    else:
                        continue
                        
                    if path.exists():
                        file_path = str(path)
                        break
                if file_path:
                    break
            
            if file_path is None:
                # If we still haven't found it, try checking if the category folder contains ANY json file
                category_dir = self.data_path / f"{self.category}.json"
                if category_dir.exists() and category_dir.is_dir():
                    for ext in ['*.json', '*.json.gz', '*.jsonl']:
                        found = list(category_dir.glob(ext))
                        if found:
                            file_path = str(found[0])
                            break

            if file_path is None:
                raise FileNotFoundError(f"Reviews file for category '{self.category}' not found in {self.data_path}")
        
        print(f"Loading reviews from {file_path}...")
        
        reviews = []
        is_gz = str(file_path).endswith('.gz')
        open_func = gzip.open if is_gz else open
        
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            count = 0
            # Use tqdm only if not in limit mode or if limit is large
            iterator = tqdm(f, desc="Reading reviews") if limit is None or limit > 1000 else f
            
            for line in iterator:
                if limit and count >= limit:
                    break
                try:
                    review = json.loads(line.strip())
                    # Handle both 'rating' and 'overall' field names (different datasets use different names)
                    rating_value = review.get('rating', review.get('overall', 0))
                    reviews.append({
                        'reviewerID': review.get('reviewerID', ''),
                        'asin': review.get('asin', ''),
                        'rating': float(rating_value),
                        'reviewText': review.get('reviewText', ''),
                        'summary': review.get('summary', ''),
                        'unixReviewTime': review.get('unixReviewTime', 0)
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue
        
        self.reviews_df = pd.DataFrame(reviews)
        print(f"Loaded {len(self.reviews_df)} reviews")
        return self.reviews_df
    
    def load_metadata(self, file_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load product metadata
        Format: Each line is a JSON object with keys: asin, title, description, price, category, etc.
        """
        import gzip

        if file_path is None:
            possible_names = [
                f"meta_{self.category}.json",
                f"meta_{self.category}.json.gz",
                f"meta_{self.category}.jsonl",
                "metadata.json",
                "metadata.jsonl",
                "meta.json",
                f"{self.category}_meta.json"
            ]
            
            # Check direct paths and nested directories
            search_paths = [self.data_path, self.data_path / f"{self.category}.json", self.data_path / "metadata"]
            
            for base_path in search_paths:
                if not base_path.exists():
                    continue
                
                for name in possible_names:
                    if base_path.is_dir():
                        path = base_path / name
                    elif base_path.name == name:
                        path = base_path
                    else:
                        continue
                        
                    if path.exists():
                        file_path = str(path)
                        break
                if file_path:
                    break
            
            if file_path is None:
                print("Warning: Metadata file not found. Creating minimal metadata from reviews.")
                return self._create_minimal_metadata()
        
        print(f"Loading metadata from {file_path}...")
        
        metadata = []
        is_gz = str(file_path).endswith('.gz')
        open_func = gzip.open if is_gz else open
        
        with open_func(file_path, 'rt', encoding='utf-8') as f:
            count = 0
            iterator = tqdm(f, desc="Reading metadata") if limit is None or limit > 1000 else f
            
            for line in iterator:
                if limit and count >= limit:
                    break
                try:
                    meta = json.loads(line.strip())
                    
                    # Handle description (can be string or list)
                    description = meta.get('description', '')
                    if isinstance(description, list):
                        description = ' '.join(description)
                    
                    # Handle categories (can be list of lists)
                    categories = meta.get('category', [])
                    if isinstance(categories, list) and len(categories) > 0:
                        if isinstance(categories[0], list):
                            categories = categories[0]  # Take first category path
                        category_str = ' > '.join([str(c) for c in categories[:3]])  # Limit depth
                    else:
                        category_str = self.category
                    
                    metadata.append({
                        'asin': meta.get('asin', ''),
                        'title': meta.get('title', ''),
                        'description': description,
                        'price': meta.get('price', 0),
                        'category': category_str,
                        'brand': meta.get('brand', ''),
                        'imageURL': meta.get('imageURLHighRes', meta.get('imageURL', [])) if meta.get('imageURL') else None,
                        'also_bought': meta.get('also_bought', []),
                        'also_viewed': meta.get('also_viewed', [])
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue
        
        self.metadata_df = pd.DataFrame(metadata)
        print(f"Loaded {len(self.metadata_df)} products")
        return self.metadata_df
    
    def _create_minimal_metadata(self) -> pd.DataFrame:
        """Create minimal metadata from reviews if metadata file not available"""
        if self.reviews_df is None:
            raise ValueError("Reviews must be loaded first")
        
        # Aggregate review info per product
        # Note: We only create description here, not avg_rating/num_reviews
        # since those will come from product_reviews in merge_and_preprocess
        product_info = self.reviews_df.groupby('asin').agg({
            'reviewText': lambda x: ' '.join(x.astype(str).head(5))  # First 5 reviews as description
        }).reset_index()
        
        product_info.columns = ['asin', 'description']
        product_info['title'] = 'Product ' + product_info['asin']
        product_info['price'] = 0
        product_info['category'] = self.category
        product_info['brand'] = ''
        
        self.metadata_df = product_info
        return self.metadata_df
    
    def merge_and_preprocess(self, min_reviews: int = 2, min_rating: float = 3.0) -> pd.DataFrame:
        """
        Merge reviews and metadata, then preprocess
        Returns: DataFrame with one row per product
        """
        if self.reviews_df is None:
            self.load_reviews()
        if self.metadata_df is None:
            self.load_metadata()
        
        print("Merging and preprocessing data...")
        
        # Aggregate reviews per product
        product_reviews = self.reviews_df.groupby('asin').agg({
            'rating': ['mean', 'count'],
            'reviewText': lambda x: ' | '.join(x.astype(str)[:10])  # Top 10 reviews
        }).reset_index()
        
        product_reviews.columns = ['asin', 'avg_rating', 'num_reviews', 'all_reviews']
        
        # Filter products with minimum reviews and rating
        product_reviews = product_reviews[
            (product_reviews['num_reviews'] >= min_reviews) &
            (product_reviews['avg_rating'] >= min_rating)
        ]
        
        # Merge with metadata
        # Use 'left' join to preserve all filtered products even if metadata is incomplete
        products = product_reviews.merge(
            self.metadata_df,
            on='asin',
            how='left',
            suffixes=('', '_meta')
        )
        
        # Fix duplications if they exist
        if 'avg_rating_meta' in products.columns:
            products.drop(columns=['avg_rating_meta'], inplace=True, errors='ignore')
        if 'num_reviews_meta' in products.columns:
            products.drop(columns=['num_reviews_meta'], inplace=True, errors='ignore')
        
        # Fill missing values
        products['title'] = products['title'].fillna('Untitled Product')
        products['description'] = products['description'].fillna(products['all_reviews'])
        products['description'] = products['description'].fillna('')
        products['category'] = products['category'].fillna(self.category)
        
        # Create rich text representation for embedding
        products['text_for_embedding'] = products.apply(
            lambda row: self._create_product_text(row),
            axis=1
        )
        
        # Handle images
        if 'imageURL' in products.columns:
            products['image_url'] = products['imageURL'].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
            )
        else:
            products['image_url'] = None
        
        self.products_df = products
        print(f"Final dataset: {len(self.products_df)} products")
        return self.products_df
    
    def _create_product_text(self, row: pd.Series) -> str:
        """Create rich text representation for product embedding"""
        parts = []
        
        if pd.notna(row.get('title')):
            parts.append(f"Title: {row['title']}")
        
        if pd.notna(row.get('brand')):
            parts.append(f"Brand: {row['brand']}")
        
        if pd.notna(row.get('category')):
            parts.append(f"Category: {row['category']}")
        
        if pd.notna(row.get('description')):
            desc = str(row['description'])
            if len(desc) > 500:
                desc = desc[:500] + "..."
            parts.append(f"Description: {desc}")
        
        # Add review snippets
        if pd.notna(row.get('all_reviews')):
            reviews = str(row['all_reviews'])[:300]
            parts.append(f"Reviews: {reviews}")
        
        return " | ".join(parts)
    
    def get_user_history(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        """Get purchase history for a user"""
        if self.reviews_df is None:
            raise ValueError("Reviews must be loaded first")
        
        user_reviews = self.reviews_df[
            self.reviews_df['reviewerID'] == user_id
        ].sort_values('unixReviewTime', ascending=False)
        
        if len(user_reviews) == 0:
            return pd.DataFrame()
        
        # Get top K most recent and highest rated
        user_reviews = user_reviews.head(top_k * 2)  # Get more for filtering
        user_reviews = user_reviews.nlargest(top_k, 'rating')
        
        # Merge with product metadata
        if self.products_df is not None:
            # Select available columns (image_url might not exist in older parquet files)
            available_cols = ['asin', 'title', 'description', 'category']
            if 'image_url' in self.products_df.columns:
                available_cols.append('image_url')
            if 'price' in self.products_df.columns:
                available_cols.append('price')
            
            user_history = user_reviews.merge(
                self.products_df[available_cols],
                on='asin',
                how='left'
            )
            return user_history
        
        return user_reviews
    
    def get_all_users(self) -> List[str]:
        """Get list of all user IDs"""
        if self.reviews_df is None:
            raise ValueError("Reviews must be loaded first")
        return self.reviews_df['reviewerID'].unique().tolist()
    
    def save_processed(self, output_path: str):
        """Save processed products dataframe"""
        if self.products_df is None:
            raise ValueError("No processed data to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet for efficiency
        self.products_df.to_parquet(output_path.with_suffix('.parquet'))
        print(f"Saved processed data to {output_path.with_suffix('.parquet')}")
    
    def load_processed(self, input_path: str):
        """Load previously processed data"""
        input_path = Path(input_path)
        
        if input_path.suffix == '.parquet':
            self.products_df = pd.read_parquet(input_path)
        else:
            self.products_df = pd.read_csv(input_path)
        
        # Fix column names from previous bad merges
        if 'avg_rating_x' in self.products_df.columns:
            self.products_df.rename(columns={'avg_rating_x': 'avg_rating', 'num_reviews_x': 'num_reviews'}, inplace=True)

        # Ensure image_url column exists (for backward compatibility)
        if 'image_url' not in self.products_df.columns:
            if 'imageURL' in self.products_df.columns:
                self.products_df['image_url'] = self.products_df['imageURL'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
                )
            else:
                self.products_df['image_url'] = None
        
        print(f"Loaded {len(self.products_df)} products from {input_path}")


def load_sample_data(output_dir: str = "data/sample", num_products: int = 1000):
    """
    Create a sample dataset for testing when full dataset is not available
    """
    print("Creating sample dataset for testing...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    np.random.seed(42)
    
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
    
    df = pd.DataFrame(products)
    df['text_for_embedding'] = df.apply(
        lambda row: f"Title: {row['title']} | Category: {row['category']} | Description: {row['description']}",
        axis=1
    )
    
    output_path = output_dir / "products_sample.parquet"
    df.to_parquet(output_path)
    print(f"Created sample dataset at {output_path}")
    
    return df

