from src.data_loader import AmazonDataLoader
import pandas as pd

loader = AmazonDataLoader('data', 'Electronics')
loader.load_reviews()
print(f'Reviews loaded: {len(loader.reviews_df)}')

# Manual preprocessing to debug
product_reviews = loader.reviews_df.groupby('asin').agg({
    'rating': ['mean', 'count'],
    'reviewText': lambda x: ' | '.join(x.astype(str)[:10])
}).reset_index()

product_reviews.columns = ['asin', 'avg_rating', 'num_reviews', 'all_reviews']
print(f'\nBefore filtering: {len(product_reviews)} products')
print(f'Sample before filter: {product_reviews.iloc[0].to_dict()}')

# Filter
product_reviews_filtered = product_reviews[
    (product_reviews['num_reviews'] >= 2) &
    (product_reviews['avg_rating'] >= 3.0)
]
print(f'\nAfter filtering (min_reviews=2, min_rating=3.0): {len(product_reviews_filtered)} products')
if len(product_reviews_filtered) > 0:
    print(f'Sample after filter: {product_reviews_filtered.iloc[0].to_dict()}')

# Load metadata
loader.load_metadata()
print(f'\nMetadata: {len(loader.metadata_df)} products')
print(f'Metadata columns: {loader.metadata_df.columns.tolist()}')

# Try merge
merged = product_reviews_filtered.merge(
    loader.metadata_df,
    on='asin',
    how='left',
    suffixes=('', '_meta')
)
print(f'\nAfter merge: {len(merged)} products')
if len(merged) > 0:
    print(f'Sample merged: {merged.iloc[0].to_dict()}')
