from src.data_loader import AmazonDataLoader

loader = AmazonDataLoader('data', 'Electronics')
loader.load_reviews()

print(f'Reviews loaded: {len(loader.reviews_df)}')
print(f'Columns: {loader.reviews_df.columns.tolist()}')
print(f'\nFirst 5 reviews:')
print(loader.reviews_df[['reviewerID', 'asin', 'rating']].head())
print(f'\nRating stats:')
print(loader.reviews_df['rating'].describe())
print(f'\nUnique ratings: {sorted(loader.reviews_df["rating"].unique())}')
