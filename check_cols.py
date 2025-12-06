import pandas as pd
df = pd.read_parquet("data/products_processed.parquet")
print(f"Columns: {df.columns.tolist()}")
