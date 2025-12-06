import sys
from pathlib import Path
sys.path.append(str(Path(".").resolve()))
from src.data_loader import AmazonDataLoader

def test_loading():
    print("Testing AmazonDataLoader with Tools_and_Home_Improvement...")
    try:
        # Note: expecting the directory name "Tools_and_Home_Improvement.json" as category name doesn't feel right
        # but based on the folder structure user showed: data/Tools_and_Home_Improvement.json/Tools_and_Home_Improvement.json
        # The code tries to find {category}.json inside data. 
        # If I pass category="Tools_and_Home_Improvement.json", it might look for data/Tools_and_Home_Improvement.json.json
        # If I pass category="Tools_and_Home_Improvement", it looks for data/Tools_and_Home_Improvement.json which exists as a dir
        # and then my new logic searches INSIDE that dir.
        
        loader = AmazonDataLoader("data", category="Tools_and_Home_Improvement")
        
        print("\n1. Loading Reviews (Limit 100)...")
        reviews = loader.load_reviews(limit=100)
        print(f"Loaded {len(reviews)} reviews")
        print("First review:", reviews.iloc[0].to_dict())
        
        print("\n2. Loading Metadata (Limit 100)...")
        # Metadata might not exist for this dataset or be inside the same folder
        # My code should handle it gracefully or use minimal metadata
        meta = loader.load_metadata(limit=100)
        if meta is not None:
            print(f"Loaded {len(meta)} metadata items")
            print("First meta:", meta.iloc[0].to_dict())
        
        print("\nSUCCESS: Dataset loading verified.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
