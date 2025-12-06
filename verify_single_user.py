import pandas as pd
import yaml
import traceback
from src.data_loader import AmazonDataLoader
from src.recommender import RecLM_RAG, LLMClient
from src.embedder import ProductEmbedder
from src.indexer import FAISSIndexer
from dotenv import load_dotenv

def verify_full():
    load_dotenv()
    print("Loading reviews...")
    loader = AmazonDataLoader(category="Tools_and_Home_Improvement", data_path="data")
    reviews_df = loader.load_reviews()
    
    print("Loading metadata...")
    meta_df = pd.read_parquet("data/products_processed.parquet")
    
    # Ensure image_url column exists as required by RecLM_RAG
    if 'image_url' not in meta_df.columns:
        meta_df['image_url'] = None
    
    print("Merging...")
    full_df = reviews_df.merge(meta_df, on='asin', how='inner')
    print(f"Merged Rows: {len(full_df)}")
    
    user_counts = full_df['reviewerID'].value_counts()
    valid = user_counts[user_counts >= 5].index.tolist()
    
    if valid:
        u = valid[0]
        print(f"User: {u}")
        udata = full_df[full_df['reviewerID'] == u].sort_values('unixReviewTime')
        history_titles = udata['title'].tolist()
        print(f"History: {history_titles}")
        
        # RAG Setup
        print("Configuring RAG...")
        config = yaml.safe_load(open("config.yaml", 'r'))
        
        # Force CPU to avoid 'auto' issues
        config['embedding']['device'] = 'cpu'
        
        llm_client = LLMClient(provider=config['llm']['provider'], model=config['llm']['model'])
        embedder = ProductEmbedder(model_name=config['embedding']['model_name'], device="cpu")
        indexer = FAISSIndexer(embedding_dim=embedder.embedding_dim, index_type=config['index']['index_type'])
        
        rag = RecLM_RAG(
            data_loader=loader,
            embedder=embedder,
            indexer=indexer,
            products_df=meta_df,
            llm_client=llm_client
        )
        
        print("Building Index...")
        # RecLM_RAG expects 'text_for_embedding'
        embs = embedder.embed_products(meta_df['text_for_embedding'].tolist())
        indexer.build_index(embs, meta_df['asin'].values)
        
        print("Running Recommendation...")
        # Construct history string excluding the last item (ground truth)
        hist_str = "User bought: " + ", ".join(history_titles[:-1])
        ground_truth_asin = udata.iloc[-1]['asin']
        print(f"Ground Truth ASIN: {ground_truth_asin}")
        
        try:
            # Use k=10
            # Construct asins list from udata
            user_hist_asins = udata.iloc[:-1]['asin'].tolist()
            recs = rag.recommend(natural_query=hist_str, user_history_asins=user_hist_asins, k=10)
            rec_asins = [r['asin'] for r in recs]
            print(f"Recs: {rec_asins}")
            
            if ground_truth_asin in rec_asins:
                print("SUCCESS: Ground Truth Found!")
            else:
                print("FAILURE: Ground Truth NOT Found.")
                
        except Exception:
            print(traceback.format_exc())
            
    else:
        print("No valid users.")

if __name__ == "__main__":
    verify_full()
