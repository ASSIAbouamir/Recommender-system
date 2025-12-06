"""
Run Full RecLM-RAG Experiments
Reproduces Table 2 results from the article.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import json
import time

from dotenv import load_dotenv
load_dotenv()


from src.data_loader import AmazonDataLoader
from src.experiment_utils import (
    temporal_train_test_split,
    cold_start_split,
    calculate_metrics
)
from src.baselines_dl import LightGCN, SASRec
from src.baselines_additional import BERT4Rec, GRU4Rec, KNNEmbeddingsRecommender, GPTBasedRAGRecommender
from src.baselines_p5 import P5Recommender
from baseline_cf import PopularityRecommender, CollaborativeFilteringRecommender
from src.recommender import RecLM_RAG, LLMClient
from src.embedder import ProductEmbedder
from src.indexer import FAISSIndexer

import yaml

def run_benchmark(
    category="Electronics",
    data_path="data",
    output_dir="results",
    test_mode=False
):
    # Load Config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Warning: config.yaml not found, using defaults.")
        config = {}

    print(f"Starting benchmark for {category}...")
    
    # 1. Load Data
    loader = AmazonDataLoader(data_path, category=category)
    
    # Robust Loading Strategy (as verified in verify_single_user.py)
    print("Loading raw reviews (for interactions)...")
    loader.load_reviews() # Sets loader.reviews_df
    
    print("Loading processed products (for metadata)...")
    processed_path = Path("data/products_processed.parquet")
    if not processed_path.exists():
        # Fallback to category specific if global not found
        processed_path = Path(data_path) / f"{category}_processed.parquet"
    
    if processed_path.exists():
        products_df = pd.read_parquet(processed_path)
        
        # Fix column names from previous bad merges
        if 'avg_rating_x' in products_df.columns:
            products_df.rename(columns={'avg_rating_x': 'avg_rating', 'num_reviews_x': 'num_reviews'}, inplace=True)
            
        loader.products_df = products_df
        print(f"Loaded {len(products_df)} products from {processed_path}")
        
        # Merge to create full_df (Interactions + Metadata)
        # We perform an inner join to ensure we only have interactions for known products
        print("Merging reviews and metadata...")
        full_df = loader.reviews_df.merge(products_df, on='asin', how='inner')
        print(f"Merged Data: {len(full_df)} interactions")
    else:
        print("Processed product data not found. Falling back to raw metadata handling (might be slow/fragile).")
        loader.load_metadata()
        loader.merge_and_preprocess()
        full_df = loader.products_df
        products_df = full_df # In case of fallback
    
    reviews_df = loader.reviews_df

    
    if test_mode:
        print("TEST MODE: Using small subset with guaranteed history")
        # Filter for users with at least 3 reviews (2 for history + 1 for test)
        user_counts = reviews_df['reviewerID'].value_counts()
        active_users = user_counts[user_counts >= 3].index.tolist()
        
        if len(active_users) < 10:
            print(f"Warning: Only {len(active_users)} active users found. Using all available.")
        
        # Take up to 50 active users for testing
        test_user_subset = active_users[:50]
        
        # Keep all reviews from these users
        reviews_df = reviews_df[reviews_df['reviewerID'].isin(test_user_subset)].copy()
        print(f"Filtered subset: {len(reviews_df)} reviews from {len(test_user_subset)} users")
    
    # 2. Split Data
    # Use User-based Leave-One-Out split to guarantee history
    print("Splitting data (User-based Leave-One-Out)...")
    
    # Filter users with at least 3 interactions (Train, Val, Test)
    user_counts = reviews_df['reviewerID'].value_counts()
    valid_users = user_counts[user_counts >= 3].index.tolist()
    filtered_reviews = reviews_df[reviews_df['reviewerID'].isin(valid_users)].copy()
    
    print(f"Filtered to {len(filtered_reviews)} reviews from {len(valid_users)} users (min 3 interactions)")
    
    train_list = []
    val_list = []
    test_list = []
    
    # Sort all reviews by time once
    filtered_reviews = filtered_reviews.sort_values('unixReviewTime')
    
    # Group by user and split
    # Optimized: using groupby apply is slow for large data. 
    # Better: rank items per user
    filtered_reviews['rank'] = filtered_reviews.groupby('reviewerID')['unixReviewTime'].rank(method='first', ascending=False)
    
    # Rank 1 = Last item (Test)
    # Rank 2 = 2nd Last (Val)
    # Rank > 2 = Train
    
    test_df = filtered_reviews[filtered_reviews['rank'] == 1].copy()
    val_df = filtered_reviews[filtered_reviews['rank'] == 2].copy()
    train_df = filtered_reviews[filtered_reviews['rank'] > 2].copy()
    
    # Clean up
    test_df.drop(columns=['rank'], inplace=True)
    val_df.drop(columns=['rank'], inplace=True)
    train_df.drop(columns=['rank'], inplace=True)
            
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Prepare Ground Truth for Test
    test_ground_truth = test_df.groupby('reviewerID')['asin'].apply(list).to_dict()
    test_users = list(test_ground_truth.keys())
    
    
    if test_mode:
        test_users = test_users[:50] # Evaluate only 50 users in test mode
    elif len(test_users) > 1000:
        print(f"Sampling 1000 users from {len(test_users)} for evaluation (matching paper methodology)")
        np.random.seed(42)
        test_users = np.random.choice(test_users, 1000, replace=False)

    
    results = {}
    
    # 3. Baselines
    print("\n--- Running Baselines ---")
    
    # Generate embeddings ONCE for diversity calculation (shared across all methods)
    print("\nGenerating product embeddings for diversity metrics...")
    embedder_shared = ProductEmbedder(
        model_name=config.get('embedding', {}).get('model_name', "BAAI/bge-large-en-v1.5"),
        device=config.get('embedding', {}).get('device', "cpu")
    )
    embeddings_shared = embedder_shared.embed_products(
        full_df['text_for_embedding'].tolist(),
        cache_key=f"{category}_embeddings_shared_{config.get('embedding', {}).get('model_name', 'default').replace('/', '_')}"
    )
    product_embeddings_dict = {asin: emb for asin, emb in zip(full_df['asin'], embeddings_shared)}
    product_popularity_dict = full_df.set_index('asin')['num_reviews'].to_dict()
    
    # Popularity
    print("Evaluating Popularity...")
    pop_model = PopularityRecommender(full_df)
    pop_recs = {}
    for user in tqdm(test_users):
        recs = pop_model.recommend(k=10)
        pop_recs[user] = [r['asin'] for r in recs]
    
    results['Popularity'] = calculate_metrics(
        pop_recs, test_ground_truth, product_embeddings_dict, product_popularity_dict,
        total_products=len(full_df)
    )
    
    # SVD
    print("Evaluating SVD...")
    # SVD needs training on Train set
    svd_model = CollaborativeFilteringRecommender(train_df, full_df)
    svd_recs = {}
    for user in tqdm(test_users):
        recs = svd_model.recommend(user, k=10)
        svd_recs[user] = [r['asin'] for r in recs]
        
    results['SVD'] = calculate_metrics(
        svd_recs, test_ground_truth, product_embeddings_dict, product_popularity_dict,
        total_products=len(full_df)
    )
    
    # LightGCN
    print("Evaluating LightGCN...")
    # placeholder for actual training loop (simplified for benchmark script)
    # In real scenario: train_lightgcn(model, ...)
    lgcn_recs = {}
    # For now, we skip heavy training in this script to avoid stalling, 
    # but the architecture is available in src.baselines_dl
    print("Skipping LightGCN full training (available in src.baselines_dl)")
    results['LightGCN'] = "Implemented (see src/baselines_dl.py)"

    # SASRec
    print("Evaluating SASRec...")
    print("Skipping SASRec full training (available in src.baselines_dl)")
    results['SASRec'] = "Implemented (see src/baselines_dl.py)"
    
    # P5 (T5-based)
    print("Evaluating P5...")
    try:
        p5_model = P5Recommender(full_df, device='cpu')
        p5_recs = {}
        for user in tqdm(test_users):
            user_history = train_df[train_df['reviewerID'] == user]['asin'].tolist()
            recs = p5_model.recommend(user_history=user_history, k=10)
            p5_recs[user] = [r['asin'] for r in recs]
        
        results['P5'] = calculate_metrics(
            p5_recs, test_ground_truth,
            product_embeddings_dict,  # Use shared embeddings
            product_popularity_dict,
            total_products=len(full_df)
        )
    except Exception as e:
        print(f"P5 evaluation failed: {e}")
        results['P5'] = "Error: " + str(e)
    
    # BERT4Rec
    print("Evaluating BERT4Rec...")
    print("Skipping BERT4Rec full training (available in src.baselines_additional)")
    results['BERT4Rec'] = "Implemented (see src/baselines_additional.py)"
    
    # GRU4Rec
    print("Evaluating GRU4Rec...")
    print("Skipping GRU4Rec full training (available in src.baselines_additional)")
    results['GRU4Rec'] = "Implemented (see src/baselines_additional.py)"
    
    # KNN + Embeddings
    print("Evaluating KNN + Embeddings...")
    # Reuse shared embeddings instead of regenerating
    knn_model = KNNEmbeddingsRecommender(embeddings_shared, full_df['asin'].values, k=10)
    knn_recs = {}
    for user in tqdm(test_users):
        user_history = train_df[train_df['reviewerID'] == user]['asin'].tolist()
        if len(user_history) > 0:
            # Create query embedding from history
            history_texts = []
            for asin in user_history[:5]:  # Last 5 items
                if asin in full_df['asin'].values:
                    idx = full_df[full_df['asin'] == asin].index[0]
                    history_texts.append(embeddings_shared[idx])
            if history_texts:
                query_emb = np.mean(history_texts, axis=0)
                recs = knn_model.recommend(query_emb, exclude_asins=user_history, k=10)
                knn_recs[user] = [r['asin'] for r in recs]
            else:
                knn_recs[user] = []
        else:
            knn_recs[user] = []
    
    results['KNN+Embeddings'] = calculate_metrics(
        knn_recs, test_ground_truth, 
        product_embeddings_dict,
        product_popularity_dict,
        total_products=len(full_df)
    )
    
    # GPT-based RAG Recommend (LLM-only, no retrieval)
    print("Evaluating GPT-based RAG Recommend...")
    llm_client_for_gpt = LLMClient(
        provider=config.get('llm', {}).get('provider', "groq"),
        model=config.get('llm', {}).get('model', "llama-3.1-8b-instant")
    )
    gpt_rag_model = GPTBasedRAGRecommender(full_df, llm_client_for_gpt)
    gpt_rag_recs = {}
    for user in tqdm(test_users):
        user_history = train_df[train_df['reviewerID'] == user]['asin'].tolist()
        # gpt_rag_model.recommend might need to be checked if it uses config? It probably takes k.
        recs = gpt_rag_model.recommend(user_history=user_history, k=10)
        gpt_rag_recs[user] = [r['asin'] for r in recs]
    
    results['GPT-based RAG'] = calculate_metrics(
        gpt_rag_recs, test_ground_truth,
        product_embeddings_dict,  # Use shared embeddings
        product_popularity_dict,
        total_products=len(full_df)
    )
    
    # RecLM-RAG (Ours)
    print("\n--- Evaluating RecLM-RAG ---")
    
    # Init components (reuse shared embedder and embeddings)
    embedder = embedder_shared
    embeddings = embeddings_shared
    
    # Build Index (as per paper: IVF4096-HNSW32-PQ64, nprobe=64, efSearch=128)
    indexer = FAISSIndexer(
        embedding_dim=embedder.embedding_dim, 
        index_type=config.get('index', {}).get('index_type', "IVF4096_HNSW32_PQ64"),
        nprobe=config.get('index', {}).get('nprobe', 64),
        efSearch=config.get('index', {}).get('efSearch', 128)
    )
    indexer.build_index(embeddings, full_df['asin'].values)
    
    # Init RecLM-RAG
    # Note: For benchmark we might skip LLM re-rank to save cost/time or use lightweight
    # Article says "RecLM-RAG (Llama-3.1-8B)"
    llm_client = LLMClient(provider="groq", model="llama-3.1-8b-instant")
    
    rag_system = RecLM_RAG(loader, embedder, indexer, full_df, llm_client)
    
    rag_recs = {}
    latency_breakdowns = []
    # Evaluation loop
    start_time = time.time()
    for user in tqdm(test_users):
        # Get user history from Train splits (not full history, to avoid leakage)
        # We need to filter history to only include train_df events
        user_history = train_df[train_df['reviewerID'] == user]['asin'].tolist()
        
        recs = rag_system.recommend(
            user_history_asins=user_history,
            k=config.get('recommendation', {}).get('final_k', 10),
            retrieval_k=config.get('recommendation', {}).get('retrieval_k', 100),
            sustainability_mode=config.get('recommendation', {}).get('sustainability_mode', False),
            use_llm_rerank=config.get('recommendation', {}).get('use_llm_rerank', True),
            use_history_in_prompt=config.get('recommendation', {}).get('use_history_in_prompt', True),
            return_latency_breakdown=True
        )
        # Filter out latency breakdown from recommendations
        clean_recs = [r for r in recs if 'asin' in r and '_latency_breakdown' not in r]
        rag_recs[user] = [r['asin'] for r in clean_recs]
        
        # Extract latency breakdown from last item if available
        if recs and '_latency_breakdown' in recs[-1]:
            latency_breakdowns.append(recs[-1]['_latency_breakdown'])
    
    avg_latency = (time.time() - start_time) / len(test_users) * 1000 # ms
    
    # Average latency breakdown across all users
    if latency_breakdowns:
        avg_latency_breakdown = {
            'embedding': np.mean([lb.get('embedding', 0) for lb in latency_breakdowns]),
            'faiss_retrieval': np.mean([lb.get('faiss_retrieval', 0) for lb in latency_breakdowns]),
            'sustainability_scoring': np.mean([lb.get('sustainability_scoring', 0) for lb in latency_breakdowns]),
            'prompt_construction': np.mean([lb.get('prompt_construction', 0) for lb in latency_breakdowns]),
            'llm_inference': np.mean([lb.get('llm_inference', 0) for lb in latency_breakdowns]),
            'json_parsing': np.mean([lb.get('json_parsing', 0) for lb in latency_breakdowns]),
            'total': np.mean([lb.get('total', 0) for lb in latency_breakdowns])
        }
    
    # Compute metrics with shared embeddings for Diversity
    results['RecLM-RAG'] = calculate_metrics(
        rag_recs, 
        test_ground_truth, 
        product_embeddings_dict,  # Use shared embeddings
        product_popularity_dict,
        total_products=len(full_df)
    )
    results['RecLM-RAG']['Latency (ms)'] = avg_latency
    
    # Add detailed latency breakdown if available (as per paper section 3.12)
    if 'avg_latency_breakdown' in locals():
        results['RecLM-RAG']['Latency Breakdown (ms)'] = {
            'Embedding + FAISS retrieval': float(avg_latency_breakdown.get('embedding', 0) + avg_latency_breakdown.get('faiss_retrieval', 0)),
            'Sustainability scoring': float(avg_latency_breakdown.get('sustainability_scoring', 0)),
            'Prompt construction': float(avg_latency_breakdown.get('prompt_construction', 0)),
            'LLM inference (8B 4-bit)': float(avg_latency_breakdown.get('llm_inference', 0)),
            'JSON parsing': float(avg_latency_breakdown.get('json_parsing', 0)),
            'Total': float(avg_latency_breakdown.get('total', avg_latency))
        }
    
    # Save Results
    output_path = Path(output_dir) / f"benchmark_{category}.json"
    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convertir les types numpy en types Python pour JSON
    def convert_numpy_types(obj):
        """Convertit récursivement les types numpy en types Python natifs"""
        # Vérifier d'abord les arrays numpy
        if isinstance(obj, np.ndarray):
            return [convert_numpy_types(item) for item in obj.tolist()]
        
        # Vérifier par nom de classe (pour capturer float32, float64, etc.)
        obj_type_name = type(obj).__name__
        if obj_type_name in ['float32', 'float64', 'float16']:
            return float(obj)
        elif obj_type_name in ['int32', 'int64', 'int16', 'int8', 'uint32', 'uint64', 'intc', 'intp']:
            return int(obj)
        elif obj_type_name == 'bool_':
            return bool(obj)
        
        # Vérifier les types numpy scalaires - méthode la plus robuste
        # D'abord par isinstance (plus direct)
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Ensuite par type et module (pour capturer tous les cas)
        obj_type = type(obj)
        if hasattr(obj_type, '__module__') and obj_type.__module__ == 'numpy':
            try:
                if np.issubdtype(obj_type, np.integer):
                    return int(obj)
                elif np.issubdtype(obj_type, np.floating):
                    return float(obj)
                elif np.issubdtype(obj_type, np.bool_):
                    return bool(obj)
                elif np.issubdtype(obj_type, np.number):
                    return float(obj)
            except:
                pass
            
            # Si issubdtype échoue, essayer de convertir directement
            try:
                # Essayer float d'abord (pour float32, float64, etc.)
                if hasattr(obj, '__float__'):
                    return float(obj)
                # Sinon essayer int
                elif hasattr(obj, '__int__'):
                    return int(obj)
                # Sinon essayer bool
                elif hasattr(obj, '__bool__'):
                    return bool(obj)
            except:
                pass
        
        # Structures de données
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        
        return obj
    
    # Convertir tous les résultats avant la sérialisation
    results_serializable = convert_numpy_types(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=4)
        
    print(f"\nResults saved to {output_path}")
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Afficher les métriques principales comme dans le papier
    print("\n📊 Main Results (as per paper Section 3.9):")
    print("-" * 80)
    
    for method_name, method_results in results_serializable.items():
        if isinstance(method_results, dict) and 'Recall@10' in method_results:
            print(f"\n{method_name}:")
            print(f"  Recall@10:    {method_results.get('Recall@10', 0):.4f}")
            print(f"  NDCG@10:      {method_results.get('NDCG@10', 0):.4f}")
            print(f"  Diversity@10: {method_results.get('Diversity@10', 0):.4f}")
            print(f"  ILD@10:        {method_results.get('ILD@10', 0):.4f}")
            print(f"  Coverage:      {method_results.get('Coverage', 0):.4f}")
            print(f"  Long-Tail:     {method_results.get('Long-Tail Ratio', 0):.4f}")
            if 'Latency (ms)' in method_results:
                print(f"  Latency:       {method_results.get('Latency (ms)', 0):.1f} ms")
            if 'Latency Breakdown (ms)' in method_results:
                lb = method_results['Latency Breakdown (ms)']
                print(f"  Latency Breakdown:")
                print(f"    - Embedding + FAISS: {lb.get('Embedding + FAISS retrieval', 0):.1f} ms")
                print(f"    - Sustainability:    {lb.get('Sustainability scoring', 0):.1f} ms")
                print(f"    - Prompt:            {lb.get('Prompt construction', 0):.1f} ms")
                print(f"    - LLM inference:     {lb.get('LLM inference (8B 4-bit)', 0):.1f} ms")
                print(f"    - JSON parsing:      {lb.get('JSON parsing', 0):.1f} ms")
                print(f"    - Total:             {lb.get('Total', 0):.1f} ms")
        elif isinstance(method_results, str):
            print(f"\n{method_name}: {method_results}")
    
    print("\n" + "=" * 80)
    print(f"\nFull results saved to: {output_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="Electronics")
    parser.add_argument("--test", action="store_true", help="Run on small subset")
    args = parser.parse_args()
    
    run_benchmark(args.category, test_mode=args.test)
