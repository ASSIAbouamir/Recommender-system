"""
Experiment utilities: Data splitting and Evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
import torch
from tqdm import tqdm

def temporal_train_test_split(
    reviews_df: pd.DataFrame,
    days_test: int = 1,
    days_val: int = 1,
    gap_days: int = 5  # implicit gap if "train on all but last 7 days" and val/test are last 2 days
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on timestamps:
    Test: Last `days_test`
    Validation: `days_val` before Test
    Train: Everything before (Test + Val + Gap)
    
    Based on article: "training on all but the last 7 days... validation on penultimate day, testing on last day"
    Implies:
    Test = [T-1, T]
    Valid = [T-2, T-1]
    Train = [Start, T-7]
    Gap = [T-7, T-2] (Unused/Ignored)
    """
    print("Performing temporal split...")
    reviews_df = reviews_df.copy()
    
    # Ensure unixReviewTime is int
    if reviews_df['unixReviewTime'].dtype != int:
        reviews_df['unixReviewTime'] = reviews_df['unixReviewTime'].astype(int)
        
    max_time = reviews_df['unixReviewTime'].max()
    day_seconds = 24 * 60 * 60
    
    # Define thresholds
    test_start = max_time - (days_test * day_seconds)
    val_start = test_start - (days_val * day_seconds)
    train_end = max_time - (7 * day_seconds) # Article specific: "except those from the last 7 days"
    
    # Create splits
    train_df = reviews_df[reviews_df['unixReviewTime'] < train_end].copy()
    val_df = reviews_df[
        (reviews_df['unixReviewTime'] >= val_start) & 
        (reviews_df['unixReviewTime'] < test_start)
    ].copy()
    test_df = reviews_df[reviews_df['unixReviewTime'] >= test_start].copy()
    
    print(f"Train set: {len(train_df)} interactions")
    print(f"Val set:   {len(val_df)} interactions")
    print(f"Test set:  {len(test_df)} interactions")
    print(f"Ignored:   {len(reviews_df) - len(train_df) - len(val_df) - len(test_df)} interactions (Gap T-7 to T-2)")
    
    return train_df, val_df, test_df

def cold_start_split(
    full_df: pd.DataFrame,
    min_history: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a cold-start test scenario.
    
    Strategy:
    1. Identify users with NO history in a hypothetical 'train' set.
    2. OR, for the article's specific strict cold start: "separate test set of users with zero historical interactions"
    
    Since we only have one static dataset, standard practice for cold-start sim is:
    Split users into TrainUsers and TestUsers.
    Train on TrainUsers.
    Test on TestUsers (who are effectively 'new' to the model).
    """
    print("Performing cold-start split...")
    user_ids = full_df['reviewerID'].unique()
    np.random.seed(42)
    
    # 10% users for cold-start testing
    test_user_ids = np.random.choice(user_ids, size=int(len(user_ids) * 0.1), replace=False)
    
    test_df = full_df[full_df['reviewerID'].isin(test_user_ids)].copy()
    train_df = full_df[~full_df['reviewerID'].isin(test_user_ids)].copy()
    
    print(f"Cold-Start Train: {len(train_df)} interactions")
    print(f"Cold-Start Test:  {len(test_df)} interactions")
    
    return train_df, test_df

def calculate_metrics(
    recommendations: Dict[str, List[str]], # user_id -> list of recommended ASINs
    ground_truth: Dict[str, List[str]],   # user_id -> list of relevant ASINs (ground truth)
    product_embeddings: Dict[str, np.ndarray], # asin -> embedding vector (for diversity)
    product_popularity: Dict[str, int],   # asin -> purchase count (for long tail)
    k: int = 10,
    total_products: int = 0
) -> Dict[str, float]:
    """
    Calculate generic ranking metrics: Recall@K, NDCG@K, Diversity@K, Long-Tail Ratio
    """
    recalls = []
    ndcgs = []
    diversities = []
    long_tail_counts = 0
    total_recs = 0
    
    # Identify top 5% popular items
    if product_popularity:
        sorted_pop = sorted(product_popularity.items(), key=lambda x: x[1], reverse=True)
        top_5_percent_idx = int(len(sorted_pop) * 0.05)
        head_items = set(t[0] for t in sorted_pop[:top_5_percent_idx])
    else:
        head_items = set()

    for user_id, rec_items in tqdm(recommendations.items(), desc="Evaluating metrics"):
        # Ensure we only consider top k
        rec_items = rec_items[:k]
        relevant_items = set(ground_truth.get(user_id, []))
        
        if not relevant_items:
            # print(f"DEBUG: No ground truth for user {user_id}")
            continue
            
        hits = len(set(rec_items) & relevant_items)
        if hits > 0:
            print(f"DEBUG: Hit for user {user_id}! Recs: {rec_items[:3]}... GT: {list(relevant_items)}")

            
        # --- Recall@K ---
        hits = len(set(rec_items) & relevant_items)
        recall = float(hits / len(relevant_items))
        recalls.append(recall)
        
        # --- NDCG@K ---
        dcg = 0.0
        idcg = 0.0
        
        # DCG
        for i, item in enumerate(rec_items):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG
        num_relevant = min(len(relevant_items), k)
        for i in range(num_relevant):
            idcg += 1.0 / np.log2(i + 2)
            
        ndcg_val = float(dcg / idcg) if idcg > 0 else 0.0
        ndcgs.append(ndcg_val)
        
        # --- Diversity@K ---
        # Mean pairwise cosine distance = 1 - mean pairwise cosine similarity
        if len(rec_items) > 1 and product_embeddings:
            vectors = []
            for item in rec_items:
                if item in product_embeddings:
                    vectors.append(product_embeddings[item])
            
            if len(vectors) > 1:
                vectors = np.array(vectors)
                # Normalize
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / (norms + 1e-8)
                
                # Cosine similarity matrix
                sim_matrix = np.dot(vectors, vectors.T)
                
                # Mean of upper triangle (excluding diagonal)
                triu_indices = np.triu_indices(len(vectors), k=1)
                mean_sim = float(np.mean(sim_matrix[triu_indices]))
                
                diversities.append(float(1.0 - mean_sim)) # Distance
        
        # --- Long-Tail ---
        for item in rec_items:
            total_recs += 1
            if item not in head_items:
                long_tail_counts += 1

    # --- Intra-List Diversity (ILD) ---
    # ILD = average pairwise distance between recommended items
    # Same as Diversity@K but with explicit name
    ild = np.mean(diversities) if diversities else 0.0
    
    # --- Coverage ---
    # Coverage = proportion of unique items recommended across all users
    all_recommended_items = set()
    for rec_items in recommendations.values():
        all_recommended_items.update(rec_items[:k])
    
    coverage = len(all_recommended_items) / total_products if total_products > 0 else 0.0
    
    # Convertir explicitement tous les résultats en float Python natif
    result_dict = {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Diversity@{k}": float(np.mean(diversities)) if diversities else 0.0,
        f"ILD@{k}": float(ild),  # Intra-List Diversity (same as Diversity but explicit)
        "Coverage": float(coverage),
        "Long-Tail Ratio": float(long_tail_counts / total_recs) if total_recs > 0 else 0.0
    }
    
    # Double conversion pour s'assurer que tout est en float Python natif
    return {k: float(v) if isinstance(v, (int, float, np.number)) else v 
            for k, v in result_dict.items()}
