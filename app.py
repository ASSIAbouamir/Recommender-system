"""
Gradio interface for RecLM-RAG recommendation system
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Optional, List
import sys

# Import our modules
try:
    from src.data_loader import AmazonDataLoader
    from src.embedder import ProductEmbedder
    from src.indexer import FAISSIndexer
    from src.recommender import RecLM_RAG, LLMClient
    from src.utils import (
        create_product_card_html,
        format_user_history,
        download_image
    )
    from baseline_cf import PopularityRecommender, CollaborativeFilteringRecommender
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.data_loader import AmazonDataLoader
    from src.embedder import ProductEmbedder
    from src.indexer import FAISSIndexer
    from src.recommender import RecLM_RAG, LLMClient
    from src.utils import (
        create_product_card_html,
        format_user_history,
        download_image
    )
    from baseline_cf import PopularityRecommender, CollaborativeFilteringRecommender


# Global variables (will be initialized in main)
reclm_rag = None
data_loader = None
popularity_baseline = None
cf_baseline = None
products_df = None
all_users = None


def initialize_system(
    data_path: str,
    processed_data_path: str,
    index_path: str,
    model_name: str = "BAAI/bge-large-en-v1.5",
    llm_provider: str = "groq",
    llm_model: str = "llama-3.1-70b-versatile"
):
    """Initialize the recommendation system"""
    global reclm_rag, data_loader, popularity_baseline, cf_baseline, products_df, all_users
    
    print("=" * 60)
    print("Initializing RecLM-RAG System")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = AmazonDataLoader(data_path=data_path)
    
    if Path(processed_data_path).exists():
        print(f"Loading processed data from {processed_data_path}")
        data_loader.load_processed(processed_data_path)
        products_df = data_loader.products_df
        
        # Try to load reviews if they exist
        # Priority: tools_reviews_sample > reviews_sample > reviews
        reviews_file = Path(data_path) / "tools_reviews_sample.jsonl"
        if not reviews_file.exists():
            reviews_file = Path(data_path) / "reviews_sample.jsonl"
        if not reviews_file.exists():
            reviews_file = Path(data_path) / "reviews.jsonl"
        if not reviews_file.exists():
            reviews_file = Path(data_path) / "reviews.json"
        
        if reviews_file.exists():
            print(f"Loading reviews from {reviews_file}")
            try:
                data_loader.load_reviews(str(reviews_file))
            except Exception as e:
                print(f"Could not load reviews: {e}")
    else:
        print("Processing raw data...")
        data_loader.load_reviews()
        data_loader.load_metadata()
        products_df = data_loader.merge_and_preprocess(min_reviews=1)  # Lower threshold for sample data
        if len(products_df) == 0:
            print("Warning: No products after preprocessing. Using all products with at least 1 review.")
            products_df = data_loader.merge_and_preprocess(min_reviews=1, min_rating=0.0)
        data_loader.save_processed(processed_data_path)
    
    # Initialize embedder
    print("\n2. Initializing embedder...")
    embedder = ProductEmbedder(model_name=model_name)
    
    # Load or create index
    print("\n3. Loading/creating FAISS index...")
    indexer = FAISSIndexer(
        embedding_dim=embedder.embedding_dim,
        index_type="IVF4096_HNSW32_PQ64",
        nprobe=64,
        efSearch=128
    )
    
    if Path(index_path).with_suffix('.index').exists():
        print(f"Loading index from {index_path}")
        indexer.load(index_path)
    else:
        print("Creating new index...")
        # Generate embeddings
        embeddings = embedder.embed_products(
            products_df['text_for_embedding'].tolist(),
            cache_key="products"
        )
        
        # Build index
        indexer.build_index(
            embeddings, 
            products_df['asin'].values,
            n_clusters=min(100, len(products_df) // 10)
        )
        indexer.save(index_path)
    
    # Initialize LLM client
    print("\n4. Initializing LLM client...")
    llm_client = LLMClient(provider=llm_provider, model=llm_model)
    
    # Initialize RecLM-RAG
    print("\n5. Initializing RecLM-RAG system...")
    reclm_rag = RecLM_RAG(
        data_loader=data_loader,
        embedder=embedder,
        indexer=indexer,
        products_df=products_df,
        llm_client=llm_client
    )
    
    # Initialize baselines
    print("\n6. Initializing baselines...")
    popularity_baseline = PopularityRecommender(products_df)
    print("Popularity baseline initialized")
    
    cf_baseline = None
    try:
        reviews_df = data_loader.reviews_df
        if reviews_df is not None and len(reviews_df) > 1000:
            print("Initializing collaborative filtering baseline...")
            print(f"Reviews shape: {reviews_df.shape}")
            cf_baseline = CollaborativeFilteringRecommender(
                reviews_df.head(50000),  # Limit for speed
                products_df
            )
            print("CF baseline initialized successfully")
        else:
            print("Not enough reviews for CF baseline")
            cf_baseline = None
    except Exception as e:
        print(f"Could not initialize CF baseline: {e}")
        import traceback
        traceback.print_exc()
        cf_baseline = None
    
    # Get user list
    print("\n7. Getting user list...")
    if data_loader.reviews_df is not None:
        all_users = data_loader.get_all_users()[:1000]  # Limit for UI
        print(f"Found {len(all_users)} users")
    else:
        all_users = []
        print("No users found")
    
    print("\n" + "=" * 60)
    print("System initialized successfully!")
    print(f"Products: {len(products_df)}")
    print(f"Users: {len(all_users)}")
    print("=" * 60 + "\n")


def get_recommendations(
    user_id: Optional[str],
    natural_query: str,
    k: int,
    use_llm: bool,
    sustainability: bool
) -> tuple:
    """Generate recommendations"""
    if reclm_rag is None:
        return "System not initialized. Please check logs.", "", ""
    
    try:
        # Get user history
        user_history = pd.DataFrame()
        if user_id and user_id.strip():
            user_history = data_loader.get_user_history(user_id.strip(), top_k=10)
        
        # Generate recommendations
        recommendations = reclm_rag.recommend(
            user_id=user_id.strip() if user_id and user_id.strip() else None,
            natural_query=natural_query if natural_query.strip() else None,
            k=k,
            use_llm_rerank=use_llm,
            sustainability_mode=sustainability
        )
        
        # Format output
        history_html = format_user_history(user_history)
        
        rec_html = "<h2>Top Recommendations</h2>"
        for i, rec in enumerate(recommendations, 1):
            rec_html += create_product_card_html(
                asin=rec['asin'],
                title=rec.get('title', 'Unknown'),
                score=rec.get('score', 0),
                explanation=rec.get('explanation', ''),
                image_url=rec.get('image_url'),
                price=rec.get('price', 0),
                sustainability_score=rec.get('sustainability_score'),
                sustainability_reason=rec.get('short_reason')
            )
        
        # Comparison with baselines
        comparison_html = "<h3>Baseline Comparisons</h3>"
        
        # Popularity baseline
        if popularity_baseline:
            pop_recs = popularity_baseline.recommend(
                exclude_asins=[rec['asin'] for rec in recommendations[:5]]
            )
            comparison_html += f"<p><strong>Popularity:</strong> {len([r for r in pop_recs if r['asin'] in [rec['asin'] for rec in recommendations]])} overlap with RecLM-RAG</p>"
        
        return rec_html, history_html, comparison_html
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "", ""


def compare_with_baselines(
    user_id: Optional[str],
    k: int = 10
):
    """Compare RecLM-RAG with baselines"""
    if reclm_rag is None:
        return "❌ System not initialized. Please check logs.", ""
    
    try:
        # Validate user_id
        user_id_clean = user_id.strip() if user_id and user_id.strip() else None
        
        if not user_id_clean:
            return "⚠️ Please select or enter a user ID to compare baselines.", ""
        
        # Get user history
        user_history = pd.DataFrame()
        if user_id_clean:
            try:
                user_history = data_loader.get_user_history(user_id_clean, top_k=10)
            except Exception as e:
                print(f"Could not get user history: {e}")
        
        history_html = format_user_history(user_history)
        
        # Get RecLM-RAG recommendations
        try:
            rec_lm_recs = reclm_rag.recommend(
                user_id=user_id_clean, 
                k=k, 
                use_llm_rerank=True
            )
        except Exception as e:
            return f"❌ Error getting RecLM-RAG recommendations: {str(e)}", history_html
        
        if not rec_lm_recs:
            return "⚠️ No recommendations found for this user.", history_html
        
        comparison_text = "## 📊 Comparison Results\n\n"
        comparison_text += f"**User ID:** {user_id_clean}\n\n"
        
        comparison_text += f"### 🚀 RecLM-RAG ({len(rec_lm_recs)} recommendations)\n"
        for i, rec in enumerate(rec_lm_recs, 1):
            comparison_text += f"{i}. **{rec.get('title', 'Unknown')}** (Score: {rec.get('score', 0):.3f})\n"
        
        # Popularity baseline
        if popularity_baseline:
            try:
                pop_recs = popularity_baseline.recommend(user_id=user_id_clean, k=k)
                comparison_text += f"\n### 📈 Popularity Baseline ({len(pop_recs)} recommendations)\n"
                for i, rec in enumerate(pop_recs, 1):
                    comparison_text += f"{i}. **{rec.get('title', 'Unknown')}** (Score: {rec.get('score', 0):.3f})\n"
                
                # Overlap
                rec_asins = {r['asin'] for r in rec_lm_recs}
                pop_asins = {r['asin'] for r in pop_recs}
                overlap = len(rec_asins & pop_asins)
                comparison_text += f"\n**🔄 Overlap: {overlap}/{k} products**\n"
            except Exception as e:
                comparison_text += f"\n### 📈 Popularity Baseline: Error - {str(e)}\n"
        else:
            comparison_text += "\n### 📈 Popularity Baseline: Not available\n"
        
        # CF baseline
        if cf_baseline and user_id_clean:
            try:
                cf_recs = cf_baseline.recommend(user_id=user_id_clean, k=k)
                if cf_recs:
                    comparison_text += f"\n### 🤝 Collaborative Filtering ({len(cf_recs)} recommendations)\n"
                    for i, rec in enumerate(cf_recs, 1):
                        comparison_text += f"{i}. **{rec.get('title', 'Unknown')}** (Score: {rec.get('score', 0):.3f})\n"
                    
                    rec_asins = {r['asin'] for r in rec_lm_recs}
                    cf_asins = {r['asin'] for r in cf_recs}
                    overlap = len(rec_asins & cf_asins)
                    comparison_text += f"\n**🔄 Overlap with CF: {overlap}/{k} products**\n"
                else:
                    comparison_text += "\n### 🤝 Collaborative Filtering: No recommendations for this user\n"
            except Exception as e:
                comparison_text += f"\n### 🤝 Collaborative Filtering: Error - {str(e)}\n"
        else:
            if not cf_baseline:
                comparison_text += "\n### 🤝 Collaborative Filtering: Not initialized\n"
            else:
                comparison_text += "\n### 🤝 Collaborative Filtering: User ID required\n"
        
        return comparison_text, history_html
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return error_msg, ""


# ... existing imports ...
import yaml
import json
import matplotlib.pyplot as plt
import io
import base64

# ... existing code ...

def load_benchmark_results():
    """Load benchmark results from JSON files"""
    results_dir = Path("results")
    if not results_dir.exists():
        return "No results found. Run benchmark first."
    
    results_data = {}
    for file in results_dir.glob("benchmark_*.json"):
        category = file.stem.replace("benchmark_", "")
        with open(file, 'r') as f:
            results_data[category] = json.load(f)
            
    return results_data

def visualize_benchmark(category_results):
    """Create visualization for benchmark results"""
    if not category_results:
        return None
    
    # Extract metrics
    methods = []
    ndcg_scores = []
    recall_scores = []
    
    for method, metrics in category_results.items():
        if isinstance(metrics, dict):
            methods.append(method)
            ndcg_scores.append(metrics.get('NDCG@10', 0))
            recall_scores.append(metrics.get('Recall@10', 0))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, ndcg_scores, width, label='NDCG@10')
    rects2 = ax.bar(x + width/2, recall_scores, width, label='Recall@10')
    
    ax.set_ylabel('Score')
    ax.set_title('Benchmark Results by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return f'<img src="data:image/png;base64,{img_str}" style="width:100%">'

def get_ablation_status():
    """Read config.yaml and return current status"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        status_md = "### Current Experiment Configuration (Ablation Status)\n"
        status_md += f"- **Sustainability Mode:** {'✅ Enabled' if config['recommendation']['sustainability_mode'] else '❌ Disabled (Ablation)'}\n"
        status_md += f"- **User History in Prompt:** {'✅ Enabled' if config['recommendation']['use_history_in_prompt'] else '❌ Disabled (Ablation)'}\n"
        status_md += f"- **LLM Re-ranking:** {'✅ Enabled' if config['recommendation']['use_llm_rerank'] else '❌ Disabled (Ablation)'}\n"
        status_md += f"- **Retrieval K:** {config['recommendation']['retrieval_k']} (Default: 100)\n"
        status_md += f"- **Embedding Model:** `{config['embedding']['model_name']}`\n"
        
        return status_md
    except Exception as e:
        return f"Error reading config: {e}"

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="RecLM-RAG: State-of-the-Art Recommender") as app:
        gr.Markdown("""
        # 🚀 RecLM-RAG: State-of-the-Art Retrieval-Augmented Recommender System
        
        **2025 Edition** - Combining dense retrieval (FAISS) with LLM-powered re-ranking and explanations
        
        ### Features:
        - ✅ History-based recommendations
        - ✅ Zero-shot natural language queries
        - ✅ Sustainability mode (eco-friendly products)
        - ✅ LLM-generated explanations
        - ✅ Comparison with baselines
        """)
        
        with gr.Tabs():
            with gr.Tab("Interactive Demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_id_input = gr.Dropdown(
                            choices=[],  # Will be populated on load
                            label="Select User",
                            allow_custom_value=True,
                            value=None
                        )
                        
                        natural_query = gr.Textbox(
                            label="Natural Language Query (optional)",
                            placeholder="e.g., 'sustainable products similar to my purchases'",
                            value=""
                        )
                        
                        k_slider = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=10,
                            step=1,
                            label="Number of Recommendations"
                        )
                        
                        use_llm = gr.Checkbox(
                            label="Use LLM Re-ranking & Explanations",
                            value=True
                        )
                        
                        sustainability = gr.Checkbox(
                            label="Sustainability Mode (Prioritize Eco-Friendly)",
                            value=False
                        )
                        
                        recommend_btn = gr.Button("Get Recommendations", variant="primary", size="lg")
                        compare_btn = gr.Button("Compare with Baselines", variant="secondary")
                    
                    with gr.Column(scale=2):
                        recommendations_html = gr.HTML(label="Recommendations")
                        
                        with gr.Accordion("User Purchase History", open=False):
                            history_html = gr.Markdown()
                        
                        with gr.Accordion("Baseline Comparison", open=False):
                            comparison_html = gr.Markdown()
                
                # Event handlers
                recommend_btn.click(
                    fn=get_recommendations,
                    inputs=[user_id_input, natural_query, k_slider, use_llm, sustainability],
                    outputs=[recommendations_html, history_html, comparison_html]
                )
                
                compare_btn.click(
                    fn=compare_with_baselines,
                    inputs=[user_id_input, k_slider],
                    outputs=[comparison_html, history_html]
                )
                
                gr.Markdown("""
                ---
                ### How it works:
                1. **Select a user** or enter a custom user ID
                2. **Optionally** provide a natural language query (e.g., "sustainable products")
                3. The system retrieves similar products using dense embeddings (FAISS)
                4. An LLM re-ranks the results and generates personalized explanations
                5. Results are displayed with scores and explanations
                """)

            with gr.Tab("Benchmark Results & Analysis"):
                gr.Markdown("## 📊 Offline Benchmark Results")
                refresh_btn = gr.Button("Refresh Results")
                
                with gr.Row():
                    with gr.Column():
                        ablation_status = gr.Markdown(get_ablation_status())
                        results_json = gr.JSON(label="Raw Metrics")
                    with gr.Column():
                        plot_html = gr.HTML(label="Visualization")
                
                def update_benchmark_view():
                    data = load_benchmark_results()
                    if isinstance(data, str):
                        return data, None, get_ablation_status()
                    
                    # Assume single category for now or pick first
                    cat = list(data.keys())[0] if data else None
                    plot = visualize_benchmark(data[cat]) if cat else None
                    
                    return data, plot, get_ablation_status()
                
                refresh_btn.click(update_benchmark_view, outputs=[results_json, plot_html, ablation_status])
                
                # Load on start
                app.load(update_benchmark_view, outputs=[results_json, plot_html, ablation_status])
        
        # Populate user dropdown on load
        def load_users():
            return gr.Dropdown(choices=all_users if all_users else [])
        
        app.load(load_users, outputs=[user_id_input])
    
    return app



if __name__ == "__main__":
    import argparse
    import yaml
    
    # Load config first to set defaults
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
        
    embedding_config = config.get('embedding', {})
    llm_config = config.get('llm', {})
    data_config = config.get('data', {})
    ui_config = config.get('ui', {})
    
    default_model = embedding_config.get('model_name', "BAAI/bge-large-en-v1.5")
    default_llm_provider = llm_config.get('provider', "groq")
    default_llm_model = llm_config.get('model', "llama-3.1-70b-versatile")
    default_data_path = data_config.get('data_path', "data")
    default_processed_path = data_config.get('processed_path', "data/products_processed.parquet")
    default_index_path = data_config.get('index_path', "index/faiss_index")
    default_port = ui_config.get('port', 7860)
    default_share = ui_config.get('share', False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to dataset")
    parser.add_argument("--processed_path", type=str, default=default_processed_path)
    parser.add_argument("--index_path", type=str, default=default_index_path)
    parser.add_argument("--model", type=str, default=default_model, help="Embedding model")
    parser.add_argument("--llm_provider", type=str, default=default_llm_provider, help="LLM provider")
    parser.add_argument("--llm_model", type=str, default=default_llm_model, help="LLM model")
    parser.add_argument("--port", type=int, default=default_port, help="Gradio port")
    parser.add_argument("--share", action="store_true", default=default_share, help="Create public link")
    
    args = parser.parse_args()
    
    # Initialize system
    initialize_system(
        data_path=args.data_path,
        processed_data_path=args.processed_path,
        index_path=args.index_path,
        model_name=args.model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
    
    # Launch interface
    app = create_interface()
    app.launch(server_port=args.port, share=args.share)

