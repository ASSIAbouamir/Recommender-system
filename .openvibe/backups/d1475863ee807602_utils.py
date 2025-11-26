"""
Utility functions for RecLM-RAG
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Any
import requests
from io import BytesIO
from PIL import Image
import re
from pathlib import Path


def download_image(url: str, max_size: int = 224) -> Optional[Image.Image]:
    """Download and resize product image"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.thumbnail((max_size, max_size))
            return img
    except Exception as e:
        pass
    return None


def create_product_card_html(
    asin: str,
    title: str,
    score: float,
    explanation: str,
    image_url: Optional[str] = None,
    price: Optional[float] = None,
    sustainability_score: Optional[float] = None,
    sustainability_reason: Optional[str] = None
) -> str:
    """Create HTML card for product display"""
    image_html = ""
    if image_url:
        image_html = f'<img src="{image_url}" style="max-width:150px;max-height:150px;" />'
    else:
        image_html = '<div style="width:150px;height:150px;background:#f0f0f0;display:flex;align-items:center;justify-content:center;">No Image</div>'
    
    price_html = f"${price:.2f}" if price else "N/A"
    
    # Sustainability badge
    sustainability_html = ""
    if sustainability_score is not None:
        # Color based on score
        if sustainability_score >= 8:
            color = "#28a745"  # Green
        elif sustainability_score >= 6:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        sustainability_html = f"""
        <div style="margin-top:8px;padding:8px;background:#f8f9fa;border-radius:4px;border-left:3px solid {color};">
            <p style="margin:0;font-size:12px;font-weight:bold;color:{color};">
                🌱 Durabilité: {sustainability_score}/10
            </p>
            <p style="margin:4px 0 0 0;font-size:11px;color:#666;">
                {sustainability_reason or "Évaluation en cours"}
            </p>
        </div>
        """
    
    html = f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:15px;margin:10px;background:white;">
        <div style="display:flex;gap:15px;">
            <div>{image_html}</div>
            <div style="flex:1;">
                <h3 style="margin:0 0 10px 0;font-size:16px;">{title[:80]}</h3>
                <p style="color:#666;font-size:12px;margin:5px 0;">ASIN: {asin}</p>
                <p style="color:#28a745;font-size:14px;font-weight:bold;margin:5px 0;">Score: {score:.3f} | Price: {price_html}</p>
                {sustainability_html}
                <p style="font-size:13px;color:#555;margin-top:10px;">{explanation}</p>
            </div>
        </div>
    </div>
    """
    return html


def format_user_history(history_df: pd.DataFrame) -> str:
    """Format user history for display"""
    if len(history_df) == 0:
        return "No purchase history"
    
    items = []
    for _, row in history_df.iterrows():
        title = row.get('title', 'Unknown Product')
        rating = row.get('rating', 'N/A')
        items.append(f"• {title} (Rating: {rating}/5)")
    
    return "\n".join(items[:10])  # Show top 10


def extract_sustainability_keywords(text: str) -> List[str]:
    """Extract sustainability-related keywords"""
    keywords = [
        'organic', 'recycled', 'sustainable', 'eco-friendly', 'vegan',
        'biodegradable', 'compostable', 'carbon neutral', 'fair trade',
        'green', 'environmentally friendly', 'renewable', 'ethical'
    ]
    
    found = []
    text_lower = text.lower()
    for keyword in keywords:
        if keyword in text_lower:
            found.append(keyword)
    
    return found


def evaluate_sustainability_score(
    title: str,
    description: str = "",
    brand: str = "",
    features: str = "",
    keywords: Optional[List[str]] = None,
    llm_client: Optional[object] = None
) -> Dict[str, any]:
    """
    Evaluate product sustainability score from 0 to 10
    
    Args:
        title: Product title
        description: Product description
        brand: Brand name
        features: Product features
        keywords: Detected sustainability keywords
        llm_client: Optional LLM client for advanced evaluation
    
    Returns:
        Dict with 'sustainability_score' (float) and 'short_reason' (str)
    """
    # Extract keywords if not provided
    if keywords is None:
        full_text = f"{title} {description} {features} {brand}".lower()
        keywords = extract_sustainability_keywords(full_text)
    
    # Create evaluation prompt
    keywords_text = ", ".join(keywords) if keywords else "Aucun"
    
    evaluation_prompt = f"""Tu es un expert en durabilité et éco-responsabilité produit.

Évalue ce produit de 0 à 10 en termes de durabilité réelle (et pas juste greenwashing).

Critères pris en compte :

- Matériaux recyclés, biosourcés, certifiés (GOTS, OEKO-TEX, etc.)

- Empreinte carbone de fabrication et transport

- Durabilité et réparabilité

- Certifications officielles

- Politique de la marque (B-Corp, 1% for the Planet, etc.)

- Avis clients mentionnant la longévité ou l'éthique

Réponds EXACTEMENT avec ce format (rien d'autre) :

{{"sustainability_score": 8.7, "short_reason": "Cuir de pomme recyclé, marque B-Corp, fabriqué en Italie, réparable à vie"}}

Produit :

Titre : {title}

Description : {description}

Marque : {brand}

Caractéristiques : {features}

Mots-clés détectés : {keywords_text}
"""
    
    # Try to use LLM if available
    if llm_client and hasattr(llm_client, 'generate'):
        try:
            response = llm_client.generate(evaluation_prompt, temperature=0.2, max_tokens=500)
            # Parse JSON response
            json_match = re.search(r'\{.*?"sustainability_score".*?\}', response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(0))
                score = float(eval_data.get('sustainability_score', 0))
                reason = eval_data.get('short_reason', '')
                # Ensure score is between 0 and 10
                score = max(0, min(10, score))
                return {
                    'sustainability_score': score,
                    'short_reason': reason
                }
        except Exception as e:
            print(f"Error evaluating sustainability with LLM: {e}, falling back to rule-based")
    
    # Fallback to rule-based evaluation
    full_text = f"{title} {description} {features} {brand}".lower()
    score = 5.0  # Base score
    reasons = []
    
    # Material certifications (high weight)
    if any(cert in full_text for cert in ['gots', 'oeko-tex', 'bluesign', 'cradle to cradle']):
        score += 2.0
        reasons.append("certification matériaux")
    
    # Recycled materials
    if any(word in full_text for word in ['recycled', 'recyclé', 'upcycled', 'récupéré']):
        score += 1.5
        reasons.append("matériaux recyclés")
    
    # Bio-based materials
    if any(word in full_text for word in ['bio', 'organic', 'biologique', 'biosourcé', 'cork', 'liège']):
        score += 1.0
        reasons.append("matériaux biosourcés")
    
    # Brand certifications
    if any(cert in full_text for cert in ['b-corp', 'b corp', '1% for the planet', 'fair trade', 'commerce équitable']):
        score += 1.5
        reasons.append("marque certifiée éthique")
    
    # Repairability
    if any(word in full_text for word in ['réparable', 'repairable', 'réparation', 'garantie', 'warranty']):
        score += 0.5
        reasons.append("réparable")
    
    # Local/EU production
    if any(word in full_text for word in ['made in france', 'fabriqué en france', 'made in europe', 'fabriqué en europe', 'local']):
        score += 0.5
        reasons.append("production locale")
    
    # Vegan/animal-free
    if any(word in full_text for word in ['vegan', 'végan', 'cruelty-free', 'sans cruauté']):
        score += 0.5
        reasons.append("végan")
    
    # Durability keywords
    if any(word in full_text for word in ['durable', 'durability', 'long-lasting', 'résistant', 'robust']):
        score += 0.5
        reasons.append("durable")
    
    # Negative indicators (greenwashing)
    if any(word in full_text for word in ['green', 'eco', 'éco']) and not any(word in full_text for word in ['certified', 'certifié', 'recycled', 'recyclé']):
        score -= 0.5  # Possible greenwashing
    
    # Ensure score is between 0 and 10
    score = max(0, min(10, score))
    
    # Generate short reason
    if reasons:
        short_reason = ", ".join(reasons[:3])  # Top 3 reasons
    else:
        short_reason = "Évaluation standard"
    
    return {
        'sustainability_score': round(score, 1),
        'short_reason': short_reason
    }


def calculate_ndcg(recommended_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Calculate NDCG@k metric"""
    if len(relevant_ids) == 0:
        return 0.0
    
    # Take top k
    recommended_ids = recommended_ids[:k]
    relevant_set = set(relevant_ids)
    
    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(recommended_ids):
        if item_id in relevant_set:
            rel = 1.0
            dcg += rel / np.log2(i + 2)  # i+2 because index starts at 0
    
    # Calculate IDCG (ideal DCG)
    idcg = 0.0
    num_relevant = min(len(relevant_ids), k)
    for i in range(num_relevant):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def clean_text(text: str) -> str:
    """Clean text for display"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def generate_search_criteria(
    user_query: Optional[str] = None,
    user_history: Optional[pd.DataFrame] = None,
    sustainability_mode: bool = False,
    llm_client: Optional[object] = None
) -> List[str]:
    """
    Generate precise search criteria from user query and history
    
    Args:
        user_query: Natural language query from user
        user_history: DataFrame with user's purchase history
        sustainability_mode: Whether to prioritize sustainable products
        llm_client: Optional LLM client for advanced criteria generation
    
    Returns:
        List of precise search criteria
    """
    # Get last 5 products from history
    last_5_products = []
    if user_history is not None and len(user_history) > 0:
        last_5_products = user_history.head(5)['title'].tolist()
    
    # Create prompt for criteria generation
    history_text = "\n".join([f"- {title}" for title in last_5_products]) if last_5_products else "Aucun historique disponible"
    
    criteria_prompt = f"""Tu es un expert en shopping personnalisé ultra-précis.

À partir de cette requête utilisateur (ou historique d'achats), génère 8 à 12 critères de recherche très précis et variés que l'utilisateur recherche implicitement.

Réponds EXACTEMENT sous ce format JSON (rien d'autre, pas de ```json) :

{{
  "expanded_criteria": [
    "critère 1",
    "critère 2",
    ...
  ]
}}

Requête utilisateur : "{user_query if user_query else 'Aucune requête spécifique'}"

Historique récent (si disponible) : {history_text}
"""
    
    # Try to use LLM if available
    if llm_client and hasattr(llm_client, 'generate'):
        try:
            response = llm_client.generate(criteria_prompt, temperature=0.3, max_tokens=1000)
            # Parse JSON response
            json_match = re.search(r'\{.*?"expanded_criteria".*?\}', response, re.DOTALL)
            if json_match:
                criteria_data = json.loads(json_match.group(0))
                criteria = criteria_data.get('expanded_criteria', [])
                if criteria and len(criteria) >= 8:
                    return criteria[:12]
        except Exception as e:
            print(f"Error generating criteria with LLM: {e}, falling back to rule-based")
    
    # Fallback to rule-based criteria extraction
    criteria = []
    
    if user_query:
        query_lower = user_query.lower()
        
        # Extract price information
        if any(word in query_lower for word in ['sous', 'moins de', 'inférieur', 'budget', 'prix']):
            criteria.append("prix raisonnable ou dans le budget")
        
        # Extract material information
        if 'végan' in query_lower or 'vegan' in query_lower:
            criteria.append("cuir végan ou matériaux alternatifs")
        if 'recyclé' in query_lower or 'recycled' in query_lower:
            criteria.append("matériaux recyclés")
        if 'cuir' in query_lower:
            criteria.append("cuir authentique ou alternative végan")
        
        # Extract sustainability
        if any(word in query_lower for word in ['durable', 'sustainable', 'éco', 'eco', 'bio', 'organic']):
            criteria.append("produit durable et éco-responsable")
            criteria.append("marque certifiée B-Corp ou Fair Trade")
        
        # Extract color
        colors = ['noir', 'blanc', 'rouge', 'bleu', 'vert', 'marron', 'beige', 'gris']
        for color in colors:
            if color in query_lower:
                criteria.append(f"couleur {color} ou similaire")
                break
        
        # Extract size/weight
        if 'léger' in query_lower or 'light' in query_lower:
            criteria.append("poids réduit")
        if 'compact' in query_lower:
            criteria.append("format compact")
        
        # Extract quality
        if any(word in query_lower for word in ['qualité', 'premium', 'haut de gamme']):
            criteria.append("note moyenne ≥ 4.5 étoiles")
            criteria.append("qualité premium")
        
        # Extract delivery
        if 'livraison' in query_lower:
            criteria.append("livraison gratuite ou rapide")
        
        # Extract origin
        if any(word in query_lower for word in ['france', 'europe', 'local', 'made in']):
            criteria.append("fabriqué en Europe ou localement")
        
        # Extract warranty
        if 'garantie' in query_lower:
            criteria.append("garantie minimum 2 ans")
        
        # Extract type/style
        if 'sac' in query_lower:
            criteria.append("sac bandoulière ou sac à dos")
        if 'résistant' in query_lower or 'waterproof' in query_lower:
            criteria.append("résistant à l'eau")
    
    # Add sustainability criteria if mode is enabled
    if sustainability_mode:
        criteria.extend([
            "produit durable et éco-responsable",
            "marque certifiée B-Corp ou Fair Trade",
            "matériaux recyclés ou biologiques",
            "emballage minimal ou recyclable"
        ])
    
    # Add criteria from history
    if last_5_products:
        # Analyze history to infer preferences
        criteria.append("style cohérent avec les achats précédents")
        criteria.append("qualité similaire aux produits précédemment achetés")
    
    # Ensure we have at least 8 criteria
    default_criteria = [
        "note moyenne ≥ 4.0 étoiles",
        "avis clients positifs",
        "produit disponible en stock",
        "retour gratuit possible"
    ]
    
    # Combine and deduplicate
    all_criteria = list(set(criteria + default_criteria))
    
    # Return 8-12 criteria
    return all_criteria[:12] if len(all_criteria) >= 8 else (all_criteria + default_criteria)[:12]


def create_rag_prompt(
    user_history_summary: str,
    candidate_products: List[Dict],
    query: Optional[str] = None,
    sustainability_mode: bool = False,
    search_criteria: Optional[List[str]] = None
) -> str:
    """
    Create RAG prompt for LLM
    
    Args:
        user_history_summary: Summary of user's purchase history
        candidate_products: List of dicts with product info (top 100 from retrieval)
        query: Optional natural language query
        sustainability_mode: Whether to prioritize sustainable products
    
    Returns:
        Formatted prompt string
    """
    # Format candidate products
    products_text = ""
    for i, product in enumerate(candidate_products[:100], 1):  # Top 100
        products_text += f"\n{i}. ASIN: {product['asin']}\n"
        products_text += f"   Title: {product['title']}\n"
        products_text += f"   Category: {product.get('category', 'N/A')}\n"
        products_text += f"   Description: {product.get('description', '')[:200]}\n"
        products_text += f"   Retrieval Score: {product.get('score', 0):.4f}\n"
    
    sustainability_note = ""
    if sustainability_mode:
        sustainability_note = """
IMPORTANT: Prioritize products that are sustainable, eco-friendly, organic, recycled, or vegan.
Look for keywords like: organic, recycled, sustainable, eco-friendly, vegan, biodegradable, fair trade.
"""
    
    criteria_section = ""
    if search_criteria:
        criteria_list = "\n".join([f"- {criterion}" for criterion in search_criteria])
        criteria_section = f"""
Precise Search Criteria (implicit user requirements):
{criteria_list}

IMPORTANT: Prioritize products that match these criteria when ranking.
"""
    
    prompt = f"""You are an expert product recommendation assistant for an e-commerce platform.

User's Purchase History:
{user_history_summary}

{f"User Query: {query}" if query else ""}

{criteria_section}

Candidate Products (retrieved from similarity search):
{products_text}

Your task:
1. Analyze the user's purchase history and preferences
2. From the candidate products above, select the TOP 10 most relevant products
3. Re-rank them in order of relevance (1 = most relevant)
4. For each product, provide a natural, convincing explanation (2-3 sentences) explaining why you recommend it

{sustainability_note}

Output your recommendations in the following JSON format:
{{
  "recommendations": [
    {{
      "rank": 1,
      "asin": "B00XXXXX",
      "explanation": "Why this product is perfect for the user..."
    }},
    ...
  ]
}}

Be specific and reference the user's past purchases when relevant. Make the explanations engaging and personalized.
"""
    
    return prompt


def parse_llm_response(response_text: str) -> List[Dict]:
    """Parse LLM JSON response"""
    try:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        data = json.loads(response_text)
        return data.get('recommendations', [])
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response text: {response_text[:500]}")
        return []

