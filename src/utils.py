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
        # Color based on score (0-100 scale)
        if sustainability_score >= 80:
            color = "#28a745"  # Green
        elif sustainability_score >= 60:
            color = "#ffc107"  # Yellow
        else:
            color = "#dc3545"  # Red
        
        sustainability_html = f"""
        <div style="margin-top:8px;padding:8px;background:#f8f9fa;border-radius:4px;border-left:3px solid {color};">
            <p style="margin:0;font-size:12px;font-weight:bold;color:{color};">
                🌱 Durabilité: {sustainability_score}/100
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

Évalue ce produit de 0 à 100 en termes de durabilité réelle (et pas juste greenwashing).

Critères pris en compte :

- Matériaux recyclés, biosourcés, certifiés (GOTS, OEKO-TEX, etc.)

- Empreinte carbone de fabrication et transport

- Durabilité et réparabilité

- Certifications officielles

- Politique de la marque (B-Corp, 1% for the Planet, etc.)

- Avis clients mentionnant la longévité ou l'éthique

Réponds EXACTEMENT avec ce format (rien d'autre) :

{{"sustainability_score": 87, "short_reason": "Cuir de pomme recyclé, marque B-Corp, fabriqué en Italie, réparable à vie"}}

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
                # Convert from 0-10 to 0-100 scale (as per paper)
                if score <= 10:
                    score = score * 10
                # Ensure score is between 0 and 100
                score = max(0, min(100, score))
                return {
                    'sustainability_score': round(score, 1),
                    'short_reason': reason
                }
        except Exception as e:
            print(f"Error evaluating sustainability with LLM: {e}, falling back to rule-based")
    
    # Fallback to rule-based evaluation
    # Article formula: S(i) = 100 * (w_pos * I[pos] - w_neg * I[neg] + b_cert)
    
    full_text = f"{title} {description} {features} {brand}".lower()
    
    # Weights (calibrated as per article inspiration)
    w_pos = 0.15  # Weight for each positive attribute
    w_neg = 0.20  # Weight for each negative attribute
    b_cert = 0.10 # Bonus for certification
    
    # Positive keywords / attributes
    pos_count = 0
    positive_reasons = []
    
    # Positive: Recycled / Circular
    if any(word in full_text for word in ['recycled', 'recyclé', 'upcycled', 'récupéré', 'circular']):
        pos_count += 1
        positive_reasons.append("Recycled materials")
        
    # Positive: Organic / Bio
    if any(word in full_text for word in ['organic', 'bio', 'biologique', 'natural fibers', 'coton bio']):
        pos_count += 1
        positive_reasons.append("Organic materials")
        
    # Positive: Fair Trade / Ethical
    if any(word in full_text for word in ['fair trade', 'équitable', 'ethical', 'ethique', 'b-corp', 'b corp']):
         pos_count += 1
         positive_reasons.append("Ethical certification")
         
    # Positive: Local / Made in Europe
    if any(word in full_text for word in ['made in europe', 'fabriqué en europe', 'made in france', 'local']):
        pos_count += 1
        positive_reasons.append("Made in Europe/Local")

    # Positive: Carbon/Energy
    if any(word in full_text for word in ['carbon neutral', 'solar', 'low energy', 'basse consommation']):
        pos_count += 1
        positive_reasons.append("Energy efficient")
    
    # Negative keywords / attributes
    neg_count = 0
    negative_reasons = []
    
    # Negative: Virgin plastic / Synthetic
    if any(word in full_text for word in ['virgin plastic', 'plastique vierge', 'polyester', 'synthetic', 'acrylique']) and 'recycled' not in full_text:
        neg_count += 1
        negative_reasons.append("Virgin synthetic materials")
        
    # Negative: Fast Fashion indicators (generic)
    if any(word in full_text for word in ['fast fashion', 'trend', 'jetable', 'disposable']):
        neg_count += 1
        negative_reasons.append("Disposable/Fast fashion")
    
    # Certification bonus (b_cert)
    cert_score = 0
    if any(cert in full_text for cert in ['gots', 'oeko-tex', 'bluesign', 'cradle to cradle', 'epeat']):
        cert_score = b_cert
        positive_reasons.append("High-standard Certification")
    
    # Calculate raw score (Article Formula)
    # S(i) = 100 * (w_pos * pos_count - w_neg * neg_count + cert_score)
    raw_score = 100 * (w_pos * pos_count - w_neg * neg_count + cert_score)
    
    # Clamp to [0, 100]
    score = max(0, min(100, raw_score))
    
    # Generate short reason
    if positive_reasons:
        short_reason = ", ".join(positive_reasons[:3])
    elif negative_reasons:
        short_reason = "Concerns: " + ", ".join(negative_reasons[:2])
    else:
        short_reason = "Standard product"
    
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
    Create RAG prompt for LLM (as per paper format)
    
    Args:
        user_history_summary: Summary of user's purchase history
        candidate_products: List of dicts with product info (top 100 from retrieval)
        query: Optional natural language query
        sustainability_mode: Whether to prioritize sustainable products
    
    Returns:
        Formatted prompt string matching paper format
    """
    # Format candidate products as per paper: title, short description, sustainability score
    candidate_list = ""
    for i, product in enumerate(candidate_products[:100], 1):  # Top 100
        title = product.get('title', 'Unknown')
        desc = product.get('description', '')[:150]  # Short description
        sust_score = product.get('sustainability_score', 0)
        candidate_list += f"{i}. {title} | {desc} | Sustainability: {sust_score}/100\n"
    
    # Format user history (last 5 items as per paper)
    history_text = ""
    if user_history_summary and user_history_summary != "No purchase history available.":
        # Extract last 5 items from history summary
        lines = user_history_summary.split('\n')[:6]  # First line + up to 5 items
        history_text = "\n".join(lines)
    else:
        history_text = "No purchase history"
    
    sustainability_goal = ""
    if sustainability_mode:
        sustainability_goal = "Sustainability goal: prioritize eco-friendly products."
    
    # Paper format prompt
    prompt = f"""You are an ethical and expert shopping assistant.

User query: "{query if query else 'Recommend products based on my purchase history'}"

User history (last 5 items): {history_text}

{sustainability_goal}

Here are 100 candidate products with title, short description and sustainability score (0-100):

{candidate_list}

Return exactly the TOP-10 best matches, ranked by relevance + sustainability. For each:
- Product title
- One-sentence natural explanation in French/English
- Sustainability score

Format: JSON array, no extra text.

{{
  "recommendations": [
    {{
      "rank": 1,
      "title": "Product Title",
      "asin": "B00XXXXX",
      "explanation": "One-sentence explanation in French or English",
      "sustainability_score": 85
    }},
    ...
  ]
}}
"""
    
    return prompt


def parse_llm_response(response_text: str) -> List[Dict]:
    """
    Parse LLM JSON response (as per paper format)
    Expected format:
    {
      "recommendations": [
        {
          "rank": 1,
          "title": "Product Title",
          "asin": "B00XXXXX",
          "explanation": "One-sentence explanation",
          "sustainability_score": 85
        }
      ]
    }
    """
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


def generate_synthetic_user_profiles(
    category: str = "Electronics",
    num_profiles: int = 10,
    llm_client: Optional[object] = None,
    products_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic user profiles for benchmarking
    
    Args:
        category: Product category (e.g., "Electronics", "Clothing", "Home")
        num_profiles: Number of profiles to generate (default: 10)
        llm_client: Optional LLM client for advanced profile generation
        products_df: Optional DataFrame with products to sample realistic purchases
    
    Returns:
        List of user profile dicts with user_id, persona, past_purchases, natural_queries
    """
    # Create prompt for LLM generation
    prompt = f"""Tu es un générateur de données synthétiques ultra-réalistes pour benchmark de recommandation.

Crée {num_profiles} profils utilisateurs très différents qui achètent sur Amazon dans la catégorie {category}.

Pour chaque utilisateur, donne :
- Un petit persona (âge, genre, situation, valeurs principales)
- Ses 5 à 12 derniers achats (titres réels ou très plausibles du dataset)
- 3 requêtes en langage naturel qu'il pourrait taper (variées : une vague, une précise, une avec contrainte budget/durabilité)

Réponds UNIQUEMENT en JSON valide, rien d'autre :

[
  {{
    "user_id": "synthetic_001",
    "persona": "Femme 29 ans, graphiste freelance à Lisbonne, très sensible à la durabilité et au design scandinave",
    "past_purchases": ["Lampe LED en bois recyclé", "T-shirt en coton bio oversize", "Sac à dos Fjällräven Kånken recyclé"],
    "natural_queries": [
      "lampe de bureau design et écolo",
      "vêtements en matières recyclées pas trop chers",
      "sac à dos résistant à l'eau vegan sous 120€"
    ]
  }},
  ...
]
"""
    
    # Try to use LLM if available
    if llm_client and hasattr(llm_client, 'generate'):
        try:
            response = llm_client.generate(prompt, temperature=0.7, max_tokens=4000)
            # Parse JSON response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                profiles = json.loads(json_match.group(0))
                if profiles and len(profiles) >= num_profiles:
                    return profiles[:num_profiles]
        except Exception as e:
            print(f"Error generating profiles with LLM: {e}, falling back to template-based")
    
    # Fallback to template-based generation
    profiles = []
    
    # Template personas for different user types
    personas_templates = [
        {
            "persona": "Femme 29 ans, graphiste freelance à Lisbonne, très sensible à la durabilité et au design scandinave",
            "purchases": ["Lampe LED en bois recyclé", "T-shirt en coton bio oversize", "Sac à dos Fjällräven Kånken recyclé", "Coussin en lin naturel", "Stylo rechargeable en métal"],
            "queries": ["lampe de bureau design et écolo", "vêtements en matières recyclées pas trop chers", "sac à dos résistant à l'eau vegan sous 120€"]
        },
        {
            "persona": "Homme 42 ans, ingénieur à Paris, passionné de technologie et qualité premium",
            "purchases": ["Casque audio sans fil premium", "Montre connectée sport", "Chargeur rapide USB-C", "Écouteurs Bluetooth pro", "Powerbank haute capacité"],
            "queries": ["casque audio professionnel", "montre connectée avec GPS et résistance à l'eau", "chargeur rapide pour iPhone et Android"]
        },
        {
            "persona": "Femme 35 ans, mère de 2 enfants à Lyon, recherche praticité et durabilité",
            "purchases": ["Aspirateur robot autonome", "Machine à laver économe", "Coussin ergonomique", "Organisateur de rangement", "Brosse à dents électrique"],
            "queries": ["aspirateur robot pas cher", "produits ménagers écologiques", "objets durables pour la maison"]
        },
        {
            "persona": "Homme 28 ans, étudiant en design à Berlin, budget serré mais goût pour le design",
            "purchases": ["Écouteurs filaires bon marché", "Clavier mécanique budget", "Souris ergonomique", "Lampadaire design", "Câble USB-C"],
            "queries": ["écouteurs pas chers avec bonne qualité son", "clavier mécanique sous 50€", "accessoires tech design et abordables"]
        },
        {
            "persona": "Femme 50 ans, professeure à Marseille, préfère la qualité et le made in France",
            "purchases": ["Livre électronique", "Carnet en cuir français", "Stylo plume de qualité", "Lampe de lecture", "Marque-pages magnétiques"],
            "queries": ["produits français de qualité", "livre électronique avec rétro-éclairage", "accessoires de bureau durables"]
        },
        {
            "persona": "Homme 33 ans, entrepreneur à Barcelone, aime les produits innovants et écologiques",
            "purchases": ["Smartphone reconditionné", "Chargeur solaire portable", "Enceinte Bluetooth étanche", "Montre fitness", "Écouteurs sans fil"],
            "queries": ["smartphone reconditionné fiable", "produits tech écologiques", "accessoires nomades durables"]
        },
        {
            "persona": "Femme 26 ans, influenceuse lifestyle à Amsterdam, recherche esthétique et tendance",
            "purchases": ["Enceinte Bluetooth design", "Coque de téléphone personnalisée", "Lampadaire LED coloré", "Organisateur de bureau", "Miroir avec LED"],
            "queries": ["produits design pour Instagram", "déco tendance pas chère", "accessoires tech esthétiques"]
        },
        {
            "persona": "Homme 45 ans, consultant à Londres, privilégie efficacité et professionnalisme",
            "purchases": ["Ordinateur portable professionnel", "Souris ergonomique", "Casque avec réduction de bruit", "Chargeur universel", "Hub USB-C"],
            "queries": ["équipement de bureau professionnel", "casque avec ANC de qualité", "accessoires pour télétravail"]
        },
        {
            "persona": "Femme 31 ans, médecin à Zurich, recherche hygiène et praticité",
            "purchases": ["Stérilisateur UV", "Balance connectée", "Tensiomètre digital", "Lampe de bureau médicale", "Organisateur de médicaments"],
            "queries": ["produits d'hygiène innovants", "appareils médicaux fiables", "objets pratiques pour la santé"]
        },
        {
            "persona": "Homme 38 ans, musicien à Vienne, passionné de son et qualité audio",
            "purchases": ["Casque audio studio", "Interface audio USB", "Microphone professionnel", "Enceinte monitoring", "Câbles audio de qualité"],
            "queries": ["équipement audio professionnel", "casque studio avec bonne réponse fréquentielle", "microphone USB pour enregistrement"]
        }
    ]
    
    # Generate profiles from templates
    for i in range(min(num_profiles, len(personas_templates))):
        template = personas_templates[i]
        profiles.append({
            "user_id": f"synthetic_{i+1:03d}",
            "persona": template["persona"],
            "past_purchases": template["purchases"][:np.random.randint(5, 13)],
            "natural_queries": template["queries"]
        })
    
    # If we need more profiles, generate variations
    while len(profiles) < num_profiles:
        base_template = np.random.choice(personas_templates)
        profiles.append({
            "user_id": f"synthetic_{len(profiles)+1:03d}",
            "persona": base_template["persona"].replace("29 ans", f"{np.random.randint(20, 55)} ans"),
            "past_purchases": np.random.choice(base_template["purchases"], size=np.random.randint(5, 10), replace=False).tolist(),
            "natural_queries": base_template["queries"]
        })
    
    return profiles[:num_profiles]

