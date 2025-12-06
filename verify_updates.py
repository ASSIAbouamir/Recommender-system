
from src.utils import evaluate_sustainability_score

def test_scoring():
    # Test case 1: Standard product (should be low score without base 40)
    prod1 = {
        "title": "Standard Plastic Toy",
        "description": "Made of plastic, fun for kids.",
        "brand": "Generic",
        "features": "Durable"
    }
    score1 = evaluate_sustainability_score(**prod1)
    print(f"Product 1: {prod1['title']} -> Score: {score1['sustainability_score']} (Reason: {score1['short_reason']})")
    
    # Test case 2: Sustainable product
    prod2 = {
        "title": "Eco-friendly Bamboo Toothbrush",
        "description": "100% biodegradable handle, natural bristles.",
        "brand": "EcoSmile",
        "features": "Organic, Plastic-free"
    }
    score2 = evaluate_sustainability_score(**prod2)
    print(f"Product 2: {prod2['title']} -> Score: {score2['sustainability_score']} (Reason: {score2['short_reason']})")

    # Test case 3: Negative product
    prod3 = {
        "title": "Disposable Plastic Cutlery",
        "description": "Single-use virgin plastic forks.",
        "brand": "PartyTime",
        "features": "Disposable"
    }
    score3 = evaluate_sustainability_score(**prod3)
    print(f"Product 3: {prod3['title']} -> Score: {score3['sustainability_score']} (Reason: {score3['short_reason']})")

if __name__ == "__main__":
    test_scoring()
