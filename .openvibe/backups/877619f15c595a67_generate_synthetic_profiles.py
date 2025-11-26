"""
Script to generate synthetic user profiles for benchmarking
"""
import json
from pathlib import Path
from src.utils import generate_synthetic_user_profiles
from src.recommender import LLMClient
import os

def main():
    """Generate and save synthetic user profiles"""
    print("=" * 60)
    print("Generating Synthetic User Profiles for Benchmarking")
    print("=" * 60)
    
    # Initialize LLM client if available
    llm_client = None
    try:
        # Try to get API key from environment
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            llm_client = LLMClient(provider="groq", model="llama-3.1-70b-versatile")
            print("✓ LLM client initialized (will use for realistic profile generation)")
        else:
            print("⚠ No GROQ_API_KEY found, using template-based generation")
    except Exception as e:
        print(f"⚠ Could not initialize LLM client: {e}, using template-based generation")
    
    # Generate profiles
    category = "Electronics"  # Can be changed
    num_profiles = 10
    
    print(f"\nGenerating {num_profiles} profiles for category: {category}")
    profiles = generate_synthetic_user_profiles(
        category=category,
        num_profiles=num_profiles,
        llm_client=llm_client
    )
    
    # Save to JSON file
    output_path = Path("data/synthetic_user_profiles.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Generated {len(profiles)} profiles")
    print(f"✓ Saved to {output_path}")
    
    # Display summary
    print("\n" + "=" * 60)
    print("Profile Summary:")
    print("=" * 60)
    for i, profile in enumerate(profiles, 1):
        print(f"\n{i}. {profile['user_id']}")
        print(f"   Persona: {profile['persona']}")
        print(f"   Purchases: {len(profile['past_purchases'])} items")
        print(f"   Queries: {len(profile['natural_queries'])} queries")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()

