"""
recommend_games.py
------------------
Semantic recommendation engine using precomputed embeddings.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# CONFIG
# =====================================
SAVE_DIR = "recommender_model"
TOP_K = 10

# =====================================
# LOAD SAVED DATA
# =====================================
print("📂 Loading embeddings and data...")

embeddings = np.load(os.path.join(SAVE_DIR, "embeddings.npy"))
df = joblib.load(os.path.join(SAVE_DIR, "dataframe.joblib"))

with open(os.path.join(SAVE_DIR, "model_name.txt")) as f:
    MODEL_NAME = f.read().strip()

print(f"✅ Loaded {len(df):,} games and model '{MODEL_NAME}'.")


# =====================================
# RECOMMENDATION FUNCTION
# =====================================
def recommend_games(query: str,
                    top_k: int = TOP_K,
                    price_max: float = None,
                    min_rating: float = None,
                    year_min: int = None,
                    platforms=None):
    """Return top_k semantically similar games to the query."""
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_vec, embeddings)[0]

    # Apply optional filters
    mask = np.ones(len(df), dtype=bool)

    if price_max is not None and "price" in df.columns:
        mask &= df["price"].fillna(0) <= price_max
    if min_rating is not None and "positive_ratings" in df.columns:
        mask &= df["positive_ratings"].fillna(0) >= min_rating
    if year_min is not None and "release_year" in df.columns:
        mask &= df["release_year"].fillna(0) >= year_min
    if platforms:
        platforms = [p.lower() for p in platforms]
        if "platforms" in df.columns:
            mask &= df["platforms"].fillna("").apply(lambda x: any(p in x.lower() for p in platforms))

    sims[~mask] = -1  # exclude filtered games
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append({
            "Name": row["name"],
            "Score": round(float(sims[idx]), 4),
            "Year": int(row.get("release_year", 0)) if not np.isnan(row.get("release_year", np.nan)) else None,
            "Price": row.get("price", None),
            "Genres": row.get("genres", ""),
            "Tags": row.get("tags", ""),
            "Summary": (row.get("short_description") or row.get("about_the_game") or "")[:200] + "..."
        })
    return results


# =====================================
# CLI INTERFACE
# =====================================
if __name__ == "__main__":
    print("\n🎮 Steam Game Recommender CLI — Semantic Mode\n")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🔍 Describe a game: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        price = input("💰 Max price (Enter to skip): ").strip()
        price = float(price) if price else None

        year = input("📅 Min release year (Enter to skip): ").strip()
        year = int(year) if year else None

        results = recommend_games(query, top_k=5, price_max=price, year_min=year)

        print(f"\n🎯 Top results for: '{query}'\n")
        for i, rec in enumerate(results, 1):
            print(f"{i}. {rec['Name']} ({rec['Year']}) | Score: {rec['Score']}")
            print(f"   Price: ${rec['Price']} | Genres: {rec['Genres']}")
            print(f"   Tags: {rec['Tags']}")
            print(f"   {rec['Summary']}\n")
