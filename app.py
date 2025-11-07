"""
app.py
------
Streamlit UI for the Steam Game Recommender (Semantic Embeddings)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================
SAVE_DIR = "recommender_model"
TOP_K = 10

st.set_page_config(
    page_title="🎮 Steam Game Recommender",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Steam Game Recommender")
st.caption("Find games you'll love")

# =====================================================
# LOAD DATA AND MODEL
# =====================================================
@st.cache_resource
def load_data_and_model():
    embeddings = np.load(os.path.join(SAVE_DIR, "embeddings.npy"))
    df = joblib.load(os.path.join(SAVE_DIR, "dataframe.joblib"))
    with open(os.path.join(SAVE_DIR, "model_name.txt")) as f:
        model_name = f.read().strip()
    model = SentenceTransformer(model_name)
    return embeddings, df, model

with st.spinner("Loading recommender data..."):
    embeddings, df, model = load_data_and_model()
st.success(f"Loaded {len(df):,} games.")


# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
def recommend_games(query, top_k=TOP_K, price_max=None, year_min=None, platforms=None):
    """Return top_k semantically similar games."""
    query_vec = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_vec, embeddings)[0]

    # Apply optional filters
    mask = np.ones(len(df), dtype=bool)
    if price_max is not None and "price" in df.columns:
        mask &= df["price"].fillna(0) <= price_max
    if year_min is not None and "release_year" in df.columns:
        mask &= df["release_year"].fillna(0) >= year_min
    if platforms:
        platforms = [p.lower() for p in platforms]
        if "platforms" in df.columns:
            mask &= df["platforms"].fillna("").apply(lambda x: any(p in x.lower() for p in platforms))

    sims[~mask] = -1
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
            "Summary": (row.get("short_description") or row.get("about_the_game") or "")[:250] + "..."
        })
    return results


# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🎚️ Filters")

price_max = st.sidebar.number_input("💰 Max Price ($)", min_value=0.0, step=1.0, value=30.0)
year_min = st.sidebar.number_input("📅 Minimum Release Year", min_value=1980, max_value=2030, step=1, value=2010)
platforms = st.sidebar.multiselect("🖥️ Platforms", ["Windows", "Mac", "Linux"])

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# =====================================================
# MAIN QUERY BOX
# =====================================================
query = st.text_input(
    "🔍 Describe the kind of game you're looking for:",
    placeholder="e.g. co-op zombie survival shooter under $20",
)

if st.button("Recommend Games") or query:
    if not query.strip():
        st.warning("Please enter a game description.")
    else:
        with st.spinner("Finding your games..."):
            results = recommend_games(
                query,
                top_k=top_k,
                price_max=price_max,
                year_min=year_min,
                platforms=platforms
            )

        if not results:
            st.error("No matching games found. Try broadening your filters.")
        else:
            st.markdown(f"### 🎯 Top {len(results)} Results for: `{query}`")

            for rec in results:
                with st.container():
                    st.subheader(f"{rec['Name']} ({rec['Year']}) — 💲{rec['Price']}")
                    st.caption(f"**Genres:** {rec['Genres']}  |  **Tags:** {rec['Tags']}")
                    st.write(rec["Summary"])
                    st.progress(rec["Score"])
                    st.markdown("---")
