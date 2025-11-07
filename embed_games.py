"""
embed_games.py
--------------
Generate and save embeddings for the unified Steam dataset
using SentenceTransformer.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer

# =====================================
# CONFIG
# =====================================
DATA_PATH = "steam_combined.csv"
SAVE_DIR = "recommender_model"
MODEL_NAME = "intfloat/e5-small-v2"

# =====================================
# LOAD DATASET
# =====================================
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.dropna(subset=["combined_text"], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"✅ Loaded {len(df):,} games for embedding.")

# =====================================
# BUILD EMBEDDINGS
# =====================================
print(f"⚙️ Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

texts = df["combined_text"].astype(str).tolist()

print("🧠 Generating embeddings (this may take a few minutes)...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# =====================================
# SAVE EMBEDDINGS
# =====================================
os.makedirs(SAVE_DIR, exist_ok=True)
np.save(os.path.join(SAVE_DIR, "embeddings.npy"), embeddings)
joblib.dump(df, os.path.join(SAVE_DIR, "dataframe.joblib"))

# Save model name for consistency
with open(os.path.join(SAVE_DIR, "model_name.txt"), "w") as f:
    f.write(MODEL_NAME)

print(f"💾 Saved embeddings and dataframe to '{SAVE_DIR}/'")
print(f"✅ Embeddings shape: {embeddings.shape}")
