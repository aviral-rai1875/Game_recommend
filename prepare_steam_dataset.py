"""
prepare_steam_dataset.py
------------------------
Combine and clean Steam dataset files into one unified dataset
ready for recommender system training.

Usage:
    python prepare_steam_dataset.py
"""

import pandas as pd
import numpy as np

# ======================================
# CONFIGURATION
# ======================================
STEAM_PATH = "steam.csv"
DESC_PATH = "steam_description_data.csv"
REQ_PATH = "steam_requirements_data.csv"      # optional (ignored mostly)
TAG_PATH = "steamspy_tag_data.csv"
OUTPUT_PATH = "steam_combined.csv"


# ======================================
# 1. LOAD DATASETS
# ======================================
print("📥 Loading Steam datasets...")

steam_df = pd.read_csv(STEAM_PATH)
desc_df = pd.read_csv(DESC_PATH)
req_df = pd.read_csv(REQ_PATH)
tag_df = pd.read_csv(TAG_PATH)

print(f"✅ Loaded {len(steam_df)} base entries, {len(desc_df)} descriptions, {len(tag_df)} tag entries.")


# ======================================
# 2. BASIC CLEANING
# ======================================
def normalize_price(val):
    """Convert price to float safely."""
    try:
        return float(str(val).replace("$", "").strip())
    except:
        return np.nan


def normalize_date(date_str):
    """Extract release year from release_date."""
    try:
        return pd.to_datetime(date_str, errors="coerce").year
    except:
        return np.nan


steam_df["price"] = steam_df["price"].apply(normalize_price)
steam_df["release_year"] = steam_df["release_date"].apply(normalize_date)

# Clean text columns
for col in ["genres", "categories", "steamspy_tags", "developer", "publisher"]:
    if col in steam_df.columns:
        steam_df[col] = steam_df[col].fillna("").astype(str).str.replace(";", " ")

# Rename join keys
desc_df.rename(columns={"steam_appid": "appid"}, inplace=True)
req_df.rename(columns={"steam_appid": "appid"}, inplace=True)

# ======================================
# 3. PREPARE TAG DATA
# ======================================
print("🧩 Processing tag data...")

# Extract tag columns (ignore 'appid')
tag_columns = [col for col in tag_df.columns if col != "appid"]

def top_tags(row, n=10):
    """Convert numeric tag weights into readable text tags."""
    top = row[row > 0].sort_values(ascending=False).head(n)
    return " ".join(top.index.tolist())

tag_df["tags"] = tag_df[tag_columns].apply(top_tags, axis=1)
tag_df_small = tag_df[["appid", "tags"]]

print(f"✅ Converted {len(tag_columns)} numeric tags into text tags.")


# ======================================
# 4. MERGE ALL DATASETS
# ======================================
print("🔗 Merging datasets...")

merged_df = (
    steam_df
    .merge(desc_df[["appid", "about_the_game", "short_description"]], on="appid", how="left")
    .merge(tag_df_small, on="appid", how="left")
)

# ======================================
# 5. CREATE COMBINED TEXT
# ======================================
print("🧠 Building combined_text field...")

merged_df["combined_text"] = (
    merged_df["about_the_game"].fillna("") + " " +
    merged_df["short_description"].fillna("") + " " +
    merged_df["genres"].fillna("") + " " +
    merged_df["categories"].fillna("") + " " +
    merged_df["steamspy_tags"].fillna("") + " " +
    merged_df["tags"].fillna("") + " " +
    merged_df["developer"].fillna("") + " " +
    merged_df["publisher"].fillna("")
).str.replace(r"\s+", " ", regex=True).str.strip()

# ======================================
# 6. FINAL CLEANUP
# ======================================
print("🧹 Cleaning null and duplicate entries...")

merged_df.drop_duplicates(subset=["appid"], inplace=True)
merged_df.dropna(subset=["name", "combined_text"], inplace=True)

# Select only useful columns
columns_to_keep = [
    "appid", "name", "developer", "publisher", "release_year",
    "platforms", "price", "positive_ratings", "negative_ratings",
    "owners", "genres", "categories", "steamspy_tags", "tags",
    "about_the_game", "short_description", "combined_text"
]

final_df = merged_df[columns_to_keep]

# ======================================
# 7. SAVE CLEANED DATA
# ======================================
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"💾 Saved cleaned dataset to {OUTPUT_PATH}")
print(f"✅ Final dataset shape: {final_df.shape}")
