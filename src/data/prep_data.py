import pandas as pd
from pathlib import Path
import ast

# ---------- Config ----------
RAW_CSV = Path("data/raw/movie_data.csv")
POSTER_DIR = Path("data/posters")
OUTPUT_CSV = Path("data/processed/cleaned_movie_reviews.csv")

# ---------- Load & Filter ----------
print("Loading raw metadata CSV...")
df = pd.read_csv(RAW_CSV, engine="python", on_bad_lines='skip')

# Match downloaded images
available_images = {img.name for img in POSTER_DIR.glob("*.jpg")}
def image_name_from_row(row):
    return f"{row['tmdb_id']}.jpg"

df["image_filename"] = df.apply(image_name_from_row, axis=1)
df["image_path"] = df["image_filename"].apply(lambda x: str(POSTER_DIR / x))
df = df[df["image_filename"].isin(available_images)]

# Keep only relevant columns and remove rows without text
df = df[["image_path", "overview", "movie_id", "genres"]]
df = df.rename(columns={"overview": "text", "movie_id": "title"})
df = df.dropna(subset=["text"])

# ---------- Parse Genres ----------
def parse_genres(genre_str):
    try:
        parsed = ast.literal_eval(genre_str)
        return [g.strip() for g in parsed if isinstance(g, str)]
    except Exception:
        return []

df["genres"] = df["genres"].apply(parse_genres)

# ---------- Shuffle and Save ----------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved cleaned dataset with {len(df)} entries to {OUTPUT_CSV}")

