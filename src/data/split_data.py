import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

input_path = Path("data/processed/cleaned_movie_reviews.csv")
train_path = Path("data/processed/train.csv")
test_path = Path("data/processed/test.csv")

df = pd.read_csv(input_path)

train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Split into {len(train_df)} train and {len(test_df)} test samples.")