import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Load model & processor
model = CLIPModel.from_pretrained("models/clip-finetuned-poster2review-96k")
processor = CLIPProcessor.from_pretrained("models/clip-finetuned-poster2review-96k")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# Load data (subset for evaluation)
df = pd.read_csv(Path("data/processed/test.csv"))
df = df.dropna(subset=["image_path", "text"]).sample(n=5000, random_state=42).reset_index(drop=True)

# Preload all images and texts
images = [Image.open(path).convert("RGB") for path in tqdm(df["image_path"])]
texts = df["text"].tolist()

model.to(device)

model.eval()

# Get embeddings
with torch.no_grad():
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    
    image_inputs = {"pixel_values": inputs["pixel_values"].to(device)}
    text_inputs = {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device)
    }

    image_embeds = model.get_image_features(**image_inputs)
    text_embeds = model.get_text_features(**text_inputs)

# Normalize
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Compute cosine similarity
similarity = image_embeds @ text_embeds.T  # (N, N)
top1 = (similarity.argmax(dim=-1) == torch.arange(len(df)).to(similarity.device)).float().mean()
top5 = torch.topk(similarity, 5, dim=-1).indices
top5_acc = (top5 == torch.arange(len(df)).unsqueeze(1).to(top5.device)).any(dim=1).float().mean()

print(f"Top-1 Accuracy: {top1.item():.4f}")
print(f"Top-5 Accuracy: {top5_acc.item():.4f}")