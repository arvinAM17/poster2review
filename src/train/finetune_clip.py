from transformers import CLIPProcessor, CLIPModel, get_scheduler
from torch.utils.data import DataLoader
import torch
from torch import optim
from tqdm import tqdm
from pathlib import Path

from src.data.datasets import MoviePosterDataset, clip_collate_fn

import wandb

wandb.init(
    project="poster2review-clip",
    name="clip-finetune-96k",
    config={
        "epochs": 10,
        "batch_size": 32,
        "model": "CLIP",
        "optimizer": "AdamW",
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "dataset_size": 96000
    }
)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dataset = MoviePosterDataset(Path("data/processed/train.csv"), processor, max_samples=96000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=clip_collate_fn(processor))

print("Dataset and DataLoader initialized successfully.")


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)

model.train()
for epoch in range(10):
    pbar = tqdm(dataloader)
    total_loss = 0.0
    for batch in pbar:
        for k in batch:
            batch[k] = batch[k].to(device)
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        # Optional: log step-wise loss
        wandb.log({"train/loss": loss.item()})
        pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
    # Log epoch-wise loss
    wandb.log({"epoch/loss": total_loss / len(dataloader), "epoch": epoch})



# Save the fine-tuned model
model.save_pretrained(Path("models/clip-finetuned-poster2review-96k"))
processor.save_pretrained(Path("models/clip-finetuned-poster2review-96k"))

print("Fine-tuning complete. Model and processor saved.")