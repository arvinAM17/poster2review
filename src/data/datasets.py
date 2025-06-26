from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class MoviePosterDataset(Dataset):
    def __init__(self, csv_path, processor, max_samples=None):
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df[:max_samples]
        self.processor = processor

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        text = row["text"]
        
        return {"image": image, "text": text}
    
def clip_collate_fn(processor):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        return processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    return collate_fn