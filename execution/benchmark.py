import os
import time
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"
project_root = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final"

class ImageDataset(Dataset):
    def __init__(self, df, root_dir, processor):
        self.df = df
        self.root_dir = root_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = os.path.join(self.root_dir, row['file_path'])
        try:
            image = Image.open(full_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values
        except Exception:
            return torch.zeros((3, 224, 224))

def benchmark_settings():
    df = pd.read_csv(manifest_path).head(20)  # Benchmark on 20 images
    
    print("Loading model...", flush=True)
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()

    # Configurations to test (batch_size, num_threads)
    configs = [
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 2),
        (8, 4),
        (8, 8),
    ]

    for bs, threads in configs:
        torch.set_num_threads(threads)
        dataset = ImageDataset(df, project_root, processor)
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
        
        start_time = time.time()
        with torch.no_grad():
            for batch in dataloader:
                _ = model(pixel_values=batch)
        elapsed = time.time() - start_time
        print(f"Config: BatchSize={bs}, Threads={threads} | Time for 20 images: {elapsed:.2f}s ({elapsed/20:.3f}s/img)", flush=True)

if __name__ == "__main__":
    benchmark_settings()
