import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"
output_embeddings_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\embeddings.pt"
checkpoint_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\embeddings_checkpoint.pt"
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
            return pixel_values, idx
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return torch.zeros((3, 224, 224)), idx

def compute_embeddings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        torch.set_num_threads(4)
        print("Set PyTorch num_threads to 4 for optimized CPU inference.")

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file not found at {manifest_path}. Please run scan_and_manifest.py first.")
        return

    df = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df)} images.")

    print("Loading DINOv2 model and image processor...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small")
    model.to(device)
    model.eval()

    processed_embeddings = []
    start_idx = 0

    if os.path.exists(checkpoint_path):
        print("Found existing checkpoint. Loading...")
        checkpoint = torch.load(checkpoint_path)
        processed_embeddings = checkpoint['embeddings']
        start_idx = checkpoint['last_idx']
        print(f"Resuming from image index {start_idx} (already processed {start_idx} images)")

    if start_idx >= len(df):
        print("All images already processed.")
        embeddings = torch.cat(processed_embeddings, dim=0)
        torch.save(embeddings, output_embeddings_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return

    remaining_df = df.iloc[start_idx:].reset_index(drop=True)
    dataset = ImageDataset(remaining_df, project_root, processor)
    batch_size = 32 if device.type == "cuda" else 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    current_chunk = []
    chunk_size = 100
    
    print("Computing embeddings...")
    total_batches = (len(df) + batch_size - 1) // batch_size
    initial_batches = start_idx // batch_size

    with torch.no_grad():
        for batch_pixel_values, _ in tqdm(dataloader, initial=initial_batches, total=total_batches):
            batch_pixel_values = batch_pixel_values.to(device)
            outputs = model(pixel_values=batch_pixel_values)
            cls_embeddings = outputs.pooler_output
            current_chunk.append(cls_embeddings.cpu())
            
            images_in_chunk = sum(b.shape[0] for b in current_chunk)
            if images_in_chunk >= chunk_size:
                processed_embeddings.extend(current_chunk)
                start_idx += images_in_chunk
                
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({'embeddings': processed_embeddings, 'last_idx': start_idx}, checkpoint_path)
                current_chunk = []
        
        if current_chunk:
            processed_embeddings.extend(current_chunk)
            start_idx += sum(b.shape[0] for b in current_chunk)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({'embeddings': processed_embeddings, 'last_idx': start_idx}, checkpoint_path)

    embeddings = torch.cat(processed_embeddings, dim=0)
    print(f"Computed embeddings shape: {embeddings.shape}")
    
    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    torch.save(embeddings, output_embeddings_path)
    print(f"Saved final embeddings to: {output_embeddings_path}")
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed temporary checkpoint file.")

if __name__ == "__main__":
    compute_embeddings()
