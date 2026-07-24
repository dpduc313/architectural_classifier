import os
import torch
import pandas as pd
import numpy as np

manifest_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\manifest.csv"
embeddings_path = r"c:\Users\teflo\Desktop\Study\VLU\Comp vision\BT\Final\.tmp\embeddings.pt"
similarity_threshold = 0.97  # Threshold for near-duplicates in DINOv2 space

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, i):
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        for node in path:
            self.parent[node] = i
        return i
        
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def detect_duplicates():
    if not os.path.exists(manifest_path) or not os.path.exists(embeddings_path):
        print("Error: manifest.csv or embeddings.pt not found. Run previous steps first.")
        return

    df = pd.read_csv(manifest_path)
    embeddings = torch.load(embeddings_path)
    n_images = len(df)
    
    print(f"Loaded {n_images} images and embeddings.")

    # Normalize embeddings for cosine similarity
    norm_embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    uf = UnionFind(n_images)
    
    batch_size = 500
    print("Computing cosine similarities and clustering duplicates...")
    
    for i in range(0, n_images, batch_size):
        end_i = min(i + batch_size, n_images)
        batch_embeddings = norm_embeddings[i:end_i]
        sim = torch.mm(batch_embeddings, norm_embeddings.t())
        matches = (sim > similarity_threshold).nonzero()
        
        for match in matches:
            idx_in_batch = match[0].item()
            global_i = i + idx_in_batch
            global_j = match[1].item()
            
            if global_i < global_j:
                uf.union(global_i, global_j)
                
    root_to_nodes = {}
    for idx in range(n_images):
        root = uf.find(idx)
        if root not in root_to_nodes:
            root_to_nodes[root] = []
        root_to_nodes[root].append(idx)
        
    dup_cluster_ids = [-1] * n_images
    cluster_counter = 0
    
    for root, nodes in root_to_nodes.items():
        if len(nodes) > 1:
            for node in nodes:
                dup_cluster_ids[node] = cluster_counter
            cluster_counter += 1
            
    df['dup_cluster_id'] = dup_cluster_ids
    df.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    
    n_duplicates = sum(1 for cid in dup_cluster_ids if cid != -1)
    print(f"Duplicate detection completed.")
    print(f"  Found {cluster_counter} duplicate clusters.")
    print(f"  Total duplicate images: {n_duplicates} ({n_duplicates / n_images * 100:.2f}%)")
    print(f"  Saved duplicate labels back to: {manifest_path}")

if __name__ == "__main__":
    detect_duplicates()
