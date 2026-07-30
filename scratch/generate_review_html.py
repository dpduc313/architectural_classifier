import os
import random
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"
CLEANED_MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_cleaned.csv"
OUTPUT_HTML_PATH = PROJECT_ROOT / "review_patches.html"

def main():
    if not MANIFEST_PATH.exists() or not CLEANED_MANIFEST_PATH.exists():
        print("Error: Manifest files not found.")
        return

    # Load manifests
    print("Loading manifests...")
    df_all = pd.read_csv(MANIFEST_PATH)
    df_cleaned = pd.read_csv(CLEANED_MANIFEST_PATH)

    # Normalize path by extracting suffix from 'train/' or 'val/' or 'test/'
    def get_subpath(p):
        p_str = str(p).replace('\\', '/')
        for split in ['train/', 'val/', 'test/']:
            if split in p_str:
                return p_str.split(split, 1)[1]
        return p_str

    kept_subpaths = set(df_cleaned['processed_path'].apply(get_subpath).tolist())
    df_all['is_kept'] = df_all['processed_path'].apply(get_subpath).isin(kept_subpaths)
    
    kept_df = df_all[df_all['is_kept']].copy()
    # Map their processed_path to the cleaned directory so the HTML can load the images correctly
    kept_df['processed_path'] = kept_df['processed_path'].apply(lambda x: x.replace('processed_data', 'processed_data_cleaned'))
    
    filtered_df = df_all[~df_all['is_kept']].copy()
    
    print(f"Total: {len(df_all):,} | Kept: {len(kept_df):,} | Filtered: {len(filtered_df):,}")

    # Sample 150 kept and 150 filtered patches for visual verification
    random.seed(42)
    sample_size = min(150, len(kept_df), len(filtered_df))
    
    sampled_kept = kept_df.sample(n=sample_size, random_state=42).to_dict('records')
    sampled_filtered = filtered_df.sample(n=sample_size, random_state=42).to_dict('records')

    # Build items array for JS
    items = []
    for item in sampled_kept:
        items.append({
            "path": item['processed_path'],
            "style": item['style_label'],
            "split": item['split'],
            "building": item['building_id'],
            "status": "KEPT"
        })
    for item in sampled_filtered:
        items.append({
            "path": item['processed_path'],
            "style": item['style_label'],
            "split": item['split'],
            "building": item['building_id'],
            "status": "FILTERED"
        })

    # Shuffle them to make it interesting, or group them by status
    random.shuffle(items)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Patch Filtering Verification — Heritage Dataset</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }}
  header {{ background: #1a1d2e; padding: 14px 24px; border-bottom: 1px solid #2d3748;
            display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.2rem; font-weight: 600; }}
  .badge {{ background: #4f46e5; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; color: white; }}
  .badge-filtered {{ background: #ef4444; }}
  .badge-kept {{ background: #10b981; }}
  
  .filter-controls {{ display: flex; gap: 10px; margin-left: auto; }}
  button {{ cursor: pointer; border: none; border-radius: 6px; padding: 8px 16px;
            font-size: 0.85rem; font-weight: 600; transition: opacity .15s; background: #374151; color: white; }}
  button.active {{ background: #4f46e5; }}
  button:hover {{ opacity: 0.85; }}

  main {{ padding: 24px; max-width: 1600px; margin: 0 auto; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
  
  .card {{ background: #1a1d2e; border-radius: 10px; overflow: hidden; border: 2px solid #2d3748;
           transition: transform 0.2s, border-color 0.2s; }}
  .card:hover {{ transform: translateY(-2px); }}
  .card.kept {{ border-color: #10b981; }}
  .card.filtered {{ border-color: #ef4444; }}
  
  .img-container {{ position: relative; width: 100%; padding-top: 100%; background: #111827; }}
  .img-container img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; cursor: pointer; }}
  
  .meta {{ padding: 12px; font-size: 0.75rem; display: flex; flex-direction: column; gap: 4px; }}
  .meta-row {{ display: flex; justify-content: space-between; align-items: center; }}
  .label {{ font-weight: 600; color: #94a3b8; }}
  .val {{ font-weight: 600; }}
  
  .status-badge {{ padding: 2px 6px; border-radius: 4px; font-size: 0.65rem; font-weight: 800; color: white; }}
  .status-badge.kept {{ background: #10b981; }}
  .status-badge.filtered {{ background: #ef4444; }}

  #lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.95);
               z-index: 200; align-items: center; justify-content: center; flex-direction: column; }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 90vw; max-height: 85vh; border-radius: 8px; object-fit: contain; border: 3px solid #2d3748; }}
  #lightbox-caption {{ margin-top: 14px; font-size: 0.9rem; color: #cbd5e1; font-weight: 500; }}
  #lightbox-close {{ position: fixed; top: 20px; right: 28px; font-size: 2rem; cursor: pointer; color: white; }}
</style>
</head>
<body>
<header>
  <h1>🔍 Patch Sifting Verification</h1>
  <span class="badge badge-kept">Kept: {len(kept_df):,}</span>
  <span class="badge badge-filtered">Filtered: {len(filtered_df):,}</span>
  
  <div class="filter-controls">
    <button id="btn-all" class="active" onclick="setFilter('ALL')">All ({sample_size * 2})</button>
    <button id="btn-kept" onclick="setFilter('KEPT')">Kept ({sample_size})</button>
    <button id="btn-filtered" onclick="setFilter('FILTERED')">Filtered ({sample_size})</button>
  </div>
</header>

<main>
  <div id="grid-container" class="grid"></div>
</main>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">✕</span>
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-caption"></div>
</div>

<script>
const ITEMS = {items};

function render() {{
  const container = document.getElementById('grid-container');
  container.innerHTML = '';
  
  const activeFilter = currentFilter;
  
  ITEMS.forEach(item => {{
    if (activeFilter !== 'ALL' && item.status !== activeFilter) return;
    
    const card = document.createElement('div');
    card.className = `card ${{item.status.toLowerCase()}}`;
    
    card.innerHTML = `
      <div class="img-container">
        <img src="${{item.path}}" onclick="openLightbox('${{item.path}}', '${{item.status}}', '${{item.building}}', '${{item.style}}', '${{item.split}}')" loading="lazy">
      </div>
      <div class="meta">
        <div class="meta-row">
          <span class="val">${{item.building}}</span>
          <span class="status-badge ${{item.status.toLowerCase()}}">${{item.status}}</span>
        </div>
        <div class="meta-row" style="margin-top: 4px;">
          <span class="label">Style:</span>
          <span class="val" style="color: #cbd5e1;">${{item.style}}</span>
        </div>
        <div class="meta-row">
          <span class="label">Split:</span>
          <span class="val" style="color: #cbd5e1;">${{item.split}}</span>
        </div>
      </div>
    `;
    container.appendChild(card);
  }});
}}

let currentFilter = 'ALL';

function setFilter(filterType) {{
  currentFilter = filterType;
  document.querySelectorAll('.filter-controls button').forEach(btn => btn.classList.remove('active'));
  if (filterType === 'ALL') document.getElementById('btn-all').classList.add('active');
  if (filterType === 'KEPT') document.getElementById('btn-kept').classList.add('active');
  if (filterType === 'FILTERED') document.getElementById('btn-filtered').classList.add('active');
  render();
}}

function openLightbox(src, status, building, style, split) {{
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  const cap = document.getElementById('lightbox-caption');
  
  img.src = src;
  cap.innerText = `[${{status}}] Building: ${{building}} | Style: ${{style}} | Split: ${{split}} | Path: ${{src}}`;
  lb.classList.add('open');
}}

function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('open');
}}

// Close lightbox on escape key
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') closeLightbox();
}});

// Initial render
render();
</script>
</body>
</html>
"""

    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML verification page saved successfully to: {OUTPUT_HTML_PATH}")

if __name__ == "__main__":
    main()
