import os
import random
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"
CLEANED_MANIFEST_PATH = PROJECT_ROOT / ".tmp" / "processed_manifest_cleaned.csv"

FILTERED_HTML_PATH = PROJECT_ROOT / "review_filtered_patches.html"
KEPT_HTML_PATH = PROJECT_ROOT / "review_kept_patches.html"

def get_subpath(p):
    p_str = str(p).replace('\\', '/')
    for split in ['train/', 'val/', 'test/']:
        if split in p_str:
            return p_str.split(split, 1)[1]
    return p_str

def main():
    if not MANIFEST_PATH.exists() or not CLEANED_MANIFEST_PATH.exists():
        print("Error: Manifest files not found.")
        return

    # Load manifests
    print("Loading manifests...")
    df_all = pd.read_csv(MANIFEST_PATH)
    df_cleaned = pd.read_csv(CLEANED_MANIFEST_PATH)

    kept_subpaths = set(df_cleaned['processed_path'].apply(get_subpath).tolist())
    df_all['is_kept'] = df_all['processed_path'].apply(get_subpath).isin(kept_subpaths)
    
    kept_df = df_all[df_all['is_kept']].copy()
    kept_df['img_url'] = kept_df['processed_path'].apply(lambda x: x.replace('processed_data', 'processed_data_cleaned'))
    
    filtered_df = df_all[~df_all['is_kept']].copy()
    filtered_df['img_url'] = filtered_df['processed_path']
    
    print(f"Total: {len(df_all):,} | Kept: {len(kept_df):,} | Filtered: {len(filtered_df):,}")

    # Load tracker file
    tracker_path = PROJECT_ROOT / ".tmp" / "reviewed_paths_tracker.json"
    tracker = {"kept": [], "filtered": []}
    if tracker_path.exists():
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                tracker = json.load(f)
            print(f"Loaded tracker: {len(tracker['kept']):,} kept and {len(tracker['filtered']):,} filtered patches already reviewed.")
        except Exception as e:
            print(f"Error loading tracker: {e}")
    else:
        # Pre-populate with the first batch (seed 42) to make sure we don't repeat them
        print("Initializing tracker with Batch 1 (seed 42)...")
        first_kept = kept_df.sample(n=min(1000, len(kept_df)), random_state=42)['processed_path'].tolist()
        first_filtered = filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42)['processed_path'].tolist()
        tracker["kept"] = first_kept
        tracker["filtered"] = first_filtered
        try:
            with open(tracker_path, "w", encoding="utf-8") as f:
                json.dump(tracker, f, indent=2)
        except Exception as e:
            print(f"Error initializing tracker: {e}")

    # Load or initialize current_under_review
    under_review_path = PROJECT_ROOT / ".tmp" / "current_under_review.json"
    under_review = {"kept": [], "filtered": []}
    if under_review_path.exists():
        try:
            with open(under_review_path, "r", encoding="utf-8") as f:
                under_review = json.load(f)
        except Exception as e:
            print(f"Error loading under_review: {e}")

    # Extract paths currently in under_review to avoid duplicating them
    under_review_kept_paths = [item['processed_path'] for item in under_review.get('kept', [])]
    under_review_filtered_paths = [item['processed_path'] for item in under_review.get('filtered', [])]

    # Filter out already reviewed paths
    kept_pool = kept_df[
        ~kept_df['processed_path'].isin(tracker['kept']) & 
        ~kept_df['processed_path'].isin(under_review_kept_paths)
    ]
    filtered_pool = filtered_df[
        ~filtered_df['processed_path'].isin(tracker['filtered']) & 
        ~filtered_df['processed_path'].isin(under_review_filtered_paths)
    ]

    print(f"Available pools for next batch — Kept: {len(kept_pool):,} | Filtered: {len(filtered_pool):,}")

    sample_size = 2000
    
    # Process Kept pool (to generate review_kept_patches.html)
    if under_review.get('kept'):
        print("Reusing existing Kept patches currently under review.")
        sampled_kept = under_review['kept']
    else:
        print("Sampling new Kept patches for review.")
        sampled_kept = kept_pool.sample(n=min(sample_size, len(kept_pool)), random_state=random.randint(1, 100000)).to_dict('records')
        under_review['kept'] = sampled_kept

    # Process Filtered pool (to generate review_filtered_patches.html)
    if under_review.get('filtered'):
        print("Reusing existing Filtered patches currently under review.")
        sampled_filtered = under_review['filtered']
    else:
        print("Sampling new Filtered patches for review.")
        sampled_filtered = filtered_pool.sample(n=min(sample_size, len(filtered_pool)), random_state=random.randint(1, 100000)).to_dict('records')
        under_review['filtered'] = sampled_filtered

    # Save under_review state
    try:
        with open(under_review_path, "w", encoding="utf-8") as f:
            json.dump(under_review, f, indent=2)
    except Exception as e:
        print(f"Error saving under_review: {e}")

    # Print cumulative progress stats
    total_reviewed = len(tracker['kept']) + len(tracker['filtered'])
    print(f"PROGRESS UPDATE: You have manually reviewed {len(tracker['kept']):,} kept and {len(tracker['filtered']):,} filtered patches. Total Reviewed: {total_reviewed:,} patches.")

    # Convert items to serializable list for JS
    kept_items = []
    for item in sampled_kept:
        kept_items.append({
            "path": item['processed_path'],
            "url": item['img_url'].replace('\\', '/'),
            "style": item['style_label'],
            "split": item['split'],
            "building": item['building_id']
        })

    filtered_items = []
    for item in sampled_filtered:
        filtered_items.append({
            "path": item['processed_path'],
            "url": item['img_url'].replace('\\', '/'),
            "style": item['style_label'],
            "split": item['split'],
            "building": item['building_id']
        })

    # Generate Filtered Patches Review HTML
    generate_html(
        filepath=FILTERED_HTML_PATH,
        title="Filtered Patches Review (Sifting Verification)",
        items_json=json.dumps(filtered_items),
        badge_text=f"Sample size: {len(filtered_items)} / Total filtered: {len(filtered_df):,}",
        mode="filtered"
    )

    # Generate Kept Patches Review HTML
    generate_html(
        filepath=KEPT_HTML_PATH,
        title="Kept Patches Review (Sifting Verification)",
        items_json=json.dumps(kept_items),
        badge_text=f"Sample size: {len(kept_items)} / Total kept: {len(kept_df):,}",
        mode="kept"
    )

    print("Both HTML verification files have been created successfully!")

def generate_html(filepath, title, items_json, badge_text, mode):
    action_text = "MOVE TO KEPT" if mode == "filtered" else "FILTER OUT"
    color_theme = "#10b981" if mode == "filtered" else "#ef4444"
    text_color = "white"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }}
  header {{ background: #1a1d2e; padding: 14px 24px; border-bottom: 1px solid #2d3748;
            display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.2rem; font-weight: 600; }}
  .badge {{ background: #4f46e5; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; color: white; }}
  
  button {{ cursor: pointer; border: none; border-radius: 6px; padding: 8px 16px;
            font-size: 0.85rem; font-weight: 600; transition: opacity .15s; }}
  button:hover {{ opacity: 0.85; }}
  .btn-export {{ background: {color_theme}; color: {text_color}; margin-left: auto; }}

  main {{ padding: 24px; max-width: 1600px; margin: 0 auto; }}
  .instruction-banner {{ background: #1e293b; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;
                         font-size: 0.85rem; color: #cbd5e1; border-left: 4px solid #4f46e5; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 16px; }}
  
  .card {{ background: #1a1d2e; border-radius: 10px; overflow: hidden; border: 2px solid #374151;
           transition: transform 0.2s, border-color 0.2s; cursor: pointer; position: relative; }}
  .card:hover {{ transform: translateY(-2px); }}
  
  .img-container {{ position: relative; width: 100%; padding-top: 100%; background: #111827; }}
  .img-container img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }}
  
  .meta {{ padding: 10px; font-size: 0.7rem; display: flex; flex-direction: column; gap: 4px; }}
  .meta-row {{ display: flex; justify-content: space-between; align-items: center; }}
  .label {{ font-weight: 600; color: #94a3b8; }}
  .val {{ font-weight: 600; }}
  
  /* Selected states */
  .card.selected {{ border-color: {color_theme}; box-shadow: 0 0 10px rgba({int(color_theme[1:3], 16)}, {int(color_theme[3:5], 16)}, {int(color_theme[5:7], 16)}, 0.25); }}
  
  .action-indicator {{ position: absolute; top: 8px; right: 8px; background: {color_theme}; color: {text_color};
                       font-size: 0.55rem; font-weight: 800; padding: 2px 6px; border-radius: 4px; display: none; }}
  .card.selected .action-indicator {{ display: block; }}

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
  <h1>{title}</h1>
  <span class="badge">{badge_text}</span>
  <button class="btn-export" onclick="exportCSV()">💾 Export Changes to CSV</button>
</header>

<main>
  <div class="instruction-banner">
    <strong>💡 Instructions:</strong> Click on any patch card to toggle its status. Selected patches will be marked for action (<strong>{action_text}</strong>). Click "Export Changes to CSV" when finished to download a CSV containing your modifications.
  </div>
  <div id="grid-container" class="grid"></div>
</main>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">✕</span>
  <img id="lightbox-img" src="" alt="">
  <div id="lightbox-caption"></div>
</div>

<script>
const ITEMS = {items_json};
const MODE = "{mode}";

// Track selected item indexes
const selectedSet = new Set();

function toggleItem(index) {{
  const card = document.getElementById(`card-${{index}}`);
  if (selectedSet.has(index)) {{
    selectedSet.delete(index);
    card.classList.remove('selected');
  }} else {{
    selectedSet.add(index);
    card.classList.add('selected');
  }}
}}

function render() {{
  const container = document.getElementById('grid-container');
  container.innerHTML = '';
  
  ITEMS.forEach((item, index) => {{
    const card = document.createElement('div');
    card.id = `card-${{index}}`;
    card.className = 'card';
    card.onclick = (e) => {{
      // Don't toggle if clicking on the overlay or something else
      toggleItem(index);
    }};
    
    card.innerHTML = `
      <div class="img-container">
        <img src="${{item.url}}" loading="lazy">
        <div class="action-indicator">{action_text}</div>
      </div>
      <div class="meta">
        <div class="meta-row">
          <span class="val">${{item.building}}</span>
          <span style="color: #4f46e5; font-weight: 800; font-size: 0.6rem; cursor: zoom-in;" onclick="e.stopPropagation(); openLightbox('${{item.url}}', '${{item.building}}', '${{item.style}}', '${{item.split}}')">🔍 ZOOM</span>
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

function exportCSV() {{
  if (selectedSet.size === 0) {{
    alert("No changes made yet. Click on patches to select them before exporting!");
    return;
  }}
  
  let csvRows = ["processed_path,new_action"];
  selectedSet.forEach(index => {{
    const item = ITEMS[index];
    const newAction = MODE === "filtered" ? "KEEP" : "FILTER";
    csvRows.push(`"${{item.path}}",${{newAction}}`);
  }});
  
  const csvContent = "data:text/csv;charset=utf-8," + csvRows.join("\\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", MODE === "filtered" ? "move_to_kept_patches.csv" : "filter_out_patches.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}}

function openLightbox(src, building, style, split) {{
  event.stopPropagation();
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  const cap = document.getElementById('lightbox-caption');
  
  img.src = src;
  cap.innerText = `Building: ${{building}} | Style: ${{style}} | Split: ${{split}} | Path: ${{src}}`;
  lb.classList.add('open');
}}

function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('open');
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    closeLightbox();
  }}
}});

// Initial render
render();
</script>
</body>
</html>
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
