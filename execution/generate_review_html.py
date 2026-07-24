"""
generate_review_html.py — Standalone, 100% Offline-Compatible Dataset Review Generator.
Generates:
  - review_outliers.html   (browse outlier images, mark KEEP/REMOVE)
  - review_duplicates.html (browse duplicate clusters, set cluster decisions)

Uses relative image paths so files open instantly off disk via file:// protocol
without needing any background HTTP server. All user decisions auto-save to browser LocalStorage.
"""

import pandas as pd
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


import urllib.parse


def to_relative_url(relative_path):
    clean = relative_path.replace("\\", "/").lstrip("/")
    return urllib.parse.quote(clean, safe="/()")


# ══════════════════════════════════════════════════════════════════════════════
# 1. OUTLIERS REVIEW TOOL
# ══════════════════════════════════════════════════════════════════════════════
def generate_outliers_html():
    csv_path = os.path.join(PROJECT_ROOT, ".tmp", "outliers_review.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    items = []
    for idx, row in df.iterrows():
        items.append({
            "id":        int(idx),
            "building":  str(row["building_id"]),
            "style":     str(row["style_label"]),
            "filename":  str(row["filename"]),
            "file_path": str(row["file_path"]),
            "url":       to_relative_url(row["file_path"]),
            "score":     round(float(row["outlier_score"]), 4),
            "action":    str(row["action"]),
        })

    data_json = json.dumps(items, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Outlier Review — Heritage Dataset</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; }}
  header {{ background: #1a1d2e; padding: 14px 24px; border-bottom: 1px solid #2d3748;
            display: flex; align-items: center; gap: 16px; position: sticky; top: 0; z-index: 100; }}
  header h1 {{ font-size: 1.2rem; font-weight: 600; }}
  .badge {{ background: #4f46e5; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; color: white; }}
  .badge.green {{ background: #16a34a; }}
  .badge.red {{ background: #dc2626; }}
  .controls {{ margin-left: auto; display: flex; gap: 8px; }}
  button {{ cursor: pointer; border: none; border-radius: 6px; padding: 8px 16px;
            font-size: 0.85rem; font-weight: 600; transition: all .15s ease; }}
  button:hover {{ opacity: 0.85; transform: translateY(-1px); }}
  .btn-all-keep {{ background: #16a34a; color: white; }}
  .btn-all-remove {{ background: #dc2626; color: white; }}
  .btn-export {{ background: #4f46e5; color: white; }}

  .sidebar {{ width: 240px; background: #1a1d2e; border-right: 1px solid #2d3748;
              position: fixed; top: 57px; bottom: 0; overflow-y: auto; padding: 12px; }}
  .sidebar h3 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: .08em;
                 color: #94a3b8; margin-bottom: 8px; }}
  .sidebar-item {{ padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
                   margin-bottom: 3px; display: flex; justify-content: space-between; align-items: center; color: #cbd5e1; }}
  .sidebar-item:hover {{ background: #2d3748; }}
  .sidebar-item.active {{ background: #4f46e5; color: white; font-weight: 600; }}
  .count {{ background: #374151; padding: 2px 8px; border-radius: 9999px; font-size: 0.7rem; }}

  main {{ margin-left: 240px; padding: 24px; }}
  .building-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }}
  .building-header h2 {{ font-size: 1.1rem; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 16px; margin-bottom: 40px; }}
  .card {{ background: #1a1d2e; border-radius: 10px; overflow: hidden;
           border: 2px solid #374151; transition: border-color .2s, box-shadow .2s; }}
  .card.KEEP   {{ border-color: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, 0.2); }}
  .card.REMOVE {{ border-color: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }}
  .card img {{ width: 100%; height: 160px; object-fit: cover; display: block; cursor: pointer; background: #111827; }}
  .card-body {{ padding: 10px; }}
  .card-name {{ font-size: 0.72rem; color: #94a3b8; white-space: nowrap;
                overflow: hidden; text-overflow: ellipsis; margin-bottom: 4px; }}
  .score {{ font-size: 0.75rem; color: #f59e0b; font-weight: 600; margin-bottom: 8px; }}
  .card-btns {{ display: flex; gap: 6px; }}
  .btn-keep {{ flex: 1; background: #1f2937; color: #9ca3af; font-size: 0.75rem; padding: 6px; border-radius: 6px; border: 1px solid #374151; }}
  .btn-remove {{ flex: 1; background: #1f2937; color: #9ca3af; font-size: 0.75rem; padding: 6px; border-radius: 6px; border: 1px solid #374151; }}
  .card.KEEP .btn-keep {{ background: #16a34a; color: white; border-color: #22c55e; font-weight: 700; }}
  .card.REMOVE .btn-remove {{ background: #dc2626; color: white; border-color: #ef4444; font-weight: 700; }}

  #lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.9);
               z-index: 200; align-items: center; justify-content: center; padding: 20px; }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 8px; object-fit: contain; }}
  #lightbox-close {{ position: fixed; top: 20px; right: 28px; font-size: 2rem;
                     cursor: pointer; color: white; line-height: 1; }}

  .section {{ display: none; }}
  .section.active {{ display: block; }}
  .stats-bar {{ display: flex; gap: 12px; margin-bottom: 20px; font-size: 0.85rem; }}
  .stat {{ background: #1a1d2e; padding: 8px 14px; border-radius: 8px; border: 1px solid #2d3748; }}
  .stat span {{ font-weight: 700; color: #a78bfa; }}
  #toast {{ position: fixed; bottom: 20px; right: 20px; background: #16a34a; color: white; padding: 10px 18px; border-radius: 8px; font-weight: 600; display: none; z-index: 300; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
</style>
</head>
<body>

<header>
  <h1>🔍 Outlier Review</h1>
  <span class="badge" id="total-badge">487 images</span>
  <span class="badge green" id="keep-badge">0 KEEP</span>
  <span class="badge red"   id="remove-badge">487 REMOVE</span>
  <div class="controls">
    <button class="btn-all-keep"   onclick="markVisible('KEEP')">✓ Keep All Visible</button>
    <button class="btn-all-remove" onclick="markVisible('REMOVE')">✗ Remove All Visible</button>
    <button class="btn-export"     onclick="exportCSV()">💾 Save / Export CSV</button>
  </div>
</header>

<nav class="sidebar">
  <h3>Buildings</h3>
  <div id="sidebar-list"></div>
</nav>

<main>
  <div class="stats-bar">
    <div class="stat">Total Flagged: <span id="s-total">487</span></div>
    <div class="stat" style="color:#22c55e">Keep (Restore): <span id="s-keep">0</span></div>
    <div class="stat" style="color:#ef4444">Remove (Drop): <span id="s-remove">487</span></div>
  </div>
  <div id="sections"></div>
</main>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">✕</span>
  <img id="lightbox-img" src="" alt="">
</div>

<div id="toast">Saved!</div>

<script>
const ITEMS = {data_json};
let decisions = {{}};

// Load decisions from LocalStorage if available
const savedLS = localStorage.getItem('outlier_decisions_v2');
if (savedLS) {{
  try {{ decisions = JSON.parse(savedLS); }} catch(e) {{ decisions = {{}}; }}
}}

ITEMS.forEach(item => {{
  if (!decisions[item.id]) {{
    decisions[item.id] = item.action;
  }}
}});

function saveLS() {{
  localStorage.setItem('outlier_decisions_v2', JSON.stringify(decisions));
}}

function buildUI() {{
  const sidebar = document.getElementById('sidebar-list');
  const sections = document.getElementById('sections');
  sidebar.innerHTML = '';
  sections.innerHTML = '';

  const groups = {{}};
  ITEMS.forEach(item => {{
    if (!groups[item.building]) groups[item.building] = [];
    groups[item.building].push(item);
  }});

  const buildings = Object.keys(groups).sort();
  document.getElementById('s-total').textContent = ITEMS.length;
  document.getElementById('total-badge').textContent = ITEMS.length + ' images';

  buildings.forEach((bid, i) => {{
    const imgs = groups[bid];
    const navItem = document.createElement('div');
    navItem.className = 'sidebar-item' + (i === 0 ? ' active' : '');
    navItem.id = 'nav-' + bid;
    navItem.innerHTML = `${{bid}} <span class="count">${{imgs.length}}</span>`;
    navItem.onclick = () => showSection(bid);
    sidebar.appendChild(navItem);

    const sec = document.createElement('div');
    sec.className = 'section' + (i === 0 ? ' active' : '');
    sec.id = 'sec-' + bid;

    const style = imgs[0].style;
    sec.innerHTML = `
      <div class="building-header">
        <h2>${{bid}}</h2>
        <span class="badge">${{style}}</span>
        <span style="color:#94a3b8;font-size:.85rem">${{imgs.length}} flagged outlier images</span>
      </div>
      <div class="grid" id="grid-${{bid}}"></div>
    `;
    sections.appendChild(sec);

    const grid = sec.querySelector('.grid');
    imgs.forEach(img => {{
      const card = document.createElement('div');
      card.className = 'card ' + (decisions[img.id] || 'REMOVE');
      card.id = 'card-' + img.id;
      card.innerHTML = `
        <img src="${{img.url}}" alt="${{img.filename}}" loading="lazy"
             onclick="openLightbox('${{img.url}}')">
        <div class="card-body">
          <div class="card-name" title="${{img.filename}}">${{img.filename}}</div>
          <div class="score">⚠ outlier score: ${{img.score}}</div>
          <div class="card-btns">
            <button class="btn-keep"   onclick="mark(${{img.id}}, 'KEEP')">✓ Keep</button>
            <button class="btn-remove" onclick="mark(${{img.id}}, 'REMOVE')">✗ Remove</button>
          </div>
        </div>
      `;
      grid.appendChild(card);
    }});
  }});

  updateStats();
}}

function showSection(bid) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.sidebar-item').forEach(s => s.classList.remove('active'));
  const targetSec = document.getElementById('sec-' + bid);
  const targetNav = document.getElementById('nav-' + bid);
  if (targetSec) targetSec.classList.add('active');
  if (targetNav) targetNav.classList.add('active');
}}

function mark(id, action) {{
  decisions[id] = action;
  saveLS();
  const card = document.getElementById('card-' + id);
  if (card) {{
    card.classList.remove('KEEP', 'REMOVE');
    card.classList.add(action);
  }}
  updateStats();
}}

function markVisible(action) {{
  const activeSec = document.querySelector('.section.active');
  if (!activeSec) return;
  activeSec.querySelectorAll('.card').forEach(card => {{
    const id = parseInt(card.id.replace('card-', ''));
    mark(id, action);
  }});
}}

function updateStats() {{
  const vals = Object.values(decisions);
  const keep = vals.filter(v => v === 'KEEP').length;
  const remove = vals.filter(v => v === 'REMOVE').length;
  document.getElementById('s-keep').textContent = keep;
  document.getElementById('s-remove').textContent = remove;
  document.getElementById('keep-badge').textContent = keep + ' KEEP';
  document.getElementById('remove-badge').textContent = remove + ' REMOVE';
}}

function exportCSV() {{
  saveLS();
  const headers = ["building_id","style_label","filename","file_path","outlier_score","action"];
  const rows = ITEMS.map(item => [
    `"${{item.building}}"`,
    `"${{item.style}}"`,
    `"${{item.filename}}"`,
    `"${{item.file_path.replace(/"/g, '""')}}"`,
    item.score,
    `"${{decisions[item.id] || item.action}}"`
  ].join(','));
  
  const csvText = [headers.join(','), ...rows].join('\\n');
  const blob = new Blob([csvText], {{ type: 'text/csv;charset=utf-8;' }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "outliers_review.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("💾 Exported outliers_review.csv!");
}}

function showToast(msg) {{
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => {{ toast.style.display = 'none'; }}, 3000);
}}

function openLightbox(url) {{
  document.getElementById('lightbox-img').src = url;
  document.getElementById('lightbox').classList.add('open');
}}

function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('open');
}}

document.getElementById('lightbox').addEventListener('click', e => {{
  if (e.target === e.currentTarget) closeLightbox();
}});

buildUI();
</script>
</body>
</html>"""

    out_path = os.path.join(PROJECT_ROOT, "review_outliers.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# 2. DUPLICATES REVIEW TOOL
# ══════════════════════════════════════════════════════════════════════════════
def generate_duplicates_html():
    csv_path = os.path.join(PROJECT_ROOT, ".tmp", "duplicates_review.csv")
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)

    clusters = []
    for cid, group in df.groupby("dup_cluster_id"):
        items = []
        for idx, row in group.iterrows():
            items.append({
                "filename":  str(row["filename"]),
                "file_path": str(row["file_path"]),
                "url":       to_relative_url(str(row["file_path"])),
                "building":  str(row["building_id"]),
                "style":     str(row["style_label"]),
            })
        action = str(group["action"].iloc[0])
        clusters.append({
            "cluster_id": int(cid),
            "action":     action,
            "items":      items,
            "count":      len(items),
            "building":   items[0]["building"],
            "style":      items[0]["style"]
        })

    data_json = json.dumps(clusters, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Duplicate Review — Heritage Dataset</title>
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
  .btn-export {{ background: #4f46e5; color: white; margin-left: auto; }}

  main {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
  .cluster {{ background: #1a1d2e; border-radius: 10px; padding: 16px;
              margin-bottom: 20px; border: 2px solid #2d3748; transition: border-color .2s; }}
  .cluster-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
  .cluster-header h3 {{ font-size: 0.95rem; font-weight: 600; }}
  .policy {{ font-size: 0.75rem; color: #94a3b8; }}
  .cluster-grid {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .img-card {{ width: 180px; border-radius: 8px; overflow: hidden; border: 2px solid #374151; background: #111827; }}
  .img-card img {{ width: 100%; height: 135px; object-fit: cover; cursor: pointer; display: block; }}
  .img-meta {{ padding: 6px 8px; font-size: 0.68rem; color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .cluster-actions {{ margin-top: 10px; display: flex; gap: 8px; }}
  .btn-keepall   {{ background: #16a34a; color: white; font-size: 0.75rem; padding: 5px 12px; border-radius: 6px; }}
  .btn-removeall {{ background: #dc2626; color: white; font-size: 0.75rem; padding: 5px 12px; border-radius: 6px; }}
  .btn-default   {{ background: #374151; color: white; font-size: 0.75rem; padding: 5px 12px; border-radius: 6px; }}
  .cluster[data-action="KEEP_ALL"]    {{ border-color: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, 0.2); }}
  .cluster[data-action="REMOVE_ALL"]  {{ border-color: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }}

  #lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.9);
               z-index: 200; align-items: center; justify-content: center; }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 90vw; max-height: 90vh; border-radius: 8px; object-fit: contain; }}
  #lightbox-close {{ position: fixed; top: 20px; right: 28px; font-size: 2rem; cursor: pointer; color: white; }}
  #toast {{ position: fixed; bottom: 20px; right: 20px; background: #16a34a; color: white; padding: 10px 18px; border-radius: 8px; font-weight: 600; display: none; z-index: 300; }}
</style>
</head>
<body>
<header>
  <h1>🔁 Duplicate Review</h1>
  <span class="badge">{len(clusters)} clusters · {sum(c['count'] for c in clusters)} images</span>
  <span style="font-size:.8rem;color:#94a3b8">Default policy: keep 2 per cluster in train, 1 in val/test</span>
  <button class="btn-export" onclick="exportCSV()">💾 Save / Export CSV</button>
</header>

<main>
  <div id="clusters-container"></div>
</main>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">✕</span>
  <img id="lightbox-img" src="" alt="">
</div>

<div id="toast">Saved!</div>

<script>
const CLUSTERS = {data_json};
let clusterDecisions = {{}};

const savedLS = localStorage.getItem('duplicate_decisions_v2');
if (savedLS) {{
  try {{ clusterDecisions = JSON.parse(savedLS); }} catch(e) {{ clusterDecisions = {{}}; }}
}}

CLUSTERS.forEach(c => {{
  if (!clusterDecisions[c.cluster_id]) {{
    clusterDecisions[c.cluster_id] = c.action;
  }}
}});

function saveLS() {{
  localStorage.setItem('duplicate_decisions_v2', JSON.stringify(clusterDecisions));
}}

function render() {{
  const container = document.getElementById('clusters-container');
  container.innerHTML = '';

  CLUSTERS.forEach(c => {{
    const action = clusterDecisions[c.cluster_id] || c.action;
    const div = document.createElement('div');
    div.className = 'cluster';
    div.setAttribute('data-action', action);
    div.id = 'cluster-' + c.cluster_id;

    let imgsHtml = c.items.map(item => `
      <div class="img-card">
        <img src="${{item.url}}" alt="${{item.filename}}" loading="lazy" onclick="openLightbox('${{item.url}}')">
        <div class="img-meta" title="${{item.filename}}">${{item.filename}}</div>
      </div>
    `).join('');

    div.innerHTML = `
      <div class="cluster-header">
        <h3>Cluster #${{c.cluster_id}} — ${{c.building}} (${{c.style}})</h3>
        <span class="badge">${{c.count}} duplicate images</span>
        <span class="policy">Action: <strong>${{action}}</strong></span>
      </div>
      <div class="cluster-grid">${{imgsHtml}}</div>
      <div class="cluster-actions">
        <button class="btn-default"   onclick="setAction(${{c.cluster_id}}, 'DEFAULT')">↺ Auto Policy</button>
        <button class="btn-keepall"   onclick="setAction(${{c.cluster_id}}, 'KEEP_ALL')">✓ Keep All</button>
        <button class="btn-removeall" onclick="setAction(${{c.cluster_id}}, 'REMOVE_ALL')">✗ Remove All</button>
      </div>
    `;
    container.appendChild(div);
  }});
}}

function setAction(cid, act) {{
  clusterDecisions[cid] = act;
  saveLS();
  const elem = document.getElementById('cluster-' + cid);
  if (elem) {{
    elem.setAttribute('data-action', act);
    elem.querySelector('.policy strong').textContent = act;
  }}
}}

function exportCSV() {{
  saveLS();
  const rows = [];
  rows.push(["dup_cluster_id","building_id","style_label","filename","file_path","action"].join(','));

  CLUSTERS.forEach(c => {{
    const act = clusterDecisions[c.cluster_id] || c.action;
    c.items.forEach(item => {{
      rows.push([
        c.cluster_id,
        `"${{c.building}}"`,
        `"${{c.style}}"`,
        `"${{item.filename}}"`,
        `"${{item.file_path.replace(/"/g, '""')}}"`,
        `"${{act}}"`
      ].join(','));
    }});
  }});

  const csvText = rows.join('\\n');
  const blob = new Blob([csvText], {{ type: 'text/csv;charset=utf-8;' }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "duplicates_review.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("💾 Exported duplicates_review.csv!");
}}

function showToast(msg) {{
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => {{ toast.style.display = 'none'; }}, 3000);
}}

function openLightbox(url) {{
  document.getElementById('lightbox-img').src = url;
  document.getElementById('lightbox').classList.add('open');
}}

function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('open');
}}

document.getElementById('lightbox').addEventListener('click', e => {{
  if (e.target === e.currentTarget) closeLightbox();
}});

render();
</script>
</body>
</html>"""

    out_path = os.path.join(PROJECT_ROOT, "review_duplicates.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written: {out_path}")
    return out_path


def main():
    generate_outliers_html()
    generate_duplicates_html()


if __name__ == "__main__":
    main()
