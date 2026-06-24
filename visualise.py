"""
visualise.py
------------
Step 3 of self-model pipeline.
Reads clusters.jsonl, reduces embeddings to 3D with UMAP,
and renders an interactive 3D scatter plot.

Axes:
  X: time (unix timestamp — chronological)
  Y: UMAP dim 1 (topic space)
  Z: UMAP dim 2 (topic space)
  Color: cluster label
  Size: net_lines (cognitive output intensity)
  Hover: timestamp, commit message, files changed

Run embed_and_cluster.py first.
"""

import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("Missing: pip install numpy")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("Missing: pip install plotly")
    sys.exit(1)

try:
    from umap import UMAP
except ImportError:
    print("Missing: pip install umap-learn")
    sys.exit(1)

# --- Config ---
DATA_DIR = Path("D:/My knowledge/AI_Workflow/self-model/data")
INPUT_FILE = DATA_DIR / "clusters.jsonl"
OUTPUT_HTML = DATA_DIR / "self_model_3d.html"

CLUSTER_COLORS = [
    "#1a1a2e", "#16213e", "#0f3460", "#533483",
    "#e94560", "#f5a623", "#7ed321", "#4a90e2",
    "#9b59b6", "#1abc9c", "#e67e22", "#e74c3c",
]


def load_records():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run():
    print(f"Loading {INPUT_FILE}...")
    records = load_records()
    print(f"  {len(records)} records.")

    embeddings = np.array([r["embedding"] for r in records])
    timestamps = np.array([r["unix_ts"] for r in records])
    clusters = np.array([r["cluster"] for r in records])
    net_lines = np.array([max(1, abs(r["net_lines"])) for r in records])

    # Normalise timestamps to 0-1 for X axis
    ts_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1)

    print("Running UMAP (2D projection of embedding space)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d = reducer.fit_transform(embeddings)

    # Build hover text
    hover = []
    for r in records:
        msg = r["message"][:80].replace("<", "&lt;")
        files = ", ".join(r["files_changed"][:3]) if r["files_changed"] else "—"
        hover.append(
            f"<b>{r['timestamp'][:10]}</b> {r['timestamp'][11:16]}<br>"
            f"{msg}<br>"
            f"Files: {files}<br>"
            f"Net lines: {r['net_lines']}"
        )

    # One trace per cluster
    fig = go.Figure()
    for c in sorted(set(clusters)):
        mask = clusters == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=ts_norm[mask],
            y=umap_2d[mask, 0],
            z=umap_2d[mask, 1],
            mode="markers",
            name=f"Cluster {c}",
            marker=dict(
                size=np.clip(np.log1p(net_lines[mask]) * 2, 2, 10),
                color=color,
                opacity=0.75,
            ),
            text=[hover[i] for i, m in enumerate(mask) if m],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Self-Model: Cognitive Timeline (3D)",
        scene=dict(
            xaxis_title="Time →",
            yaxis_title="Topic Space (UMAP dim 1)",
            zaxis_title="Topic Space (UMAP dim 2)",
            bgcolor="#f8f8f8",
        ),
        paper_bgcolor="#ffffff",
        legend_title="Cluster",
        height=800,
    )

    fig.write_html(str(OUTPUT_HTML))
    print(f"\nDone. Open in browser: {OUTPUT_HTML}")

    # Also open automatically
    import webbrowser
    webbrowser.open(str(OUTPUT_HTML))


if __name__ == "__main__":
    run()
