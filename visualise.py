"""
visualise.py
------------
Step 3 of self-model pipeline.
Reads clusters.jsonl, filters noise commits, reduces embeddings to 3D with UMAP,
auto-labels clusters from commit messages, and renders an interactive 3D scatter.

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
import re
import sys
from collections import Counter
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
    "#e94560", "#f5a623", "#7ed321", "#4a90e2",
    "#9b59b6", "#1abc9c", "#e67e22", "#e74c3c",
    "#3498db", "#2ecc71", "#d35400", "#8e44ad",
]

# Pure system/cache file patterns — commits that ONLY touch these are noise
SYSTEM_FILE_PATTERNS = [
    r"^\.obsidian/",
    r"^\.smart-env/",
    r"^\.trash/",
    r"\.ajson$",      # smart connections vector cache
    r"\.log$",
]

# Words to ignore when auto-labeling clusters
STOP_WORDS = {
    "the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "with", "from", "by", "as", "this", "that", "it", "be",
    "was", "are", "were", "been", "have", "has", "had", "do", "did", "will",
    "would", "could", "should", "may", "might", "not", "no", "so", "if",
    "add", "added", "update", "updated", "fix", "fixed", "change", "changed",
    "remove", "removed", "new", "old", "file", "files", "note", "notes",
    "obsidian", "vault", "backup", "sync", "commit", "initial", "draft",
    "misc", "minor", "temp", "test", "wip",
}


def normalise_path(f):
    return f.replace("\\", "/")


def is_system_file(f):
    f = normalise_path(f)
    return any(re.search(p, f) for p in SYSTEM_FILE_PATTERNS)


def is_noise(record):
    files = record.get("files_changed", [])
    if not files:
        return False  # unknown files — keep
    # Noise only if EVERY changed file is a system/cache file (no content)
    return all(is_system_file(f) for f in files)


FILE_THEME_RULES = [
    (r"career.ops|career_ops|applications|pipeline|portals|interview", "career-ops"),
    (r"job.search|求职|面试|cover.letter|resume|cv[-_]james|cv[-_]yuting", "job-search"),
    (r"our.insync|insync|fanfic|同人|知妙|novel|fiction", "creative-writing"),
    (r"plot.ark|plotark|plot_ark", "plot-ark"),
    (r"cogito|self.model|self_model", "cogito"),
    (r"\.py$|\.js$|\.mjs$|\.ts$|\.sh$|code|script|agent", "code"),
    (r"learning|course|study|读书|笔记|notes?/", "learning"),
    (r"journal|diary|日记|reflection|review", "journal"),
    (r"obsidian|plugin|template|workspace", "obsidian-config"),
]


def infer_theme_from_files(files):
    """Map file paths to a theme label, ignoring system/cache files."""
    content_files = [f for f in files if not is_system_file(f)]
    if not content_files:
        return "system-only"
    text = " ".join(normalise_path(f) for f in content_files).lower()
    for pattern, label in FILE_THEME_RULES:
        if re.search(pattern, text):
            return label
    # Fallback: top content directory segment
    dirs = []
    for f in content_files:
        parts = normalise_path(f).split("/")
        if len(parts) > 1:
            top_dir = re.sub(r"[^\x00-\x7F]", "", parts[0]).strip()
            if top_dir:
                dirs.append(top_dir)
    if dirs:
        return Counter(dirs).most_common(1)[0][0][:20]
    return "misc"


def auto_label_cluster(records_in_cluster):
    """Extract theme from file paths in the cluster."""
    theme_counts = Counter()
    for r in records_in_cluster:
        theme = infer_theme_from_files(r.get("files_changed", []))
        theme_counts[theme] += 1
    top = [t for t, _ in theme_counts.most_common(3)]
    return " · ".join(top) if top else "misc"


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
    all_records = load_records()
    print(f"  {len(all_records)} records total.")

    records = [r for r in all_records if not is_noise(r)]
    print(f"  {len(records)} records after filtering noise ({len(all_records) - len(records)} removed).")

    embeddings = np.array([r["embedding"] for r in records])
    timestamps = np.array([r["unix_ts"] for r in records])
    clusters = np.array([r["cluster"] for r in records])
    net_lines = np.array([max(1, abs(r["net_lines"])) for r in records])

    # Auto-label each cluster
    cluster_labels = {}
    for c in sorted(set(clusters)):
        recs_in_c = [r for r, cl in zip(records, clusters) if cl == c]
        cluster_labels[c] = f"[{c}] {auto_label_cluster(recs_in_c)}"

    # Convert unix timestamps to fractional year for readable X axis
    # e.g. 2024.5 = mid-2024
    ts_norm = timestamps / (365.25 * 24 * 3600) + 1970

    print("Running UMAP (2D projection of embedding space)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d = reducer.fit_transform(embeddings)

    # Build hover text
    hover = []
    for r in records:
        msg = r["message"][:100].replace("<", "&lt;")
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
        label = cluster_labels[c]
        indices = [i for i, m in enumerate(mask) if m]
        fig.add_trace(go.Scatter3d(
            x=ts_norm[mask],
            y=umap_2d[mask, 0],
            z=umap_2d[mask, 1],
            mode="markers",
            name=label,
            marker=dict(
                size=np.clip(np.log1p(net_lines[mask]) * 2, 2, 10),
                color=color,
                opacity=0.75,
            ),
            text=[hover[i] for i in indices],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Self-Model: Cognitive Timeline (3D)",
        scene=dict(
            xaxis_title="Year →",
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

    # Print cluster labels for reference
    print("\nCluster labels:")
    for c, label in sorted(cluster_labels.items()):
        count = int((clusters == c).sum())
        print(f"  {label}  ({count} commits)")

    import webbrowser
    webbrowser.open(str(OUTPUT_HTML))


if __name__ == "__main__":
    run()
