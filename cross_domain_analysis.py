"""
cross_domain_analysis.py
------------------------
Direction C: Identity Structure.

Loads both data layers:
  - clusters.jsonl     : diff embeddings (behavioral layer — what you DID)
  - notes_chunks.jsonl : note embeddings (cognitive layer — what you THOUGHT/WROTE)

Combines them in a shared embedding space, runs UMAP + KMeans clustering,
finds clusters that span multiple domains, and outputs:
  - cross_domain_clusters.json  : machine-readable summary for AI interpretation
  - cross_domain_map.html       : interactive 2D map (not 3D — actually readable)

The clusters that appear in the most domains with the highest semantic coherence
are candidates for your cognitive identity anchors — things you return to
across all contexts without being aware of it.
"""

import json, sys, os
from pathlib import Path
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:
    os.system("pip install numpy"); import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
except ImportError:
    os.system("pip install scikit-learn"); from sklearn.cluster import KMeans; from sklearn.preprocessing import normalize

try:
    from umap import UMAP
except ImportError:
    os.system("pip install umap-learn"); from umap import UMAP

try:
    import plotly.graph_objects as go
except ImportError:
    os.system("pip install plotly"); import plotly.graph_objects as go

DATA_DIR   = Path("D:/My knowledge/AI_Workflow/self-model/data")
DIFFS_FILE = DATA_DIR / "clusters.jsonl"
NOTES_FILE = DATA_DIR / "notes_chunks.jsonl"
OUT_JSON   = DATA_DIR / "cross_domain_clusters.json"
OUT_HTML   = DATA_DIR / "cross_domain_map.html"

N_CLUSTERS = 16   # more granularity than the diff-only clustering

DOMAIN_COLORS = {
    'career-ops':  '#e94560',
    'job-search':  '#f5a623',
    'creative':    '#9b59b6',
    'plot-ark':    '#1abc9c',
    'cogito':      '#4a90e2',
    'code':        '#7ed321',
    'legal':       '#e67e22',
    'learning':    '#3498db',
    'ai-tools':    '#2ecc71',
    'general':     '#95a5a6',
    'diff':        '#cccccc',   # behavioral layer marker
}

SYSTEM_NOISE_MSG = "vault backup:"


def load_diffs():
    records = []
    with open(DIFFS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            # Skip pure noise commits
            if r.get('message', '').lower().startswith(SYSTEM_NOISE_MSG):
                files = r.get('files_changed', [])
                if not files or all(
                    f.startswith('.obsidian') or f.startswith('.smart-env')
                    for f in files
                ):
                    continue
            records.append({
                'source':    'diff',
                'domain':    _diff_domain(r),
                'text':      r.get('message', '') + ' ' + ' '.join(r.get('files_changed', [])[:5]),
                'embedding': r['embedding'],
                'meta': {
                    'timestamp': r.get('timestamp', ''),
                    'message':   r.get('message', '')[:120],
                    'files':     r.get('files_changed', [])[:3],
                }
            })
    return records


def load_notes():
    records = []
    with open(NOTES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            records.append({
                'source':    'note',
                'domain':    r.get('domain', 'general'),
                'text':      r.get('text', '')[:200],
                'embedding': r['embedding'],
                'meta': {
                    'filepath':    r.get('filepath', ''),
                    'chunk_index': r.get('chunk_index', 0),
                    'preview':     r.get('text', '')[:150],
                }
            })
    return records


DIFF_DOMAIN_RULES = [
    ('career-ops',  ['career', 'applications', 'pipeline', 'portals', 'career_ops']),
    ('job-search',  ['job', 'resume', 'cover', 'interview', 'cv-james', '求职', '面试']),
    ('creative',    ['insync', 'fanfic', 'novel', 'syna', 'workshop', '同人', '知妙']),
    ('plot-ark',    ['plot', 'ark', 'plotark', 'plot_ark']),
    ('cogito',      ['cogito', 'self-model', 'self_model']),
    ('legal',       ['ltb', 'cra', 't4a', 'evidence', 'legal', 'parking']),
    ('learning',    ['learning', 'course', 'study', 'reading']),
    ('ai-tools',    ['claude', 'openai', 'gpt', 'llm', 'prompt']),
]

def _diff_domain(r):
    text = ' '.join(r.get('files_changed', [])).lower() + ' ' + r.get('message', '').lower()
    for domain, keywords in DIFF_DOMAIN_RULES:
        if any(k in text for k in keywords):
            return domain
    return 'general'


def run():
    print("Loading diff embeddings (behavioral layer)...")
    diffs = load_diffs()
    print(f"  {len(diffs)} diff records.")

    print("Loading note embeddings (cognitive content layer)...")
    notes = load_notes()
    print(f"  {len(notes)} note chunks.")

    all_records = diffs + notes
    print(f"  Combined: {len(all_records)} records.")

    embeddings = np.array([r['embedding'] for r in all_records])
    print(f"  Embedding matrix: {embeddings.shape}")

    print("Running UMAP (2D)...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.05)
    umap_2d = reducer.fit_transform(embeddings)

    print(f"Running KMeans (k={N_CLUSTERS})...")
    X_norm = normalize(embeddings)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)

    # Annotate records
    for i, r in enumerate(all_records):
        r['umap_x'] = float(umap_2d[i, 0])
        r['umap_y'] = float(umap_2d[i, 1])
        r['cluster'] = int(labels[i])

    # Analyse each cluster
    print("Analysing cross-domain clusters...")
    cluster_info = []
    for c in range(N_CLUSTERS):
        members = [r for r in all_records if r['cluster'] == c]
        if not members:
            continue

        domain_counts = Counter(r['domain'] for r in members)
        source_counts = Counter(r['source'] for r in members)
        n_domains = len(domain_counts)

        # Representative texts (sample from each domain)
        samples = []
        for domain in list(domain_counts)[:4]:
            dom_members = [r for r in members if r['domain'] == domain][:2]
            for m in dom_members:
                if m['source'] == 'note':
                    samples.append(f"[{domain}] {m['meta']['preview'][:100]}")
                else:
                    samples.append(f"[{domain}/diff] {m['meta']['message'][:80]}")

        cluster_info.append({
            'cluster':       c,
            'size':          len(members),
            'n_domains':     n_domains,
            'domains':       dict(domain_counts),
            'sources':       dict(source_counts),
            'cross_domain':  n_domains >= 3,
            'samples':       samples[:8],
        })

    cluster_info.sort(key=lambda x: (-x['n_domains'], -x['size']))

    # Save JSON
    output = {
        'total_records':   len(all_records),
        'diff_records':    len(diffs),
        'note_records':    len(notes),
        'n_clusters':      N_CLUSTERS,
        'clusters':        cluster_info,
        'cross_domain_clusters': [c for c in cluster_info if c['cross_domain']],
    }
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  Cluster summary → {OUT_JSON}")

    # Print cross-domain clusters
    print(f"\n=== Cross-Domain Clusters (spanning 3+ domains) ===")
    for c in cluster_info:
        if c['n_domains'] >= 3:
            dom_str = ', '.join(f"{d}({n})" for d, n in
                                sorted(c['domains'].items(), key=lambda x: -x[1]))
            print(f"\nCluster {c['cluster']} — {c['size']} records, {c['n_domains']} domains")
            print(f"  Domains: {dom_str}")
            print(f"  Samples:")
            for s in c['samples'][:4]:
                print(f"    • {s[:100]}")

    # 2D interactive map
    print("\nGenerating 2D map...")
    fig = go.Figure()

    # One trace per domain (colour by domain, shape by source)
    domain_groups = defaultdict(list)
    for i, r in enumerate(all_records):
        domain_groups[r['domain']].append(i)

    for domain, idxs in domain_groups.items():
        recs = [all_records[i] for i in idxs]
        notes_in = [r for r in recs if r['source'] == 'note']
        diffs_in = [r for r in recs if r['source'] == 'diff']
        color = DOMAIN_COLORS.get(domain, '#aaaaaa')

        for subset, symbol, name_suffix in [
            (notes_in, 'circle',     ''),
            (diffs_in, 'cross-thin', ' (diff)'),
        ]:
            if not subset: continue
            hover = []
            for r in subset:
                if r['source'] == 'note':
                    hover.append(f"<b>[{r['domain']}]</b> note<br>{r['meta']['preview'][:120]}<br>cluster {r['cluster']}")
                else:
                    hover.append(f"<b>[{r['domain']}]</b> diff<br>{r['meta']['message'][:100]}<br>cluster {r['cluster']}")

            fig.add_trace(go.Scatter(
                x=[r['umap_x'] for r in subset],
                y=[r['umap_y'] for r in subset],
                mode='markers',
                name=f"{domain}{name_suffix}",
                marker=dict(
                    symbol=symbol,
                    size=5 if r['source'] == 'note' else 4,
                    color=color,
                    opacity=0.6,
                    line=dict(width=0),
                ),
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            ))

    fig.update_layout(
        title='Cross-Domain Semantic Map (Notes + Diffs)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
        paper_bgcolor='white', plot_bgcolor='#fafafa',
        height=850,
        legend_title='Domain · Source',
    )
    fig.write_html(str(OUT_HTML))
    print(f"  Map → {OUT_HTML}")

    import webbrowser
    webbrowser.open(str(OUT_HTML))
    print("\nDone.")


if __name__ == "__main__":
    run()
