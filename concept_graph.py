"""
concept_graph.py
----------------
Step 4 of self-model pipeline (Direction C: Identity Structure).
Reads all .md files from the vault, extracts concepts,
builds a co-occurrence graph, and outputs:
  1. concept_heatmap.png  — top 30 concepts × top 30, co-occurrence intensity
  2. concept_graph.html   — interactive force-directed graph
  3. concept_summary.json — machine-readable bridge concept list for AI querying
"""

import os, re, json, sys
from pathlib import Path
from collections import Counter, defaultdict

try:
    import numpy as np
except ImportError:
    os.system("pip install numpy"); import numpy as np

try:
    import networkx as nx
except ImportError:
    os.system("pip install networkx"); import networkx as nx

try:
    import plotly.graph_objects as go
except ImportError:
    os.system("pip install plotly"); import plotly.graph_objects as go

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    os.system("pip install matplotlib")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True

try:
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(20)
    HAS_JIEBA = True
except ImportError:
    os.system("pip install jieba")
    try:
        import jieba, jieba.posseg as pseg
        jieba.setLogLevel(20)
        HAS_JIEBA = True
    except:
        HAS_JIEBA = False

# --- Config ---
VAULT_DIR  = Path("D:/My knowledge/AI_Workflow")
OUTPUT_DIR = Path("D:/My knowledge/AI_Workflow/self-model/data")

MIN_FILE_CHARS      = 150   # skip very short files
MIN_FILES_FOR_TERM  = 3     # term must appear in this many files to matter
MAX_TERMS           = 250   # top N terms to include in graph
MIN_COOCCUR         = 2     # minimum edge weight to keep

# --- Stop words ---
EN_STOP = {
    "the","a","an","is","in","on","at","to","for","of","and","or","but",
    "with","from","by","as","this","that","it","be","was","are","were","been",
    "have","has","had","do","did","will","would","could","should","may","might",
    "not","no","so","if","when","then","which","who","what","how","where","can",
    "just","also","than","more","all","one","out","up","about","after","before",
    "into","through","over","between","each","other","your","my","our","their",
    "its","his","her","we","they","you","i","am","re","ve","ll","s","t","d",
    "note","file","use","used","make","made","get","got","see","like","know",
    "think","want","need","work","time","way","day","year","here","there",
    "some","any","new","old","good","well","even","now","still","add","update",
    "fix","create","delete","remove","change","done","todo","https","http",
    "com","www","github","obsidian","vault","backup","draft","misc","temp",
    "very","much","many","most","few","same","such","both","only","then","too",
}

ZH_STOP = {
    "的","了","在","是","我","有","和","就","不","人","都","一","上","也",
    "很","到","说","要","去","你","会","着","没有","看","好","自己","这",
    "那","他","她","它","这个","那个","什么","如何","因为","所以","但是",
    "可以","应该","需要","已经","还是","或者","然后","如果","对于","通过",
    "关于","以及","而且","不是","这样","那么","进行","使用","包括","相关",
    "方面","情况","问题","可能","非常","比较","主要","一些","很多","其他",
    "这些","那些","时候","一下","一种","一个","两个","没有","不会","不能",
    "工作","时间","方式","内容","系统","功能","数据","信息","文件","用户",
}

# --- Domain classifier ---
DOMAIN_RULES = [
    (r"career.ops|career_ops|applications\.md|pipeline\.md|portals", "career-ops"),
    (r"job.search|求职|面试|cover.letter|cv[-_]james|已投简历", "job-search"),
    (r"our.insync|insync|fanfic|同人|知妙|novel|fiction|syna|workshop|writing", "creative"),
    (r"plot.ark|plotark|plot_ark", "plot-ark"),
    (r"cogito|self.model|self_model", "cogito"),
    (r"\.py$|\.js$|\.mjs$|code|script|agent|coding", "code"),
    (r"legal|ltb|cra|t4a|evidence|finance|parking", "legal"),
    (r"learning|course|study|reading|book|notes?/", "learning"),
    (r"claude|openai|gpt|llm|ai.workflow|prompt", "ai-tools"),
]

SYSTEM_SKIP = {'.obsidian', '.smart-env', '.trash', 'archive', '__pycache__'}
SYSTEM_EXTS = {'.ajson', '.log', '.canvas', '.excalidraw'}


def is_system_path(p: Path) -> bool:
    for part in p.parts:
        if part in SYSTEM_SKIP or part.startswith('.'):
            return True
    return p.suffix.lower() in SYSTEM_EXTS


def classify_domain(p: Path) -> str:
    s = str(p).lower().replace("\\", "/")
    for pat, label in DOMAIN_RULES:
        if re.search(pat, s):
            return label
    return "general"


def clean_text(text: str) -> str:
    text = re.sub(r'^---[\s\S]*?---\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    text = re.sub(r'`[^`]+`', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    return text


def extract_terms(text: str):
    terms = []
    # English: words 4+ chars
    en = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{3,}\b', text)
    terms += [w.lower() for w in en if w.lower() not in EN_STOP]

    # Chinese
    zh_text = re.sub(r'[^一-鿿]', ' ', text)
    if HAS_JIEBA:
        for word, flag in pseg.cut(zh_text):
            if (flag.startswith('n') or flag in ('vn', 'an')) and word not in ZH_STOP and len(word) >= 2:
                terms.append(word)
    else:
        chars = re.findall(r'[一-鿿]{2,4}', zh_text)
        terms += [c for c in chars if c not in ZH_STOP]

    return terms


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect .md files
    print("Scanning vault for .md files...")
    all_files = []
    for f in VAULT_DIR.rglob("*.md"):
        if not is_system_path(f):
            try:
                content = f.read_text(encoding='utf-8', errors='replace')
                if len(content) >= MIN_FILE_CHARS:
                    all_files.append((f, content))
            except Exception:
                pass
    print(f"  {len(all_files)} content files found.")

    # 2. Extract terms per file
    print("Extracting concepts...")
    file_terms = []
    for fpath, content in all_files:
        text  = clean_text(content)
        terms = extract_terms(text)
        domain = classify_domain(fpath)
        # Unique terms per file (co-occurrence is file-level, not mention-level)
        file_terms.append((fpath, domain, set(terms)))

    # 3. Filter to cross-file terms
    term_file_count = Counter()
    for _, _, terms in file_terms:
        for t in terms:
            term_file_count[t] += 1

    valid = {t for t, n in term_file_count.items() if n >= MIN_FILES_FOR_TERM}
    top   = {t for t, _ in term_file_count.most_common(MAX_TERMS) if t in valid}
    print(f"  {len(valid)} terms in {MIN_FILES_FOR_TERM}+ files → keeping top {len(top)}.")

    # 4. Build co-occurrence graph
    print("Building co-occurrence graph...")
    G = nx.Graph()
    concept_domains = defaultdict(Counter)

    for fpath, domain, terms in file_terms:
        file_valid = [t for t in terms if t in top]
        for t in file_valid:
            G.add_node(t)
            G.nodes[t]['file_count'] = G.nodes[t].get('file_count', 0) + 1
            concept_domains[t][domain] += 1
        for i, t1 in enumerate(file_valid):
            for t2 in file_valid[i+1:]:
                if t1 != t2:
                    if G.has_edge(t1, t2):
                        G[t1][t2]['weight'] += 1
                    else:
                        G.add_edge(t1, t2, weight=1)

    weak = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < MIN_COOCCUR]
    G.remove_edges_from(weak)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # 5. Centrality
    print("Computing centrality...")
    betweenness = nx.betweenness_centrality(G, weight='weight')
    degree_w    = dict(G.degree(weight='weight'))

    # 6. Bridge concept list
    bridges = []
    for node in G.nodes():
        bridges.append({
            'concept':     node,
            'betweenness': round(betweenness.get(node, 0), 6),
            'degree':      degree_w.get(node, 0),
            'file_count':  term_file_count[node],
            'n_domains':   len(concept_domains[node]),
            'domains':     dict(concept_domains[node]),
        })
    bridges.sort(key=lambda x: x['betweenness'], reverse=True)

    # Save summary JSON
    summary = {
        'total_files':   len(all_files),
        'total_terms':   len(top),
        'graph_nodes':   G.number_of_nodes(),
        'graph_edges':   G.number_of_edges(),
        'top_bridges':   bridges[:40],
        'top_by_files':  sorted(bridges, key=lambda x: x['file_count'], reverse=True)[:40],
    }
    (OUTPUT_DIR / "concept_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f"  Summary → {OUTPUT_DIR / 'concept_summary.json'}")

    # 7. Heatmap (top 30 by degree)
    print("Generating heatmap...")
    top30 = [b['concept'] for b in sorted(bridges, key=lambda x: x['degree'], reverse=True)[:30]]
    matrix = np.zeros((len(top30), len(top30)))
    for i, c1 in enumerate(top30):
        for j, c2 in enumerate(top30):
            if G.has_edge(c1, c2):
                matrix[i][j] = G[c1][c2]['weight']

    fig_h, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(top30)))
    ax.set_yticks(range(len(top30)))
    ax.set_xticklabels(top30, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(top30, fontsize=8)
    plt.colorbar(im, ax=ax, label='Co-occurrence count')
    ax.set_title('Concept Co-occurrence Heatmap (Top 30 by Degree)', pad=20)
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "concept_heatmap.png"
    plt.savefig(str(heatmap_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap → {heatmap_path}")

    # 8. Interactive force-directed graph
    print("Generating interactive graph...")
    pos = nx.spring_layout(G, weight='weight', seed=42, k=1.5)

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
    }

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(width=0.5, color='#cccccc'),
        hoverinfo='none', showlegend=False,
    )

    node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_text.append(node)

        top_domain = max(concept_domains[node].items(), key=lambda x: x[1])[0] \
                     if concept_domains[node] else 'general'
        node_color.append(DOMAIN_COLORS.get(top_domain, '#95a5a6'))

        b = betweenness.get(node, 0)
        node_size.append(max(6, min(36, b * 600 + 6)))

        dom_str = ', '.join(f"{d}({n})" for d, n in
                            sorted(concept_domains[node].items(), key=lambda x: -x[1])[:4])
        node_hover.append(
            f"<b>{node}</b><br>"
            f"Files: {term_file_count[node]}<br>"
            f"Domains: {dom_str}<br>"
            f"Bridge score: {b:.4f}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition='top center',
        textfont=dict(size=7),
        hovertext=node_hover,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(color=node_color, size=node_size, line=dict(width=1, color='white')),
        showlegend=False,
    )

    legend_traces = [
        go.Scatter(x=[None], y=[None], mode='markers',
                   marker=dict(size=10, color=c), name=d)
        for d, c in DOMAIN_COLORS.items()
    ]

    fig_g = go.Figure(data=[edge_trace, node_trace] + legend_traces)
    fig_g.update_layout(
        title='Concept Graph: Cognitive Identity Structure',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='white', plot_bgcolor='white',
        height=900, legend_title='Primary Domain',
    )
    graph_path = OUTPUT_DIR / "concept_graph.html"
    fig_g.write_html(str(graph_path))
    print(f"  Graph → {graph_path}")

    # Print top bridges
    print("\n=== Top Bridge Concepts (betweenness) ===")
    for b in bridges[:20]:
        dom = ', '.join(f"{d}({n})" for d, n in
                        sorted(b['domains'].items(), key=lambda x: -x[1])[:3])
        print(f"  {b['concept']:<22} files={b['file_count']:3d}  "
              f"domains={b['n_domains']}  bridge={b['betweenness']:.4f}  [{dom}]")

    import webbrowser
    webbrowser.open(str(graph_path))
    webbrowser.open(str(heatmap_path))


if __name__ == "__main__":
    run()
