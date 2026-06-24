# Cogito

> *"I think, therefore I am"* — but what, exactly, are you thinking?

**Cogito** is a methodology for building a cognitive self-model from your personal knowledge base git history.

The premise: if you use a tool like Obsidian with git versioning, your commit history is a timestamped record of your thinking — every idea you wrote, revised, deleted, and returned to. This pipeline extracts that history, embeds it, clusters it, and renders it as a 3D timeline so you can see your own cognitive patterns across time.

Your data stays local. What you open-source is the method.

---

## What it does

```
Git history (4000+ commits)
        ↓
extract_diffs.py      → data/diffs.jsonl      (timestamped diff records)
        ↓
embed_and_cluster.py  → data/clusters.jsonl   (embedded + KMeans clustered)
        ↓
visualise.py          → data/self_model_3d.html (interactive 3D scatter)
```

**Three axes:**
- **X** — time (chronological)
- **Y / Z** — topic space (UMAP projection of embedding clusters)
- **Dot size** — cognitive output intensity (net lines changed)
- **Color** — topic cluster

---

## Why git diffs, not just notes?

A note is a final state. A diff is a decision.

When you delete a sentence and rewrite it, that's a data point about how your thinking changed. When you return to the same topic six months later, that's a signal about what's load-bearing in your mind. Standard RAG over your notes throws this away. Cogito keeps it.

---

## Setup

```bash
pip install gitpython openai umap-learn scikit-learn plotly numpy python-dotenv
```

Create a `.env` file in your vault root:
```
OPENAI_API_KEY=sk-...
```

---

## Usage

Edit the `VAULT_DIR` path at the top of each script to point to your Obsidian vault (must be a git repo).

```bash
# Step 1: Extract git history
python extract_diffs.py

# Step 2: Embed and cluster (~$0.50–1.00 USD for 4000 commits)
python embed_and_cluster.py

# Step 3: Visualise (opens in browser)
python visualise.py
```

---

## Research problems this addresses

**1. Inverse cognitive reconstruction from behavioral output**
Can we infer a person's underlying mental model from what they write and how they revise it? This is an open problem in cognitive science, AI alignment, and educational analytics. Cogito treats git diffs as behavioral signals and attempts a bottom-up reconstruction.

**2. Knowledge dependency mapping over time**
Which ideas scaffold which other ideas in a person's learning trajectory? The timeline view surfaces these dependencies as temporal co-occurrence patterns — not declared by the user, but emergent from their actual writing behavior.

---

## What this is not

- Not a therapy tool
- Not a productivity tracker
- Not a replacement for introspection

It's a mirror with a longer memory than you have.

---

## Related work

- [Plot Ark](https://github.com/Schlaflied/Plot-Ark) — xAPI behavioral analytics for learning systems (the institutional-scale version of this problem)
- [career-ops](https://github.com/santifer/career-ops) — the OSS project where the SQLite architecture RFC (#919) that inspired this pipeline was contributed

---

## License

MIT — use it, fork it, run it on your own data.

---

*Built by someone who couldn't figure out what was driving them, so they built a system to find out.*
