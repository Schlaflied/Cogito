# Cogito

> *"I think, therefore I am"* — but what, exactly, are you thinking?

**Cogito** is a methodology for building a cognitive self-model from your personal knowledge base and git history.

The premise: your Obsidian vault (or any git-versioned knowledge base) contains two kinds of signal — what you *changed* (git diffs) and what you *wrote* (note content). Together they are a timestamped record of your thinking across time. This pipeline extracts both, embeds them, and surfaces patterns you were not aware of.

Your data stays local. What you open-source is the method.

[中文说明 →](README.zh.md)

---

## Three directions

Cogito approaches self-modeling as three separable questions. You can run any combination.

### Direction A — Attention Map
*When did you work on what? How did your focus shift over time?*

```
extract_diffs.py       → data/diffs.jsonl        (timestamped diff records)
embed_and_cluster.py   → data/clusters.jsonl      (embedded + KMeans clustered)
visualise.py           → data/self_model_3d.html  (interactive timeline)
```

Each dot is one commit. Position in space = semantic content. X axis = time. Color = topic cluster. Dot size = output intensity (net lines changed).

---

### Direction B — Behavioral Patterns
*When are you most active? What does your work rhythm look like?*

```
temporal_patterns.py   → data/temporal_report.html  (hour/weekday/intensity heatmaps)
```

Uses the `hour`, `weekday`, and `net_lines` fields already in `diffs.jsonl`. No extra API calls. Reveals when different types of thinking happen — late-night commits vs. morning commits, high-output days vs. maintenance days.

---

### Direction C — Identity Structure
*What concepts appear across all your domains without you being aware of it?*

```
embed_notes.py            → data/notes_chunks.jsonl       (note content, chunked + embedded)
cross_domain_analysis.py  → data/cross_domain_map.html    (combined behavioral + content map)
                          → data/cross_domain_clusters.json  (machine-readable, for AI querying)
```

This is the hard direction. It combines two data layers:
- **Behavioral layer** (`diffs.jsonl`) — what you *did*
- **Cognitive layer** (`notes_chunks.jsonl`) — what you *thought and wrote*

Clusters that span multiple domains (creative writing, technical work, legal documents, job search) are candidates for your cognitive identity anchors — the patterns you return to everywhere without choosing to.

**The output is not a dashboard. It's a conversation starter.**

A 2D scatter plot of 5000+ points tells you nothing. The real output is `cross_domain_clusters.json` — a structured summary of which domains cluster together, with representative samples. Feed it to an AI that knows your context and ask: *what does this mean about me?*

The system surfaces the pattern. The AI explains what it means. You decide if it's true.

This is the part no existing self-tracking tool does: not visualization, but interpretation grounded in your actual behavioral record.

---

## The core research problem

> Systems don't acknowledge humans as humans.

Seat time ≠ competence. Keywords ≠ capability. A formal record ≠ what actually happened. Every existing classification system optimizes for what it can measure, not for what is real.

Cogito is one answer to this: if you can reconstruct a person's cognitive structure from their behavioral traces, you have evidence that doesn't depend on institutional categories. The person who falls through every existing classification still left a record — in the diffs, in the notes, in the patterns that span everything they ever worked on.

This is also why git diffs matter more than just reading notes:

> A note is a final state. A diff is a decision.

When you delete a sentence and rewrite it, that's a data point about how your thinking changed. Standard RAG over your notes throws this away. Cogito keeps it.

---

## Setup

```bash
pip install gitpython openai umap-learn scikit-learn plotly numpy python-dotenv matplotlib networkx jieba
```

Create a `.env` file in your vault root:
```
OPENAI_API_KEY=sk-...
```

Edit `VAULT_DIR` at the top of each script to point to your vault (must be a git repo).

---

## Approximate API cost

| Step | Records | Est. cost |
|------|---------|-----------|
| Direction A (diffs) | ~4000 commits | ~$0.50–1.00 USD |
| Direction C (notes) | ~3000 chunks | ~$0.30–0.60 USD |

Uses `text-embedding-3-small`. Resume support built in — safe to interrupt and restart.

---

## Related work

- [Plot Ark](https://github.com/Schlaflied/Plot-Ark) — the institutional-scale version of this problem: xAPI behavioral analytics for learning systems
- [career-ops](https://github.com/santifer/career-ops) — OSS job search pipeline; SQLite architecture RFC (#919) contributed here

---

## What this is not

- Not a therapy tool
- Not a productivity tracker
- Not a replacement for introspection

It's a mirror with a longer memory than you have.

---

## License

MIT — use it, fork it, run it on your own data.

---

*Built by someone who couldn't figure out what was driving them, so they built a system to find out.*
