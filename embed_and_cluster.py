"""
embed_and_cluster.py
--------------------
Step 2 of self-model pipeline.
Reads diffs.jsonl, embeds each diff_text with text-embedding-3-small,
runs KMeans clustering to find topic groups,
and writes enriched records to clusters.jsonl.

Output: self-model/data/clusters.jsonl
Each record = original diff record + {
  "cluster": int,
  "embedding": [float, ...]   # 1536-dim, saved for visualisation
}

Run extract_diffs.py first.
"""

import os
import json
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: pip install openai")
    sys.exit(1)

try:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
except ImportError:
    print("Missing dependency: pip install numpy scikit-learn")
    sys.exit(1)

from dotenv import load_dotenv

# --- Config ---
VAULT_DIR = Path("D:/My knowledge/AI_Workflow")
DATA_DIR = Path("D:/My knowledge/AI_Workflow/self-model/data")
INPUT_FILE = DATA_DIR / "diffs.jsonl"
OUTPUT_FILE = DATA_DIR / "clusters.jsonl"

N_CLUSTERS = 12        # tune later — 12 gives readable granularity
BATCH_SIZE = 50        # OpenAI embedding batch size
EMBED_MODEL = "text-embedding-3-small"


def load_records():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def embed_batch(client, texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def run():
    load_dotenv(dotenv_path=VAULT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Loading records from {INPUT_FILE}...")
    records = load_records()
    print(f"  {len(records)} records loaded.")

    # Embed
    texts = [r["diff_text"] for r in records]
    embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Embedding {len(texts)} diffs in {total_batches} batches...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        if batch_num % 10 == 0:
            print(f"  Batch {batch_num}/{total_batches}...")
        try:
            vecs = embed_batch(client, batch)
            embeddings.extend(vecs)
        except Exception as e:
            print(f"  Error on batch {batch_num}: {e} — retrying in 5s")
            time.sleep(5)
            vecs = embed_batch(client, batch)
            embeddings.extend(vecs)

    print(f"Embedding complete. Running KMeans with {N_CLUSTERS} clusters...")
    X = np.array(embeddings)
    X_norm = normalize(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_norm)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec, label, emb in zip(records, labels, embeddings):
            rec["cluster"] = int(label)
            rec["embedding"] = emb
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Written to {OUTPUT_FILE}")

    # Cluster size summary
    from collections import Counter
    counts = Counter(labels)
    print("\nCluster sizes:")
    for c in sorted(counts):
        print(f"  Cluster {c:2d}: {counts[c]} commits")


if __name__ == "__main__":
    run()
