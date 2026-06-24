"""
embed_notes.py
--------------
Step 4b of self-model pipeline.
Reads all .md content files from the vault, chunks them (~400 words each),
embeds with text-embedding-3-small, and saves to notes_chunks.jsonl.

Output: self-model/data/notes_chunks.jsonl
Each record:
{
  "filepath": str,          # relative to vault root
  "domain": str,            # inferred domain label
  "chunk_index": int,
  "text": str,              # chunk content
  "embedding": [float],     # 1536-dim
  "file_mtime": int,        # unix timestamp of file modification
  "word_count": int
}

Run after extract_diffs.py. Diff embeddings (clusters.jsonl) are the
behavioral layer; note embeddings are the cognitive content layer.
"""

import os, re, json, sys, time
from pathlib import Path
from collections import Counter

try:
    from openai import OpenAI
except ImportError:
    os.system("pip install openai"); from openai import OpenAI

from dotenv import load_dotenv

# --- Config ---
VAULT_DIR   = Path("D:/My knowledge/AI_Workflow")
OUTPUT_DIR  = Path("D:/My knowledge/AI_Workflow/self-model/data")
OUTPUT_FILE = OUTPUT_DIR / "notes_chunks.jsonl"

WORDS_PER_CHUNK = 200   # ~267 tokens, comfortable margin
MAX_CHUNK_WORDS = 800   # hard truncation — ~1066 tokens, well under 8192
MIN_CHUNK_WORDS = 40    # skip tiny chunks
BATCH_SIZE      = 20    # smaller batches to stay under 300K tokens/request
EMBED_MODEL     = "text-embedding-3-small"

SYSTEM_SKIP = {'.obsidian', '.smart-env', '.trash', 'archive', '__pycache__'}
SYSTEM_EXTS = {'.ajson', '.log', '.canvas', '.excalidraw', '.json'}

DOMAIN_RULES = [
    (r"career.ops|career_ops|applications\.md|pipeline\.md|portals", "career-ops"),
    (r"job.search|求职|面试|cover.letter|cv[-_]james|已投简历", "job-search"),
    (r"our.insync|insync|fanfic|同人|知妙|novel|fiction|syna|workshop|writing", "creative"),
    (r"plot.ark|plotark|plot_ark", "plot-ark"),
    (r"cogito|self.model|self_model", "cogito"),
    (r"legal|ltb|cra|t4a|evidence|finance|parking", "legal"),
    (r"learning|course|study|reading|book", "learning"),
    (r"claude|openai|gpt|llm|ai.workflow|prompt", "ai-tools"),
]


def is_system_path(p: Path) -> bool:
    for part in p.parts:
        if part in SYSTEM_SKIP or (part.startswith('.') and part not in ('.', '..')):
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
    text = re.sub(r'!\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+', ' ', text)          # collapse horizontal whitespace only
    text = re.sub(r'\n{3,}', '\n\n', text)       # normalise 3+ newlines to paragraph break
    return text.strip()


def chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK):
    """Split text into chunks of ~words_per_chunk words, respecting paragraph breaks."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        # Hard-truncate individual giant paragraphs
        words = para.split()
        if len(words) > MAX_CHUNK_WORDS:
            para = ' '.join(words[:MAX_CHUNK_WORDS])
        para_words = len(para.split())
        if current_words + para_words > words_per_chunk and current:
            chunks.append(' '.join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        chunks.append(' '.join(current))

    # Final safety: truncate any chunk still over limit
    return [' '.join(c.split()[:MAX_CHUNK_WORDS]) for c in chunks]


def embed_batch(client, texts):
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(dotenv_path=VAULT_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env"); sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Resume support: skip already-processed files
    done_files = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_files.add(r['filepath'])
                except Exception:
                    pass
        print(f"Resuming: {len(done_files)} files already embedded.")

    # Collect .md files
    print("Scanning vault...")
    all_files = []
    for f in VAULT_DIR.rglob("*.md"):
        if not is_system_path(f):
            rel = str(f.relative_to(VAULT_DIR)).replace("\\", "/")
            if rel not in done_files:
                try:
                    content = f.read_text(encoding='utf-8', errors='replace')
                    if len(content) >= 100:
                        all_files.append((f, rel, content))
                except Exception:
                    pass

    print(f"  {len(all_files)} files to process (+ {len(done_files)} already done).")

    # Chunk all files
    all_chunks = []  # (rel_path, domain, chunk_idx, text, mtime)
    for fpath, rel, content in all_files:
        text = clean_text(content)
        chunks = chunk_text(text)
        domain = classify_domain(fpath)
        mtime = int(fpath.stat().st_mtime)
        for i, chunk in enumerate(chunks):
            if len(chunk.split()) >= MIN_CHUNK_WORDS:
                all_chunks.append((rel, domain, i, chunk, mtime))

    print(f"  {len(all_chunks)} chunks to embed.")
    if not all_chunks:
        print("Nothing to embed."); return

    # Embed in batches
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding {len(all_chunks)} chunks in {total_batches} batches...")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out:
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            if batch_num % 10 == 0 or batch_num == 1:
                print(f"  Batch {batch_num}/{total_batches}...")

            texts = [c[3] for c in batch]
            # Replace empty
            texts = [t if t.strip() else "[empty]" for t in texts]

            try:
                embeddings = embed_batch(client, texts)
            except Exception as e:
                print(f"  Error batch {batch_num}: {e} — retrying individually")
                time.sleep(2)
                embeddings = []
                for t in texts:
                    try:
                        emb = embed_batch(client, [t[:3000]])[0]  # hard truncate at 3000 chars as last resort
                        embeddings.append(emb)
                    except Exception as e2:
                        print(f"    Skipping chunk: {str(e2)[:60]}")
                        embeddings.append(None)

            for (rel, domain, chunk_idx, text, mtime), emb in zip(batch, embeddings):
                if emb is None:
                    continue
                record = {
                    "filepath":    rel,
                    "domain":      domain,
                    "chunk_index": chunk_idx,
                    "text":        text,
                    "embedding":   emb,
                    "file_mtime":  mtime,
                    "word_count":  len(text.split()),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone. Output: {OUTPUT_FILE}")
    # Count
    total = sum(1 for _ in open(OUTPUT_FILE, encoding='utf-8'))
    print(f"Total records in file: {total}")


if __name__ == "__main__":
    run()
