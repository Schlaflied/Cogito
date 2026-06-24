"""
extract_diffs.py
----------------
Step 1 of self-model pipeline.
Extracts all git commits from the Obsidian vault, pulls diff content per commit,
and writes a timeline JSONL file: one record per commit.

Output: self-model/data/diffs.jsonl
Each record:
{
  "commit_hash": str,
  "timestamp": ISO8601 str,
  "unix_ts": int,
  "author": str,
  "message": str,
  "files_changed": [str],
  "diff_text": str,       # full unified diff text
  "net_lines": int,       # lines added - lines removed (cognitive output proxy)
  "hour": int,            # hour of day (0-23), proxy for mental state
  "weekday": int          # 0=Monday, 6=Sunday
}
"""

import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from git import Repo
except ImportError:
    print("Missing dependency: pip install gitpython")
    sys.exit(1)

# --- Config ---
VAULT_DIR = Path("D:/My knowledge/AI_Workflow")
OUTPUT_DIR = Path("D:/My knowledge/AI_Workflow/self-model/data")
OUTPUT_FILE = OUTPUT_DIR / "diffs.jsonl"

# Max diff text length per commit (chars). Prevents bloat from huge one-off imports.
MAX_DIFF_CHARS = 8000


def extract():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    repo = Repo(VAULT_DIR)
    commits = list(repo.iter_commits())
    total = len(commits)
    print(f"Found {total} commits. Extracting...")

    records = []
    for i, commit in enumerate(commits):
        if i % 200 == 0:
            print(f"  {i}/{total}...")

        # Timestamp
        ts = commit.committed_datetime.astimezone(timezone.utc)

        # Diff text vs parent
        try:
            if commit.parents:
                diff_text = repo.git.diff(
                    commit.parents[0].hexsha,
                    commit.hexsha,
                    unified=1,
                    diff_filter="d",   # skip deleted binary blobs
                )
            else:
                diff_text = repo.git.show(commit.hexsha, unified=1)
        except Exception:
            diff_text = ""  # skip problematic commits (binary files, temp files)

        # Truncate
        if len(diff_text) > MAX_DIFF_CHARS:
            diff_text = diff_text[:MAX_DIFF_CHARS] + "\n[truncated]"

        # Net lines (additions - deletions)
        added = diff_text.count("\n+")
        removed = diff_text.count("\n-")
        net_lines = added - removed

        # Files changed
        if commit.parents:
            files = [item.a_path or item.b_path for item in commit.diff(commit.parents[0])]
        else:
            files = []

        record = {
            "commit_hash": commit.hexsha[:8],
            "timestamp": ts.isoformat(),
            "unix_ts": int(ts.timestamp()),
            "author": commit.author.name,
            "message": commit.message.strip()[:200],
            "files_changed": files[:20],  # cap at 20
            "diff_text": diff_text,
            "net_lines": net_lines,
            "hour": ts.hour,
            "weekday": ts.weekday(),
        }
        records.append(record)

    # Sort chronologically
    records.sort(key=lambda r: r["unix_ts"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(records)} commits written to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    extract()
