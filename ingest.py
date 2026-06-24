"""Cogito — Ingestion Layer

Normalises different source types into a local folder of .md files
that can be processed by direction_c/embed_notes.py.

Usage:
  python ingest.py --source folder  --path /path/to/markdown/folder
  python ingest.py --source notion  --path /path/to/notion/export.zip
  python ingest.py --source notion  --path /path/to/notion/export/folder
  python ingest.py --source gdocs   --path /path/to/google-takeout.zip

Output:
  data/ingested/<source>/          normalised .md files
  (then run: python direction_c/embed_notes.py --source-dir data/ingested/<source>/)

Supported sources:
  folder   Any folder of .md or .txt files. No history — content layer only.
  notion   Notion export (zip or folder). Strips frontmatter, keeps body.
  gdocs    Google Takeout zip. Extracts .md files exported via Docs → Download as MD.
           Note: Google Takeout does NOT include revision history — content layer only.
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

OUT_BASE = Path(__file__).parent / "data" / "ingested"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_notion_frontmatter(text: str) -> str:
    """Remove Notion's property block (everything between the first --- pair)."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text


def clean_md(text: str) -> str:
    """Light clean: collapse 3+ blank lines, strip trailing whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_out(out_dir: Path, name: str, text: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def ingest_folder(src: Path, out_dir: Path):
    """Copy .md and .txt files from a folder, preserving subdirectory structure."""
    files = list(src.rglob("*.md")) + list(src.rglob("*.txt"))
    count = 0
    for f in files:
        rel  = f.relative_to(src)
        text = clean_md(f.read_text(encoding="utf-8", errors="replace"))
        if text:
            target = out_dir / rel.with_suffix(".md")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            count += 1
    return count


def ingest_notion(src: Path, out_dir: Path):
    """
    Notion export: zip or folder containing .md files with YAML frontmatter.
    Strips the frontmatter block; keeps body text.
    """
    if src.suffix == ".zip":
        tmp = out_dir.parent / "_notion_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = tmp

    files = list(src.rglob("*.md"))
    count = 0
    for f in files:
        raw  = f.read_text(encoding="utf-8", errors="replace")
        text = clean_md(strip_notion_frontmatter(raw))
        if text:
            rel    = f.relative_to(src)
            target = out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            count += 1

    # cleanup temp extraction
    if (out_dir.parent / "_notion_tmp").exists():
        shutil.rmtree(out_dir.parent / "_notion_tmp")

    return count


def ingest_gdocs(src: Path, out_dir: Path):
    """
    Google Takeout zip.
    Docs exported as .md are inside Takeout/Google Docs/*.md
    This extracts and cleans them. No revision history available.
    """
    if not src.suffix == ".zip":
        print("  gdocs source must be a .zip file (Google Takeout archive).")
        return 0

    count = 0
    with zipfile.ZipFile(src) as z:
        md_files = [n for n in z.namelist() if n.endswith(".md")]
        for name in md_files:
            with z.open(name) as fh:
                raw  = fh.read().decode("utf-8", errors="replace")
            text = clean_md(raw)
            if text:
                safe_name = Path(name).name
                write_out(out_dir, safe_name, text)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCE_FN = {
    "folder": ingest_folder,
    "notion": ingest_notion,
    "gdocs":  ingest_gdocs,
}

def main():
    parser = argparse.ArgumentParser(description="Cogito ingestion layer")
    parser.add_argument("--source", required=True, choices=SOURCE_FN.keys(),
                        help="Source type: folder | notion | gdocs")
    parser.add_argument("--path",   required=True,
                        help="Path to source folder or zip file")
    args = parser.parse_args()

    src     = Path(args.path)
    out_dir = OUT_BASE / args.source

    if not src.exists():
        print(f"Error: path not found: {src}")
        return

    print(f"Ingesting from {src} ({args.source}) ...")
    fn    = SOURCE_FN[args.source]
    count = fn(src, out_dir)
    print(f"  {count} files written to {out_dir}")
    print()
    print("Next step:")
    print(f"  Edit direction_c/embed_notes.py — set SOURCE_DIR = Path('{out_dir}')")
    print(f"  Then run: python direction_c/embed_notes.py")


if __name__ == "__main__":
    main()
