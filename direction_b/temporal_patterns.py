"""Direction B — Behavioral Patterns
Reads data/diffs.jsonl, outputs data/temporal_report.html

No API calls. Uses hour / weekday / net_lines already in diffs.jsonl.
"""
import json
import webbrowser
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT   = DATA_DIR / "diffs.jsonl"
OUTPUT  = DATA_DIR / "temporal_report.html"

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SYSTEM_PREFIXES = {".obsidian/", ".smart-env/"}

DOMAIN_RULES = [
    ("creative",   [r"Our Insync", r"Creative", r"Writing", r"Fiction"]),
    ("legal",      [r"Legal", r"CRA", r"LTB", r"T4A", r"CEC"]),
    ("plot-ark",   [r"Plot.Ark", r"PlotArk", r"xAPI"]),
    ("career-ops", [r"career.ops", r"career_ops"]),
    ("ai-tools",   [r"AI.Workflow", r"AI_Workflow", r"ATS", r"self.model"]),
    ("learning",   [r"Learning", r"Course", r"Study", r"MPEd", r"Academic"]),
    ("job-search", [r"Job.Search", r"求起", r"application", r"resume", r"cover"]),
]

import re

def classify_domain(files):
    for name, patterns in DOMAIN_RULES:
        for f in files:
            f_norm = f.replace("\\", "/")
            for p in patterns:
                if re.search(p, f_norm, re.IGNORECASE):
                    return name
    return "general"


def is_system(r):
    files = r.get("files_changed", [])
    return files and all(
        any(f.replace("\\", "/").startswith(s) for s in SYSTEM_PREFIXES)
        for f in files
    )


def load():
    records = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = [r for r in records if not is_system(r)]
    return records


def heatmap_count(records):
    grid = np.zeros((7, 24), dtype=int)
    for r in records:
        grid[r.get("weekday", 0)][r.get("hour", 0)] += 1
    return grid


def heatmap_intensity(records):
    s = np.zeros((7, 24))
    n = np.zeros((7, 24), dtype=int)
    for r in records:
        w, h = r.get("weekday", 0), r.get("hour", 0)
        s[w][h] += abs(r.get("net_lines", 0))
        n[w][h] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, s / n, 0)


def monthly_volume(records):
    counts = defaultdict(int)
    by_domain = defaultdict(lambda: defaultdict(int))
    for r in records:
        dt  = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        dom = classify_domain(r.get("files_changed", []))
        counts[key] += 1
        by_domain[dom][key] += 1
    months = sorted(counts)
    return months, counts, by_domain


def run():
    print(f"Loading {INPUT} ...")
    records = load()
    print(f"  {len(records)} records after noise filter.")

    count_grid     = heatmap_count(records)
    intensity_grid = heatmap_intensity(records)
    months, monthly, by_domain = monthly_volume(records)

    DOMAIN_COLORS = {
        "creative":   "#e07b54",
        "job-search": "#4a90d9",
        "legal":      "#9b59b6",
        "plot-ark":   "#2ecc71",
        "ai-tools":   "#f1c40f",
        "learning":   "#1abc9c",
        "career-ops": "#e74c3c",
        "general":    "#7f8c8d",
    }

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "Commit frequency — hour × weekday",
            "Output intensity — avg lines changed",
            "Monthly commit volume (total)",
            "Monthly commit volume by domain",
        ],
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.2, 0.3],
    )

    # Row 1 — frequency heatmap
    fig.add_trace(go.Heatmap(
        z=count_grid,
        x=list(range(24)),
        y=DAYS,
        colorscale="Blues",
        colorbar=dict(x=1.02, len=0.22, y=0.88, title="commits"),
    ), row=1, col=1)

    # Row 2 — intensity heatmap
    fig.add_trace(go.Heatmap(
        z=intensity_grid,
        x=list(range(24)),
        y=DAYS,
        colorscale="YlOrRd",
        colorbar=dict(x=1.02, len=0.22, y=0.62, title="avg lines"),
    ), row=2, col=1)

    # Row 3 — total monthly bar
    fig.add_trace(go.Bar(
        x=months,
        y=[monthly[m] for m in months],
        marker_color="#4a6fa5",
        name="total",
    ), row=3, col=1)

    # Row 4 — stacked domain bars
    for dom, color in DOMAIN_COLORS.items():
        vals = [by_domain[dom].get(m, 0) for m in months]
        if any(v > 0 for v in vals):
            fig.add_trace(go.Bar(
                x=months,
                y=vals,
                name=dom,
                marker_color=color,
            ), row=4, col=1)

    fig.update_xaxes(tickvals=list(range(0, 24, 2)), title_text="Hour of day", row=1, col=1)
    fig.update_xaxes(tickvals=list(range(0, 24, 2)), title_text="Hour of day", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_xaxes(title_text="Month", row=4, col=1)
    fig.update_yaxes(title_text="Day",     row=1, col=1)
    fig.update_yaxes(title_text="Day",     row=2, col=1)
    fig.update_yaxes(title_text="Commits", row=3, col=1)
    fig.update_yaxes(title_text="Commits", row=4, col=1)

    fig.update_layout(
        barmode="stack",
        title="Cogito — Direction B: Behavioral Patterns",
        height=1300,
        font=dict(family="monospace", size=12),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#c9d1d9",
        legend=dict(x=1.03, y=0.18),
    )

    fig.write_html(str(OUTPUT))
    print(f"  Report -> {OUTPUT}")
    webbrowser.open(OUTPUT.as_uri())


if __name__ == "__main__":
    run()
