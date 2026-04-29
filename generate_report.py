"""
generate_report.py -- Build a data-driven TMaze sweep PDF from run artifacts.
"""

import os
import re
import datetime
from statistics import mean

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER

RESULTS_DIR = "/glade/derecho/scratch/adadelek/results/tmaze_full"
OUT = os.path.join(RESULTS_DIR, "tmaze_sweep_report.pdf")

PAT_NONE = re.compile(r"^none_tmaze_seed(?P<seed>\d+)$")
PAT_CONCEPT = re.compile(
    r"^(?P<concept_net>cbm|concept_ac)_(?P<temporal>none|stacked|gru)_(?P<supervision>online|none|queried)_(?P<freeze>frozen|coupled)_tmaze_seed(?P<seed>\d+)$"
)


def parse_run(name: str):
    m = PAT_NONE.match(name)
    if m:
        return dict(
            name=name, concept_net="none", temporal="none",
            supervision="none", freeze="none", seed=int(m.group("seed"))
        )
    m = PAT_CONCEPT.match(name)
    if m:
        d = m.groupdict()
        d["seed"] = int(d["seed"])
        d["name"] = name
        return d
    return None


def load_runs():
    rows = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        run_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        meta = parse_run(name)
        if meta is None:
            continue
        eval_path = os.path.join(run_dir, "eval.txt")
        if not os.path.exists(eval_path):
            continue
        m, s = None, None
        with open(eval_path) as f:
            for line in f:
                if line.startswith("mean_reward="):
                    m = float(line.split("=")[1].strip())
                if line.startswith("std_reward="):
                    s = float(line.split("=")[1].strip())
        cmean = cue = aj = None
        cap = os.path.join(run_dir, "concept_acc.npz")
        if os.path.exists(cap):
            z = np.load(cap, allow_pickle=True)
            names = [str(x) for x in z["names"]]
            vals = z["values"][-1]
            cmean = float(vals.mean())
            if "cue" in names:
                cue = float(vals[names.index("cue")])
            if "at_junction" in names:
                aj = float(vals[names.index("at_junction")])
        rows.append({**meta, "mean_reward": m, "std_reward": s, "cmean": cmean, "cue": cue, "at_junction": aj})
    rows.sort(key=lambda x: x["mean_reward"], reverse=True)
    return rows


def fmt(v, nd=3):
    if v is None:
        return "--"
    return f"{v:.{nd}f}"


def make_table(data, widths):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build():
    runs = load_runs()
    if not runs:
        raise RuntimeError(f"No TMaze runs found in {RESULTS_DIR}")

    styles = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=styles["Title"], alignment=TA_CENTER, fontSize=18)
    subtitle = ParagraphStyle("S", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10)
    h = styles["Heading2"]
    body = styles["BodyText"]

    doc = SimpleDocTemplate(OUT, pagesize=LETTER, leftMargin=0.8 * inch, rightMargin=0.8 * inch)
    story = []
    story += [
        Paragraph("TMaze Sweep Report (Data-Driven)", title),
        Paragraph(f"Generated {datetime.date.today().isoformat()} from {RESULTS_DIR}", subtitle),
        Spacer(1, 10),
    ]

    top = runs[0]
    story += [
        Paragraph("Headline", h),
        Paragraph(
            f"Best run: <b>{top['name']}</b> with mean reward <b>{fmt(top['mean_reward'])}</b>.",
            body,
        ),
        Spacer(1, 8),
    ]

    full = [["Rank", "Run", "Reward", "Std", "ConceptAcc", "CueAcc", "AtJunctionAcc"]]
    for i, r in enumerate(runs, 1):
        full.append([
            str(i), r["name"], fmt(r["mean_reward"]), fmt(r["std_reward"]),
            fmt(r["cmean"]), fmt(r["cue"]), fmt(r["at_junction"])
        ])
    story += [Paragraph("All Runs Ranked", h), make_table(full, [0.5 * inch, 3.9 * inch, 0.8 * inch, 0.6 * inch, 0.9 * inch, 0.8 * inch, 1.0 * inch]), Spacer(1, 10)]

    groups = {}
    for r in runs:
        groups.setdefault(r["concept_net"], []).append(r["mean_reward"])
    group_rows = [["Group", "Best Reward", "Mean Reward", "Num Runs"]]
    for g in ["none", "cbm", "concept_ac"]:
        vals = groups.get(g, [])
        if vals:
            group_rows.append([g, fmt(max(vals)), fmt(mean(vals)), str(len(vals))])
    story += [Paragraph("Family Summary", h), make_table(group_rows, [1.2 * inch, 1.1 * inch, 1.1 * inch, 1.0 * inch])]

    doc.build(story)
    print(f"PDF saved -> {OUT}")


if __name__ == "__main__":
    build()
