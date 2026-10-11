"""Generate docs/architecture.png: ingestion -> NLP -> event resolution
-> impact scoring -> exposure mapping -> stress testing -> dashboard.
Run: python scripts/generate_architecture_diagram.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

STAGES = [
    ("GDELT + SEC 8-K\n(+ synthetic fixture,\nexplicit opt-in)", "#1f6f8b"),
    ("Entity resolution\n(ticker/alias\nmatch)", "#2a9d8f"),
    ("Sentiment (FinBERT)\n+ Event classifier\n(rules + zero-shot)", "#2a9d8f"),
    ("Event resolution\n(entity+type+time+\ntext similarity)", "#e9c46a"),
    ("Impact engine\n(severity, relevance,\nnovelty, velocity,\ncorrob.)", "#e76f51"),
    ("Exposure mapping\n(direct / sector /\ncountry / systemic)", "#f4a261"),
    ("Portfolio stress\n(bond/loan/equity/\nswap/FX)", "#e63946"),
    ("Dashboard\n(Streamlit\n+ Plotly)", "#6d597a"),
]

fig, ax = plt.subplots(figsize=(15.5, 4.6), dpi=160)
ax.set_xlim(0, len(STAGES))
ax.set_ylim(0, 1.6)
ax.axis("off")

box_w, box_h = 0.90, 1.0
for i, (label, color) in enumerate(STAGES):
    x = i + 0.5
    box = FancyBboxPatch(
        (x - box_w / 2, 0.3), box_w, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor="#333333", facecolor=color, alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(x, 0.8, label, ha="center", va="center", fontsize=7.6, color="white", weight="bold", linespacing=1.4)
    if i < len(STAGES) - 1:
        arrow = FancyArrowPatch(
            (x + box_w / 2, 0.8), (x + 1 - box_w / 2, 0.8),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#333333",
        )
        ax.add_patch(arrow)

ax.text(
    len(STAGES) / 2, 1.42,
    "Event-Driven Financial Risk Intelligence -> Portfolio Stress Testing (Module B)",
    ha="center", va="center", fontsize=13, weight="bold",
)
ax.text(
    len(STAGES) / 2, 0.14,
    "Signal contract (JSONL) connects every stage; FastAPI serves it read-only; replay mode makes the whole chain deterministic/offline.",
    ha="center", va="center", fontsize=8.5, style="italic", color="#444444",
)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")
