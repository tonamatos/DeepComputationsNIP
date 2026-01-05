"""
Reflow LaTeX text to one sentence per line for cleaner Git diffs.
Optional utility; does not affect compilation.

Copy-pasting from Overleaf or other editors causes wrapped line breaks
which git dislikes.

- Operates on ComplexityViaTopology.tex at repo root
- Preserves paragraph breaks (blank lines)
- Leaves comment-only paragraphs unchanged
- Formatting-only change; no semantic LaTeX edits intended

Email me (Tona) if you have any issues.
"""

import re
from pathlib import Path

ROOT_TEX = Path(__file__).resolve().parents[1] / "ComplexityViaTopology.tex"

text = ROOT_TEX.read_text(encoding="utf-8")

def reflow_paragraph(p: str) -> str:
    p = re.sub(r"\s+", " ", p.strip()) # Normalize internal whitespace

    # Insert newline after sentence-ending punctuation
    p = re.sub(r"([.!?])\s+", r"\1\n", p)

    return p

# Split on blank lines (paragraph boundaries
paragraphs = re.split(r"\n\s*\n", text)

out = []
for p in paragraphs:
    stripped = p.lstrip()
    if stripped.startswith("%") or stripped == "":
        out.append(p.strip()) # Leave comment-only or empty paragraphs untouched
    else:
        out.append(reflow_paragraph(p))

reflowed = "\n\n".join(out) + "\n"

ROOT_TEX.write_text(reflowed, encoding="utf-8")
