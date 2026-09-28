#!/usr/bin/env python3
"""Render the matched-budget endpoint comparison as a dependency-free SVG."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
lookup = json.loads((ROOT / "results/career_extension_lm_summary.json").read_text())["runs"]
values = [lookup["long1b_baseline"]["val_loss"], lookup["long1b_sweep_selected"]["val_loss"]]
labels = ["Baseline", "Sweep-selected"]
colors = ["#2563eb", "#f97316"]
parts = []
for i, (label, value, color) in enumerate(zip(labels, values, colors)):
    width = value / 5.2 * 470
    y = 130 + i * 100
    parts += [f'<rect x="180" y="{y}" width="{width:.1f}" height="55" rx="8" fill="{color}"/>',
              f'<text x="165" y="{y+35}" text-anchor="end" font-size="18">{label}</text>',
              f'<text x="{190+width:.1f}" y="{y+35}" font-size="18" font-weight="700">{value:.6f}</text>']
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="360"><rect width="760" height="360" fill="#f8fafc"/>
<text x="40" y="48" font-size="25" font-weight="700">1.012B matched 16.384M-token endpoint</text>
<text x="40" y="78" font-size="16" fill="#475569">Lower validation loss is better; short-sweep winner did not win at the long endpoint.</text>{''.join(parts)}</svg>'''
(ROOT / "plots").mkdir(exist_ok=True)
(ROOT / "plots/long_endpoint.svg").write_text(svg)
