#!/usr/bin/env python3
"""Disambiguate the d^-0.6 MSE-vs-d slope: under-training artifact, or real (structured codes)?

Re-measure MSE vs d at s=3 for a SHORT and a LONG training budget. If the slope steepens toward
-1 with more steps -> the softness was under-training at large d. If it stays ~-0.6 -> it's a real
effect (SGD's structured/low-coherence codes scale differently from the random-code prediction).

Run:  uv run --with torch --with numpy src/conv_check.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaling  # noqa: E402

DS = [20, 50, 100, 150]
for steps in (3000, 10000):
    ys = [scaling.train_mse(300, d, 3.0, steps=steps) for d in DS]
    sl = float(np.polyfit(np.log(DS), np.log(ys), 1)[0])
    print(f"steps={steps:6}  slope={sl:+.2f}   " + "  ".join(f"d{d}:{y:.4f}" for d, y in zip(DS, ys)), flush=True)
print("\nread: slope -> -1 with more steps = under-training; stays ~-0.6 = real (structured codes)")
