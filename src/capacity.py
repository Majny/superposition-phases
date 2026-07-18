#!/usr/bin/env python3
"""
Empirical achieved-capacity of unary-ReLU computation-in-superposition.

For a hidden layer of d neurons, at fixed sparsity s, how many features n can the model compute
(y_i = ReLU(x_i)) to accuracy eps? We binary-search the largest such n = n_max(d), for L2 vs L4,
and plot n_max vs d. This is THE figure — the empirical trajectory the (self-derived) capacity
scaling law must match. It is a guaranteed result independent of whether the theory is exactly right.

Accuracy criterion: average over features of (per-feature MSE / per-feature signal power E[y_i^2])
below eps — i.e. the model explains >= (1-eps) of each feature's signal, on average. Success means
the d neurons genuinely compute n > d ReLUs in superposition.

Run (CPU, minutes):  uv run --with torch --with numpy --with matplotlib src/capacity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_cis  # noqa: E402


def rel_error(n: int, d: int, s: float, loss: str, seed: int, steps: int = 5000) -> float:
    """Train at (n,d,s,loss); return mean over features of per-feature MSE / signal power."""
    exponent = 2 if loss == "l2" else 4
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = toy_cis.CIS(n, d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        x = toy_cis.make_batch(n, s, 1024, gen)
        y = torch.relu(x)
        loss_v = ((model(x) - y).abs() ** exponent).mean()
        opt.zero_grad(); loss_v.backward(); opt.step()
    with torch.no_grad():
        x = toy_cis.make_batch(n, s, 20000, gen)
        y = torch.relu(x)
        mse = ((model(x) - y) ** 2).mean(dim=0)          # (n,)
        power = (y ** 2).mean(dim=0).clamp_min(1e-9)     # (n,)
    return float((mse / power).mean())


def n_max_for_d(d: int, s: float, loss: str, seed: int, eps: float, n_hi: int) -> int:
    """Largest n (>= d) whose trained model still hits mean relative error < eps. Binary search."""
    lo, hi = d, n_hi
    if rel_error(lo, d, s, loss, seed) >= eps:
        return lo  # can't even do d features to eps here
    if rel_error(hi, d, s, loss, seed) < eps:
        return hi  # censored: capacity >= n_hi
    while hi - lo > max(2, d // 4):
        mid = (lo + hi) // 2
        if rel_error(mid, d, s, loss, seed) < eps:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    S = 3.0        # active features per example (fixed sparsity)
    EPS = 0.1      # explain >= 90% of each feature's signal, on average
    SEEDS = [0]   # first pass: one seed for the curve shape; add seeds for error bars once it looks right
    DS = [10, 20, 30, 50, 75, 100]
    N_HI = 1500
    out = {"s": S, "eps": EPS, "n_hi": N_HI, "results": {}}
    print(f"achieved capacity n_max(d): s={S} eps={EPS} seeds={SEEDS}\n")
    for loss in ("l2", "l4"):
        row = []
        for d in DS:
            vals = [n_max_for_d(d, S, loss, seed, EPS, N_HI) for seed in SEEDS]
            nmax = sorted(vals)[len(vals) // 2]           # median
            row.append(nmax)
            cens = " (>=n_hi, censored)" if nmax >= N_HI else ""
            print(f"  {loss}  d={d:3}  n_max={nmax:5}  (n_max/d={nmax / d:.1f}){cens}", flush=True)
        out["results"][loss] = dict(zip(DS, row))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "capacity.json").write_text(json.dumps(out, indent=2))

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for loss, mk in (("l2", "o-"), ("l4", "s-")):
            xs = DS
            ys = [out["results"][loss][d] for d in DS]
            ax.plot(xs, ys, mk, label=f"learned ({loss.upper()})")
        ax.plot(DS, [d for d in DS], "k--", alpha=0.5, label="dedicated (n=d)")
        ax.set_yscale("log")
        ax.set_xlabel("hidden neurons  d")
        ax.set_ylabel(f"achieved capacity  n_max  (mean rel. err < {EPS})")
        ax.set_title(f"Unary-ReLU compressed computation: achieved capacity (s={S:.0f})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "capacity.png", dpi=130)
        print(f"\nwrote results/capacity.png + results/capacity.json")
    except Exception as e:
        print(f"\n(plot skipped: {e}) — data in results/capacity.json")


if __name__ == "__main__":
    main()
