#!/usr/bin/env python3
"""
Test the UNAMBIGUOUS scaling predictions of the derived interference law (avoids the metric-
dependent n_max). Derived: interference variance sigma^2 ~= s/(3d)  ->  avg per-feature MSE ~= s/(6d).

  T1  MSE vs d  (fixed s, n): predict log-log slope  -1   (MSE ∝ 1/d)
  T2  MSE vs s  (fixed d, n): predict log-log slope  +1   (MSE ∝ s)
  T3  s_max vs d (max co-active s with MSE < eps, fixed n): predict LINEAR through origin,
                 s_max ~= 6*eps*d  — the honest capacity quantity (co-active load), unlike n_max
                 which is n-independent under this criterion.

n_max is deliberately NOT used: at fixed active-count s the interference is n-independent, so the
feature count is not interference-bounded (only the co-active load s_max is). That is itself the
result — unary CiS capacity is set by s_max ∝ eps*d, sharply unlike pairwise tasks (d^2/log d).

Run (CPU, minutes):  uv run --with torch --with numpy --with matplotlib src/scaling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_cis  # noqa: E402


def train_mse(n, d, s, seed=0, loss="l4", steps=3000):
    exponent = 2 if loss == "l2" else 4
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = toy_cis.CIS(n, d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        x = toy_cis.make_batch(n, s, 1024, gen)
        loss_v = ((model(x) - torch.relu(x)).abs() ** exponent).mean()
        opt.zero_grad(); loss_v.backward(); opt.step()
    with torch.no_grad():
        x = toy_cis.make_batch(n, s, 20000, gen)
        return float(((model(x) - torch.relu(x)) ** 2).mean())


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    N = 300
    out = {"n": N}

    # T1: MSE vs d, fixed s=3
    DS = [20, 30, 50, 75, 100, 150]
    t1 = {d: train_mse(N, d, 3.0) for d in DS}
    s1 = slope(DS, [t1[d] for d in DS])
    print(f"T1  MSE vs d (s=3): slope={s1:+.2f}  (predict -1)   " +
          "  ".join(f"d{d}:{t1[d]:.4f}" for d in DS), flush=True)

    # T2: MSE vs s, fixed d=50
    SS = [1.0, 2.0, 3.0, 5.0, 8.0]
    t2 = {s: train_mse(N, 50, s) for s in SS}
    s2 = slope(SS, [t2[s] for s in SS])
    print(f"T2  MSE vs s (d=50): slope={s2:+.2f}  (predict +1)   " +
          "  ".join(f"s{s:.0f}:{t2[s]:.4f}" for s in SS), flush=True)

    # T3: s_max vs d at MSE tolerance eps
    EPS = 0.02
    t3 = {}
    for d in DS:
        smax = 0.0
        for s in np.arange(0.5, 0.6 * d, 0.5):        # scan co-active count upward
            if train_mse(N, d, float(s)) < EPS:
                smax = float(s)
            else:
                break
        t3[d] = smax
    s3 = slope([d for d in DS if t3[d] > 0], [t3[d] for d in DS if t3[d] > 0]) if any(t3.values()) else float("nan")
    print(f"T3  s_max vs d (eps={EPS}): log-log slope={s3:+.2f}  (predict +1, linear)   " +
          "  ".join(f"d{d}:{t3[d]:.1f}" for d in DS), flush=True)

    out.update({"T1_MSE_vs_d": t1, "T1_slope": s1, "T2_MSE_vs_s": t2, "T2_slope": s2,
                "T3_smax_vs_d": t3, "T3_slope": s3, "eps": EPS})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "scaling.json").write_text(json.dumps(out, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        ax[0].loglog(DS, [t1[d] for d in DS], "o-"); ax[0].loglog(DS, [3 / (6 * d) for d in DS], "k--", alpha=.6, label="s/6d")
        ax[0].set_xlabel("d"); ax[0].set_ylabel("MSE"); ax[0].set_title(f"T1 MSE∝1/d (slope {s1:+.2f})"); ax[0].legend(); ax[0].grid(alpha=.3, which="both")
        ax[1].loglog(SS, [t2[s] for s in SS], "o-"); ax[1].loglog(SS, [s / (6 * 50) for s in SS], "k--", alpha=.6, label="s/6d")
        ax[1].set_xlabel("s"); ax[1].set_ylabel("MSE"); ax[1].set_title(f"T2 MSE∝s (slope {s2:+.2f})"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")
        ax[2].plot(DS, [t3[d] for d in DS], "o-"); ax[2].plot(DS, [6 * EPS * d for d in DS], "k--", alpha=.6, label="6εd")
        ax[2].set_xlabel("d"); ax[2].set_ylabel("s_max"); ax[2].set_title("T3 s_max∝εd (capacity)"); ax[2].legend(); ax[2].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(ROOT / "results" / "scaling.png", dpi=130)
        print("\nwrote results/scaling.png + results/scaling.json")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
