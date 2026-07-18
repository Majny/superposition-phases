#!/usr/bin/env python3
"""
Test the self-derived capacity scaling for unary-ReLU computation-in-superposition.

Derived law (scaling conjecture; rigorous achievability half via compressed-sensing/RIP):
    interference variance sigma^2 ~= s/(3d)  ->  average per-feature MSE ~= s/(6d),  N-INDEPENDENT,
    and (all-features-reliable) n_max ~= s * exp(C * eps * d / s), exponential in d/s, >> d^2/log d.
Two sharp, cheap, decisive predictions:

  P1 (UNARY SIGNATURE, decisive): at fixed active-count s and fixed d, average per-feature MSE is
     FLAT in n. A pairwise/interacting task would grow with n. If flat -> interference is n-free.
  P2 (COMPUTATION IS FREE): the ReLU target (compute) behaves like the identity target (store) —
     same MSE level and same capacity slope. Computation adds no capacity cost for unary functions.
  P3 (EXPONENTIAL): under an all-features-reliable criterion (>=95% of features within eps),
     log(n_max) is LINEAR in d (not sub-linear = polynomial, not slope-2 = pairwise d^2).

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


def train_eval(n, d, s, loss, target, seed, steps=3000):
    """target: 'relu' -> y=ReLU(x) (compute); 'identity' -> y=x (store, control).
    Returns (mean per-feature MSE, fraction of features with normalized error > 0.1)."""
    exponent = 2 if loss == "l2" else 4
    tgt = (lambda x: torch.relu(x)) if target == "relu" else (lambda x: x)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = toy_cis.CIS(n, d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        x = toy_cis.make_batch(n, s, 1024, gen)
        loss_v = ((model(x) - tgt(x)).abs() ** exponent).mean()
        opt.zero_grad(); loss_v.backward(); opt.step()
    with torch.no_grad():
        x = toy_cis.make_batch(n, s, 20000, gen)
        y = tgt(x)
        mse = ((model(x) - y) ** 2).mean(dim=0)                 # (n,)
        power = (y ** 2).mean(dim=0).clamp_min(1e-9)
        rel = mse / power
    return float(mse.mean()), float((rel > 0.1).float().mean())


def n_max_reliable(d, s, loss, target, seed, bad_frac=0.05, n_hi=2000):
    """Largest n with < bad_frac of features exceeding the error tolerance (all-features-reliable)."""
    lo, hi = d, n_hi
    if train_eval(hi, d, s, loss, target, seed)[1] < bad_frac:
        return hi
    while hi - lo > max(2, d // 5):
        mid = (lo + hi) // 2
        if train_eval(mid, d, s, loss, target, seed)[1] < bad_frac:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    S, SEED = 3.0, 0
    out = {}

    # --- P1 + P2: MSE vs n at fixed (s, d), for relu vs identity ---
    d0, Ns = 50, [50, 100, 200, 400, 800, 1500]
    print(f"P1/P2  MSE vs n  (fixed s={S}, d={d0}, L4)  — predict FLAT and relu≈identity\n")
    flat = {}
    for target in ("relu", "identity"):
        ys = [train_eval(n, d0, S, "l4", target, SEED)[0] for n in Ns]
        flat[target] = dict(zip(Ns, ys))
        print(f"  {target:8}  " + "  ".join(f"n={n}:{m:.4f}" for n, m in zip(Ns, ys)), flush=True)
    out["flatness"] = {"d": d0, "s": S, "Ns": Ns, **flat}

    # --- P3: n_max vs d (all-features-reliable), relu vs identity, L4 ---
    DS = [10, 20, 30, 50, 75]
    print(f"\nP3  n_max vs d  (>=95% features within 10% rel err, s={S}, L4)  — predict log-linear\n")
    cap = {}
    for target in ("relu", "identity"):
        row = [n_max_reliable(d, S, "l4", target, SEED) for d in DS]
        cap[target] = dict(zip(DS, row))
        print(f"  {target:8}  " + "  ".join(f"d={d}:{nm}" for d, nm in zip(DS, row)), flush=True)
    out["capacity"] = {"DS": DS, "s": S, **cap}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "capacity.json").write_text(json.dumps(out, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
        for t, mk in (("relu", "o-"), ("identity", "s--")):
            a1.plot(Ns, [flat[t][n] for n in Ns], mk, label=f"{t} target")
        a1.set_xscale("log"); a1.set_xlabel("features  n  (fixed s, d)"); a1.set_ylabel("mean per-feature MSE")
        a1.set_title("P1/P2: MSE flat in n, compute≈store"); a1.legend(); a1.grid(alpha=0.3)
        for t, mk in (("relu", "o-"), ("identity", "s--")):
            a2.plot(DS, [cap[t][d] for d in DS], mk, label=f"{t} target")
        a2.set_yscale("log"); a2.set_xlabel("hidden neurons  d"); a2.set_ylabel("n_max (all-features-reliable)")
        a2.set_title("P3: capacity ~ exp(d) (log-linear)"); a2.legend(); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / "results" / "capacity.png", dpi=130)
        print("\nwrote results/capacity.png + results/capacity.json")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
