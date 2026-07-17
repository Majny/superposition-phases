#!/usr/bin/env python3
"""
Solution classifier: given a trained compressed-computation model, which of the three regimes
did SGD land in?

  1. DEDICATED       — each neuron computes ~one feature; only ~d features handled. Boring.
  2. DENSE-CC        — neurons participate in many features with unstructured, dense weights.
  3. SUPERPOSED-CODE — features sit on a structured, near-orthogonal code (low mutual coherence).

We read this off the two weight matrices (no extra training):
  W_in  (d, n): column i is the direction feature i is *written into* the hidden layer.
  W_out (n, d): row i    is the direction feature i is *read out* of the hidden layer.

Diagnostics printed (interpret these yourself — that's the point):
  - feats_per_neuron : avg # features each neuron meaningfully participates in. ~1 => dedicated.
  - n_used           : # features the model actually uses (in & out norm above noise). > d => superposition.
  - coherence        : max |cos angle| between feature input-directions. Low => structured code;
                       compare to the Welch bound sqrt((n-d)/(d(n-1))), the best achievable.
  - decode_r2        : R^2 of linearly recovering x from the hidden layer (pseudoinverse) — how
                       cleanly the features are still linearly accessible after packing.

Run:  uv run --with torch --with numpy src/classify.py --tag n100_d50_k3.0_l2_s0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def make_batch(n, k, batch, gen):
    p = k / n
    mask = (torch.rand(batch, n, generator=gen) < p).float()
    return mask * (torch.rand(batch, n, generator=gen) * 2 - 1)


def classify(tag: str, part_thresh: float = 0.3, use_thresh: float = 0.1) -> dict:
    ckpt = torch.load(ROOT / "results" / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    n, d, k = cfg["n"], cfg["d"], float(cfg["k"])
    W_in = ckpt["state_dict"]["w_in.weight"]    # (d, n)
    W_out = ckpt["state_dict"]["w_out.weight"]  # (n, d)

    # per-feature in/out directions and their norms
    U = W_in.t()                                 # (n, d): U[i] = feature i input direction
    V = W_out                                    # (n, d): V[i] = feature i output direction
    in_norm, out_norm = U.norm(dim=1), V.norm(dim=1)
    used = (in_norm > use_thresh * in_norm.max()) & (out_norm > use_thresh * out_norm.max())
    n_used = int(used.sum())

    # participation: per neuron, how many features it reads with |w| > thresh * its own max
    row_max = W_in.abs().max(dim=1, keepdim=True).values.clamp_min(1e-9)
    participation = (W_in.abs() > part_thresh * row_max).float().sum(dim=1)  # (d,)
    feats_per_neuron = float(participation.mean())

    # mutual coherence of the USED feature input-directions (structure of the code)
    Uu = torch.nn.functional.normalize(U[used], dim=1)
    G = (Uu @ Uu.t()).abs()
    G.fill_diagonal_(0.0)
    coherence = float(G.max()) if n_used > 1 else float("nan")
    welch = math.sqrt((n - d) / (d * (n - 1))) if n > d else float("nan")

    # pseudoinverse decoding: can we linearly recover x from h = ReLU(W_in x)?
    gen = torch.Generator().manual_seed(1234)
    x = make_batch(n, k, 20000, gen)
    h = torch.relu(torch.nn.functional.linear(x, W_in, ckpt["state_dict"].get("w_in.bias")))
    D, *_ = torch.linalg.lstsq(h, x)             # h @ D ~ x
    x_hat = h @ D
    ss_res = ((x - x_hat) ** 2).sum()
    ss_tot = ((x - x.mean(0)) ** 2).sum()
    decode_r2 = float(1 - ss_res / ss_tot)

    # heuristic label (v0 thresholds — refine against known cases)
    if n_used <= 1.2 * d and feats_per_neuron < 1.5:
        label = "dedicated"
    elif feats_per_neuron >= 1.5 and coherence < 0.5:
        label = "superposed-code"
    else:
        label = "dense-CC"

    # pull the trained-vs-baseline loss recorded at training time, if present
    jp = ROOT / "results" / "checkpoints" / f"{tag}.json"
    train_rec = json.loads(jp.read_text()) if jp.exists() else {}

    return {"tag": tag, "n": n, "d": d, "k": k, "loss": cfg["loss"],
            "n_used": n_used, "feats_per_neuron": round(feats_per_neuron, 2),
            "coherence": round(coherence, 3), "welch_bound": round(welch, 3),
            "decode_r2": round(decode_r2, 3),
            "trained_loss": round(train_rec.get("trained_loss", float("nan")), 5),
            "beats_baseline": train_rec.get("beats_baseline"), "label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="checkpoint tag, e.g. n100_d50_k3.0_l2_s0")
    args = ap.parse_args()
    r = classify(args.tag)
    print(json.dumps(r, indent=2))
    print(f"\n  => {r['label'].upper()}")
    print(f"     n_used={r['n_used']} (d={r['d']}; >d means superposition)   "
          f"feats/neuron={r['feats_per_neuron']} (~1 = dedicated)")
    print(f"     coherence={r['coherence']} vs Welch bound {r['welch_bound']} (closer = more structured code)   "
          f"linear-decode R^2={r['decode_r2']}")


if __name__ == "__main__":
    main()
