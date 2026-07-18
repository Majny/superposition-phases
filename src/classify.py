#!/usr/bin/env python3
"""
Solution classifier: which regime did SGD land in? — DEDICATED / DENSE-CC / SUPERPOSED-CODE.

The rule below was derived + adversarially vetted (3 independent proposals + a synthesis) to be a
UNIFIER that recovers BOTH debate papers' verdicts (skeptic 2606.14673 sparsity bifurcation; da
Silva & Heimersheim 2607.04800 L2-vs-L4), grounded in the regime definitions rather than fit to a
handful of runs. Valid for the OVERCOMPLETE regime n > d.

Three load-bearing diagnostics (+ beats_baseline as an inert guard):
  coherence            max |cos| between feature input-directions. Low = structured code heading to
                       the Welch bound sqrt((n-d)/(d(n-1))); high (0.8-1.0) = unstructured. (Only
                       separates SUPERPOSED from the rest — dedicated & dense are both high.)
  cv_mse               coefficient of variation of per-feature MSE (the rebuttal's own metric).
                       Low = error spread evenly (codeword regime); high = uneven.
  feats_per_neuron     avg #features a neuron participates in. ~1 = DEDICATED (each neuron ~one feature).

Dropped: decode_r2 (flat ~0.26 across all regimes — a nonlinear ReLU readout makes linear decoding
blind here). Demoted: n_used (never gate on it — the old n_used<=1.2d test is exactly what hid the
skeptic bifurcation; kept only as a corroborating log).

Run:  uv run --with torch --with numpy src/classify.py --tag n100_d50_k2.0_l4_s0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_cis  # noqa: E402

CKPTS = ROOT / "results" / "checkpoints"


def classify(tag: str, part_thresh: float = 0.3, use_thresh: float = 0.1) -> dict:
    ck = torch.load(CKPTS / f"{tag}.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    n, d, k = cfg["n"], cfg["d"], float(cfg["k"])
    W_in = ck["state_dict"]["w_in.weight"]    # (d, n)
    W_out = ck["state_dict"]["w_out.weight"]  # (n, d)

    U = W_in.t()                              # (n, d): feature input directions
    V = W_out                                 # (n, d): feature output directions
    used = (U.norm(dim=1) > use_thresh * U.norm(dim=1).max()) & \
           (V.norm(dim=1) > use_thresh * V.norm(dim=1).max())
    n_used = int(used.sum())

    row_max = W_in.abs().max(dim=1, keepdim=True).values.clamp_min(1e-9)
    feats_per_neuron = float((W_in.abs() > part_thresh * row_max).float().sum(dim=1).mean())

    Uu = torch.nn.functional.normalize(U[used], dim=1)
    G = (Uu @ Uu.t()).abs()
    G.fill_diagonal_(0.0)
    coherence = float(G.max()) if n_used > 1 else float("nan")
    welch = math.sqrt((n - d) / (d * (n - 1))) if n > d else float("nan")

    # CV of per-feature MSE (rebuttal's own signature) — needs a forward pass
    model = toy_cis.CIS(n, d)
    model.load_state_dict(ck["state_dict"])
    gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x = toy_cis.make_batch(n, k, 20000, gen)
        per_feat = ((model(x) - torch.relu(x)) ** 2).mean(dim=0)
    cv_mse = float(per_feat.std() / per_feat.mean().clamp_min(1e-12))

    jp = CKPTS / f"{tag}.json"
    beats = json.loads(jp.read_text()).get("beats_baseline") if jp.exists() else None

    # --- the vetted rule (overcomplete n>d), first match wins ---
    if coherence < 0.5 and cv_mse < 0.15 and beats:
        label = "superposed-code"
    elif feats_per_neuron < 1.5:
        label = "dedicated"
    else:
        label = "dense-CC"

    return {"tag": tag, "n": n, "d": d, "k": k, "loss": cfg["loss"], "label": label,
            "coherence": round(coherence, 3), "cv_mse": round(cv_mse, 3),
            "feats_per_neuron": round(feats_per_neuron, 2), "beats_baseline": beats,
            "welch_bound": round(welch, 3), "n_used_log": n_used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    r = classify(args.tag)
    print(json.dumps(r, indent=2))
    print(f"\n  => {r['label'].upper()}")
    print(f"     coherence={r['coherence']} (Welch {r['welch_bound']}; <0.5 = structured code)   "
          f"cv_mse={r['cv_mse']} (<0.15 = even error)   feats/neuron={r['feats_per_neuron']} (~1 = dedicated)   "
          f"beats={r['beats_baseline']}")


if __name__ == "__main__":
    main()
