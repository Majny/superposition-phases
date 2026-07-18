#!/usr/bin/env python3
"""
Validate the 3-class classifier by reproducing the two debate papers' headline verdicts.

The classifier is only useful if it's a *unifier* — one instrument that recovers what each camp
found with its own incompatible metric. So we reproduce their configs and check agreement.

We also compute each camp's OWN signature so the agreement is explicit:
  - da Silva & Heimersheim (2607.04800) diagnostic = CV of per-feature MSE.
    Their finding: L2 -> naive/dense solution (CV ~ 1, a few features handled, rest ignored);
    exponent > 2 -> genuine superposition (CV ~ 0.03-0.05, error spread evenly).
  - skeptic (2606.14673): sparsity sweep under L2 bifurcates into a dense-regime and a
    sparse-regime solution. We sweep sparsity and watch the label/CV move.

PASS = our {dedicated/dense-CC/superposed-code} label tracks their verdict (L2->not-superposed,
L4->superposed; and a visible bifurcation across the sparsity sweep). Then it's a credible shared
instrument, not a 5th metric.

Run:  uv run --with torch --with numpy src/reproduce.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import toy_cis          # noqa: E402
import classify as clf  # noqa: E402


def cfg(**kw) -> SimpleNamespace:
    base = dict(n=100, d=50, k=2.0, loss="l2", steps=4000, batch=2048, lr=1e-3, seed=0, verbose=False)
    base.update(kw)
    base["exponent"] = 2 if base["loss"] == "l2" else 4
    return SimpleNamespace(**base)


def per_feature_mse_cv(tag: str) -> float:
    """da Silva & Heimersheim signature: coefficient of variation of per-feature MSE."""
    ck = torch.load(SRC.parent / "results" / "checkpoints" / f"{tag}.pt", map_location="cpu", weights_only=False)
    c = ck["config"]
    model = toy_cis.CIS(c["n"], c["d"])
    model.load_state_dict(ck["state_dict"])
    gen = torch.Generator().manual_seed(999)
    with torch.no_grad():
        x = toy_cis.make_batch(c["n"], float(c["k"]), 20000, gen)
        y = torch.relu(x)
        per_feat = ((model(x) - y) ** 2).mean(dim=0)   # (n,)
    return float(per_feat.std() / per_feat.mean().clamp_min(1e-12))


def run(name, **kw):
    a = cfg(**kw)
    rec = toy_cis.train(a)
    diag = clf.classify(rec["tag"])
    cv = per_feature_mse_cv(rec["tag"])
    return {"name": name, "loss": a.loss, "k": a.k, "beats": rec["beats_baseline"],
            "cv_per_feat_mse": round(cv, 3), "label": diag["label"],
            "feats_per_neuron": diag["feats_per_neuron"], "coherence": diag["coherence"]}


def main():
    print("=" * 78)
    print("REBUTTAL (2607.04800) reproduction — n=100 d=50 k=2 (p=0.02), L2 vs L4")
    print("  expect: L2 -> high CV, NOT superposed ;  L4 -> low CV, superposed-code, beats baseline")
    print("=" * 78)
    rows = [run("rebuttal", loss="l2", k=2.0), run("rebuttal", loss="l4", k=2.0)]
    for r in rows:
        print(f"  {r['loss']:3}  CV={r['cv_per_feat_mse']:.3f}  label={r['label']:16} "
              f"beats_baseline={r['beats']}  feats/neuron={r['feats_per_neuron']}  coh={r['coherence']}")
    l2, l4 = rows
    ok_rebuttal = (l2["cv_per_feat_mse"] > l4["cv_per_feat_mse"]) and (l4["label"] == "superposed-code")
    print(f"  => rebuttal verdict recovered: {ok_rebuttal}  "
          f"(L2 CV {l2['cv_per_feat_mse']} > L4 CV {l4['cv_per_feat_mse']}; L4 label {l4['label']})")

    print("\n" + "=" * 78)
    print("SKEPTIC (2606.14673) reproduction — n=100 d=50, L2, sparsity sweep k in {1,2,4,8}")
    print("  expect: a visible bifurcation (sparse-regime vs dense-regime) as k grows")
    print("=" * 78)
    for k in (1.0, 2.0, 4.0, 8.0):
        r = run("skeptic", loss="l2", k=k)
        print(f"  k={k:>3}  CV={r['cv_per_feat_mse']:.3f}  label={r['label']:16} "
              f"beats_baseline={r['beats']}  feats/neuron={r['feats_per_neuron']}")

    print("\nInterpret: does our single 3-class label move WITH each paper's own signature? "
          "If yes, the classifier is a unifier and we can trust it across the full grid.")


if __name__ == "__main__":
    main()
