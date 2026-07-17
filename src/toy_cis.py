#!/usr/bin/env python3
"""
Compressed-computation toy model: does a d-neuron layer compute n > d ReLUs in superposition?

THE TASK (what we're asking the network to do)
----------------------------------------------
Inputs x in R^n are SPARSE: each feature x_i is "active" with probability p, and when active
takes a value in [-1, 1]; otherwise it's 0. The target is the elementwise nonlinearity
    y_i = ReLU(x_i)   for every i = 1..n.
So the job is "compute n independent ReLUs". (ReLU is a real computation here because active
values can be negative — the unit has to gate.)

THE MODEL
---------
One hidden layer of d neurons:
    h      = ReLU(W_in @ x + b_in)     # W_in: (d, n)
    y_hat  = W_out @ h + b_out         # W_out: (n, d)
When d < n the network CANNOT give each feature its own neuron. If it still matches y well, it
must be computing several ReLUs per neuron — that is computation *in superposition*.

THE BASELINE TO BEAT
--------------------
A "dedicated" solution handles only d of the n features perfectly (one neuron each) and outputs
0 for the other n - d. Its loss is whatever those unhandled features contribute. If a trained
model beats this dedicated baseline, it is genuinely doing compressed computation, not just
picking d features. We report both numbers so the comparison is unambiguous.

WHAT TO LOOK AT WHEN YOU RUN IT
-------------------------------
The final line prints  trained_loss  vs  dedicated_baseline_loss.  trained << baseline  => the
model found a superposed / compressed solution. trained ~ baseline => it just did d features.
Then run src/classify.py on the saved checkpoint to see *which* kind of solution it is.

Run (CPU is fine, ~seconds):
    uv run --with torch --with numpy src/toy_cis.py --n 100 --d 50 --k 3 --loss l2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def make_batch(n: int, k: float, batch: int, gen: torch.Generator) -> torch.Tensor:
    """Sparse inputs: each of n features active w.p. p=k/n (so ~k active per example), value ~U(-1,1)."""
    p = k / n
    mask = (torch.rand(batch, n, generator=gen) < p).float()
    vals = torch.rand(batch, n, generator=gen) * 2 - 1  # U(-1, 1)
    return mask * vals


def loss_fn(pred: torch.Tensor, target: torch.Tensor, exponent: int) -> torch.Tensor:
    """L2 = mean squared error, L4 = mean 4th-power error. The exponent changes which solution
    SGD prefers (Silva-Heimersheim): L4 punishes large per-feature errors harder and tends to
    favour clean 'codeword' solutions. We just expose it as a knob."""
    return ((pred - target).abs() ** exponent).mean()


class CIS(torch.nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.w_in = torch.nn.Linear(n, d)
        self.w_out = torch.nn.Linear(d, n)

    def forward(self, x):
        return self.w_out(torch.relu(self.w_in(x)))


def dedicated_baseline_loss(n: int, d: int, k: float, exponent: int, gen: torch.Generator,
                            batch: int = 20000) -> float:
    """Oracle that computes ReLU(x_i) exactly for the first d features and 0 for the rest."""
    x = make_batch(n, k, batch, gen)
    y = torch.relu(x)
    pred = torch.zeros_like(y)
    pred[:, :d] = y[:, :d]                       # d features handled perfectly
    return loss_fn(pred, y, exponent).item()


def train(args) -> dict:
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    model = CIS(args.n, args.d)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        x = make_batch(args.n, args.k, args.batch, gen)
        y = torch.relu(x)
        loss = loss_fn(model(x), y, args.exponent)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.verbose and step % max(1, args.steps // 10) == 0:
            print(f"  step {step:5d}  loss {loss.item():.5f}", flush=True)

    # final eval on a fresh big batch
    with torch.no_grad():
        xe = make_batch(args.n, args.k, 20000, gen)
        ye = torch.relu(xe)
        trained = loss_fn(model(xe), ye, args.exponent).item()
    baseline = dedicated_baseline_loss(args.n, args.d, args.k, args.exponent, gen)

    tag = f"n{args.n}_d{args.d}_k{args.k}_{args.loss}_s{args.seed}"
    out = ROOT / "results" / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, out / f"{tag}.pt")

    rec = {**vars(args), "trained_loss": trained, "dedicated_baseline_loss": baseline,
           "beats_baseline": trained < baseline, "ratio": trained / baseline if baseline else float("nan"),
           "tag": tag}
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser(description="Compressed-computation toy model.")
    ap.add_argument("--n", type=int, default=100, help="number of input features")
    ap.add_argument("--d", type=int, default=50, help="number of hidden neurons (d < n = compression)")
    ap.add_argument("--k", type=float, default=3.0, help="expected active features per example (sparsity)")
    ap.add_argument("--loss", choices=["l2", "l4"], default="l2")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    args.exponent = 2 if args.loss == "l2" else 4

    rec = train(args)
    print(f"\nn={rec['n']} d={rec['d']} k={rec['k']} loss={rec['loss']} seed={rec['seed']}")
    print(f"  trained_loss           = {rec['trained_loss']:.5f}")
    print(f"  dedicated_baseline_loss= {rec['dedicated_baseline_loss']:.5f}")
    print(f"  -> {'BEATS baseline (compressed computation!)' if rec['beats_baseline'] else 'no better than d dedicated features'}"
          f"   (ratio {rec['ratio']:.2f})")
    print(f"  saved: results/checkpoints/{rec['tag']}.pt  — next: uv run src/classify.py --tag {rec['tag']}")


if __name__ == "__main__":
    main()
