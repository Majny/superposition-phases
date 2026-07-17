# The Phase Diagram of Computation in Superposition

*Working plan + primer. Jakub Dvořák · started 2026-07-17. Repo name is a placeholder — rename freely.*

## The one-paragraph pitch

Neural networks routinely represent **more features than they have neurons** by packing them
into overlapping directions — *superposition*. A sharper, less-understood claim is that networks
also **compute in superposition**: a layer with `d` neurons can compute a nonlinear function of
`n ≫ d` sparse inputs at once, not just store them. When does gradient descent actually *find*
such a solution, versus collapsing to a boring "handle only `d` features" one? Nobody has mapped
that boundary. **This project maps the phase diagram** — over input sparsity, the compression
ratio `d/n`, the task, and the loss — of when SGD discovers genuine computation-in-superposition,
and measures how far the learned codes sit from the theoretical capacity bounds.

## Why this, why me

- **It rewards math, not ML-engineering.** The hard, differentiating part is coding-theory /
  compressed-sensing analysis (mutual coherence, pseudoinverse decoding, capacity bounds) — my
  actual edge. The ML is a one-hidden-layer MLP.
- **It runs on a laptop.** Toy MLPs train in seconds–minutes on a 64GB MacBook, CPU is fine.
  No cluster, no queue, no fused-kernel plumbing. So it survives September exam prep, and
  **every work session ends with a figure** (the point — I don't finish grand plans, I finish
  figures).
- **It's a live 2025-26 debate centered at Apollo Research** (compressed-computation model vs.
  codeword solutions vs. capacity bounds) — precisely the MATS/PhD community I'm applying into,
  so a solid result doubles as the application.
- **Guaranteed fallback (de-risk):** even if the headline phase-diagram story is messy, a clean
  open-source *replication + adjudication* of the current debate with ONE consistent solution
  classifier applied uniformly to all the competing setups is itself a workshop paper — the
  debate currently rests on incompatible diagnostics. This byproduct is reachable in weeks 1–2.

## What "computation in superposition" means (primer)

Toy setup (the standard "compressed computation" task):
- `n` input features `x ∈ ℝⁿ`, **sparse**: each `xᵢ` is active with probability `s` (small),
  value e.g. `Uniform(0,1)`, else 0.
- **Target = an elementwise nonlinearity**, e.g. `yᵢ = ReLU(xᵢ)` for all `i`. So the task is
  "compute `n` independent ReLUs."
- **Model = one hidden layer of `d` neurons** (`d < n`): `h = ReLU(Wᵢₙ x + b)`, `ŷ = W_out h + c`.
- With `d < n` the network *cannot* dedicate one neuron per feature. If it still computes all `n`
  ReLUs well, it must be doing it **in superposition** — exploiting that few features are active
  at once so collisions are rare. That's the phenomenon.

Three solution types a trained network can land in (the **classifier** we build decides which):
1. **Dedicated** — each neuron computes ~one feature; only ~`d` features handled. Boring.
2. **Dense compressed-computation (CC)** — neurons participate in many features, dense weights.
3. **Superposed code** — features live on a structured, near-orthogonal code (low mutual
   coherence), decodable by pseudoinverse. This is the "codeword" regime.

The open question = **where in (sparsity `s`, ratio `d/n`, loss) does each regime appear**, and
how close is the learned code to the proven capacity limit `~Õ(d²)` features.

## v0 — the first two weeks (the go/no-go gate)

**Week 1 — harness + classifier (all laptop/CPU).**
- `src/toy_cis.py` — the compressed-computation model, sparse-data generator, training under a
  configurable loss exponent (**L2** and **L4** — Silva-Heimersheim showed the exponent changes
  the solution). Reproduce the basic `n=100, d=50, ReLU` compressed-computation result; verify it
  beats the dedicated-`d`-features baseline.
- `src/classify.py` — the **solution classifier**: binary neuron-participation matrix, pseudoinverse
  decoding error, mutual coherence / Welch-bound ratio of the feature code. Output: a label +
  the diagnostics.
- ✅ *cite, don't re-derive* the L4 codeword result and the loss-exponent sweep (already done by
  Silva-Heimersheim) — we build ON it, we don't repeat it.

**Week 2 — the first phase-diagram slice.**
- `src/sweep.py` — sweep **sparsity `s ∈ {1,2,4,8}` active features × compression ratio `d/n`**
  (5 seeds, ~300 tiny runs, minutes on CPU) under L2 and L4; classify every run.
- **Deliverable = ONE plot**: the 2-D (s × d/n) slice colored by solution type, plus a same-loss
  comparison of the learned code against a hand-designed binary code. That plot immediately says
  whether the effect and the phase structure exist.

**GO / NO-GO (end of week 2):** GO if the sweep shows a *clean, reproducible boundary* between
solution regimes (the phase structure is real) → expand the diagram + the capacity-bound
comparison → public write-up. NO-GO if it's mush → pivot to the guaranteed fallback (the
consistent-classifier adjudication of the debate) and write THAT up. **Either branch ships a
public artifact.** No third rescue experiment.

## The ~2-month arc (lean — don't inflate this)

- **Wk 1–2:** v0 above → go/no-go.
- **Wk 3–4:** full phase diagram (add tasks: `abs`, `x²`, small Boolean circuits; add the tied-
  weights variant); measure learned-code vs. `Õ(d²)` capacity bound. Post a **short public
  write-up by mid-August** to stake the claim (the area is hot — see Risk).
- **Wk 5–8 (around státnice):** consolidate into a workshop-length note + clean released code;
  intermittent, low-compute — survives exam weeks.

## The one real risk

The area is hot and Apollo-adjacent researchers publish in it (the L4 result is weeks old). A
single slice could get scooped. **Mitigation:** ship the *narrow* v0 (the `s × d/n` slice, NOT the
loss-exponent sweep others already did) fast, and a public write-up by mid-August; the multi-axis
diagram + the learned-code-vs-bound comparison are broad enough to survive a partial scoop.

## How we work (pedagogical, on purpose)

AI writes the code; **I run every experiment and interpret every figure myself**, because a MATS
interview will probe this project live. Each file has a "what am I looking at" note. First action:
read `src/toy_cis.py`, run it, and eyeball whether the model beats the dedicated baseline — before
we sweep anything.
