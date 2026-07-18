# How far below the capacity ceiling do learned superposed codes sit?

*Working plan. Jakub Dvořák · started 2026-07-17 · reshaped 2026-07-18 after a novelty check.*

## The one-paragraph pitch (reshaped)

Networks can *compute* in superposition — a layer of `m` neurons computing a nonlinear function of
`n ≫ m` sparse inputs. Theory (Adler & Shavit) proves a hard ceiling of `~O(m²/log m)` features;
constructions (Hänni et al.) hit `Õ(m^{2/3})`. **Nobody has ever plotted where SGD-*learned* codes
actually sit relative to that ceiling.** The one paper positioned to do it explicitly declined
("our model does not approach these theoretical limits… we focused on the mechanics"). That gap is
the spine: **sweep model size and loss, measure the learned code's density, and plot its trajectory
against the O(m²/log m) bound** — does crossing L2→L4 (or changing `d/n`) push learned codes toward
or away from the ceiling? Second pillar: a single **3-class classifier** ({dedicated / dense-CC /
superposed-code}) that, *validated by reproducing each prior paper's own verdict*, becomes the shared
instrument the debate currently lacks — and run across the (sparsity × loss) plane the two camps
never co-sampled, it localizes *where* the transition sits.

## Prior art & honest framing (the novelty check that reshaped this)

The debate is real and hot (two Apollo-adjacent teams, papers weeks apart) — respect it:
- Braun et al., APD, [2501.14926](https://arxiv.org/abs/2501.14926) — introduces the `100-ReLU/50-neuron` testbed (single config).
- Newgas, *Universal-AND / dense circuits*, [2507.09816](https://arxiv.org/abs/2507.09816) — **already has a single-axis (sparsity) phase diagram + a hidden-dim sweep** and a 2-way classifier. → we must **never** claim "first phase diagram."
- Skeptic, [2606.14673](https://arxiv.org/abs/2606.14673) — sparsity sweep, **L2 only**, dense-vs-sparse bifurcation by observation.
- da Silva & Heimersheim, [2607.04800](https://arxiv.org/abs/2607.04800) — **owns the L2-vs-L4 takeaway** (L2 = naive, exponent > 2 = superposed), at fixed sparsity `p=0.02`.
- Capacity: Adler & Shavit [2409.15318](https://arxiv.org/abs/2409.15318) (bound, pure theory, zero trained models); Hänni et al. [2408.05451](https://arxiv.org/abs/2408.05451) (construction).

**What is genuinely un-done anywhere (arXiv, LessWrong/AF, Apollo, transformer-circuits, MATS/SPAR):**
(1) the empirical **learned-code-density vs Adler-Shavit-ceiling** overlay; (2) **one consistent
classifier** applied uniformly across all setups (the L4 paper itself calls its metric "complementary
rather than identical" to the skeptic's — the fragmentation is real); (3) the **joint (sparsity ×
loss)** grid the two camps never co-sampled.

**We claim exactly:** "first empirical capacity overlay + one reconciling classifier + the joint
sweep." **We never claim:** "first phase diagram," or "settles/breaks the debate" (say "maps *where*
the transition sits"). "We ran more configs" and the taxonomy are **not** the contribution.

## The task (primer)

Compressed-computation toy model: sparse inputs `x ∈ ℝⁿ` (each `xᵢ` active w.p. `p`, value in
`[-1,1]`); target `yᵢ = ReLU(xᵢ)`; model = one hidden layer of `m = d < n` ReLU neurons. With `d<n`
the net can't dedicate a neuron per feature — matching `y` means computing in superposition. The
`3` regimes the classifier separates: **dedicated** (≈1 feature/neuron, only `d` handled),
**dense-CC** (many features/neuron, unstructured), **superposed-code** (features on a structured
low-coherence code, pseudoinverse-decodable — the "codeword" regime).

## v0 — the minimum shippable unit

**① The classifier as a validated unifier** (`src/classify.py`, done in draft). Reproduce the two
camps' headline configs and show the *same* classifier recovers *each* paper's verdict — skeptic
(d=50/m=100, L2, sparsity sweep → dense/sparse bifurcation) and rebuttal (p=0.02, loss sweep → L2
naive vs >2 superposed). Without this it's a 5th incompatible metric, not a unifier.

**② The capacity-frontier plot** (`src/capacity.py`) — **THE figure.** Define learned-code density /
effective feature count; sweep size `m` × loss {L2,L4}; plot vs the `O(m²/log m)` ceiling + the Hänni
construction. *If we ship one figure, it's this.*

**③ The (sparsity × loss) reconciliation grid** (`src/sweep.py`) — one classifier across the plane
the camps never co-sampled; show where the L2 boundary moves/dissolves under L4.

**GO / NO-GO (end of ~2 weeks):** GO if ① validates AND ② shows a clean, interpretable
learned-vs-bound trajectory → expand + write. NO-GO if the classifier can't reproduce prior verdicts
(then it isn't a unifier) → the honest fallback is the classifier-validation-and-reconciliation note
alone. Either ships a public artifact.

## The dominant risk (not novelty — speed)

Two Apollo-adjacent teams ship weeks apart, and the failure mode here is *not finishing*. So: scope
brutally to laptop scale, **freeze ① + ② as the minimum unit**, and get a short **LessWrong/AF post
out fast** (it stakes the claim and is the MATS/Apollo-relevant signal) before any workshop paper.

## References
See the six links above (verified 2026-07-18) + Newgas 2507.09816.

## How we work (pedagogical)
AI writes the code; **I run every experiment and interpret every figure**, because MATS probes this
live. First action: reproduce one prior config and check the classifier recovers that paper's verdict.
