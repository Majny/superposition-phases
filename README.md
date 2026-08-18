# How far below the capacity ceiling do learned superposed codes sit?

**Work in progress** — a toy-model study of *computation in superposition* (CiS): where do
SGD-learned codes sit relative to the theoretical capacity ceiling
([Adler & Shavit, 2409.15318](https://arxiv.org/abs/2409.15318)) and the explicit
Õ(d^{2/3}) construction ([Hänni et al., 2408.05451](https://arxiv.org/abs/2408.05451)) —
and can one classifier reconcile the two sides of the 2026 compressed-computation debate?
Full framing, prior art, and claim discipline: [`PLAN.md`](PLAN.md).

## What exists so far (2 focused days; preliminary, single-seed)

**① A 3-class solution classifier that recovers both camps' verdicts.**
One rule set ({dedicated / dense-CC / superposed-code}: code coherence + cross-validated MSE +
features-per-neuron) applied to reproductions of both headline configs:

- **Rebuttal config** ([da Silva & Heimersheim, 2607.04800](https://arxiv.org/abs/2607.04800)):
  under L4 loss the trained model beats the dedicated-neuron baseline at MSE ratio **0.043**
  (~23×) — a crisp superposed-code verdict, matching the paper.
- **Skeptic config** ([2606.14673](https://arxiv.org/abs/2606.14673)): under L2 the ratio is
  **~1.0** (no superposition advantage), matching that paper's verdict.

The classifier is the intended shared instrument the debate currently lacks
(the two camps use metrics they themselves call "complementary rather than identical").

**② First capacity-scaling measurements** (`results/scaling.json`, `results/scaling.png`):
per-feature MSE vs width `d`, MSE vs co-active load `s`, and max co-active load `s_max` vs `d`.
Global log-log fits give exponents ~0.60 (MSE vs d), ~0.85 (MSE vs s), ~0.63 (s_max vs d) — but
the MSE-vs-d curve is visibly curved (local slopes steepen from −0.30 toward −1.0 across the
sweep), so **no exponent claim is made yet**. Whether learned codes genuinely beat the
random-code interference law (a possible connection to Hänni's d^{2/3} construction), or
converge to it at scale, is exactly the open question.

## Open next steps (kill-or-confirm)

1. Extended-`d` sweep at fixed `n/d` + fixed-`n` control, ≥5 seeds, local-slope plot — decides
   "structured codes beat the law" vs "finite-size transient".
2. Error bars everywhere (everything so far is single-seed).
3. The capacity-frontier overlay vs the Adler–Shavit ceiling (PLAN pillar ②) and the joint
   (sparsity × loss) grid (pillar ③).

## Status

Paused for my September state exam; active development resumes mid-September 2026.

## Run

```bash
uv run --with torch --with numpy src/reproduce.py                     # classifier validation on both debate configs
uv run --with torch --with numpy --with matplotlib src/scaling.py     # T1/T2/T3 scaling measurements
uv run --with torch --with numpy src/conv_check.py                    # step-count convergence control
```

## Author

**Jakub Dvořák** — [kubadvorak.com](https://kubadvorak.com) · hi@kubadvorak.com

Engineering is AI-assisted (Claude); I run every experiment and interpret every figure.
