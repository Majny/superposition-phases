# How far below the capacity ceiling do learned superposed codes sit?

**Preliminary, single-seed results** (see Status). A toy-model study of *computation in superposition* (CiS): where do
SGD-learned codes sit relative to the theoretical capacity ceiling
([Adler & Shavit, 2409.15318](https://arxiv.org/abs/2409.15318)) and the explicit
Õ(d^{2/3}) construction ([Hänni et al., 2408.05451](https://arxiv.org/abs/2408.05451)),
and can one classifier reconcile the two sides of the 2026 compressed-computation debate?

## What exists so far (2 focused days; preliminary, single-seed)

**① A 3-class solution classifier whose labels agree with both papers' headline verdicts (single seed).**
One rule set ({dedicated / dense-CC / superposed-code}: code coherence + coefficient of variation
of per-feature MSE + features-per-neuron) applied to both headline settings (n=100, d=50).
The ratios below are the trained loss divided by the dedicated-neuron baseline's loss under the
same loss function (`results/checkpoints/*_s0.json`, key `ratio`). The classifier labels are
printed by `src/reproduce.py`; the output of `src/classify.py` for the five runs it uses is stored
in `results/classify_s0.json`.

- **Rebuttal setting** ([da Silva & Heimersheim, 2607.04800](https://arxiv.org/abs/2607.04800)),
  k=2 (p=0.02): under L4 loss the ratio is **0.043** (~23× below the baseline) and the label is
  superposed-code, consistent with the paper. Under L2 at the same setting the ratio is 1.02.
- **Skeptic setting** ([Bhagat et al., 2606.14673](https://arxiv.org/abs/2606.14673)): under L2
  the ratio is **~1.0** (0.98 to 1.02 across k ∈ {1, 2, 4, 8}, i.e. within about 2% of the
  baseline), and no L2 run is labelled superposed-code. This toy model has no residual connection, so it
  lacks the mixing term that paper identifies as the source of the original model's advantage;
  the result is consistent with that paper's verdict, not a reproduction of its full analysis.

The aim is one instrument applied the same way to both settings; the two papers use different
diagnostics (the L4 paper the coefficient of variation of per-feature MSE, the skeptic paper a
split of the loss into a ReLU term and a mixing term).

**② First capacity-scaling measurements** (n=300, L4 loss, single seed; `results/scaling.json`,
`results/scaling.png`): per-feature MSE vs width `d`, MSE vs co-active load `s`, and max co-active
load `s_max` vs `d` (at MSE < 0.02). Global log-log fits give slopes of about −0.60 (MSE vs d),
+0.85 (MSE vs s) and +0.63 (s_max vs d), but the MSE-vs-d curve is visibly curved (local slopes
steepen from −0.30 to about −1.06 across d = 20 to 150), so **no exponent claim is made yet**.
Whether learned codes genuinely beat the random-code interference law (a possible connection
to Hänni's d^{2/3} construction), or converge to it at scale, is exactly the open question.

## Open next steps (kill-or-confirm)

1. Extended-`d` sweep at fixed `n/d` + fixed-`n` control, ≥5 seeds, local-slope plot: decides
   "structured codes beat the law" vs "finite-size transient".
2. Error bars everywhere (everything so far is single-seed).
3. The capacity-frontier overlay vs the Adler & Shavit ceiling (PLAN pillar ②) and the joint
   (sparsity × loss) grid (pillar ③).

## Status

**Last updated October 2026.** This repository contains only the early, preliminary work: the
code and single-seed results described above, kept as-is for provenance. The open next steps
listed above have not been run here, so none of the results above should be read as established.

## Run

```bash
uv run --with torch --with numpy src/reproduce.py                     # classifier validation on both debate configs (retrains and overwrites results/checkpoints/)
uv run --with torch --with numpy --with matplotlib src/scaling.py     # T1/T2/T3 scaling measurements
uv run --with torch --with numpy src/conv_check.py                    # step-count convergence control
```

## Author

**Jakub Dvořák** · [kubadvorak.com](https://kubadvorak.com) · hi@kubadvorak.com

Engineering is AI-assisted (Claude); I run every experiment and interpret every figure.
