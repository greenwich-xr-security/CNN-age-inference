# TODO

## MMBS Explainability Integration

Add attribution map support using the MMBS method (Schut et al., TMLR 2026).
Reference implementation: https://github.com/D1rk123/MMBS

- [ ] Decide what output to explain: `mu` (age regression) or `p_adult` (Gaussian CDF — more meaningful for age-gate use case)
- [ ] Install/import MMBS from the published repo (pip or submodule) — do not reimplement
- [ ] Write `explain.py` that:
  - Loads a trained checkpoint
  - Wraps the model to expose a scalar output for MMBS
  - Applies MMBS to hand images (recommended: 8 steps, 1024 samples, zero baseline)
  - Saves/visualises per-image attribution maps
