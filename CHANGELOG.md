# Changelog

## v2.1.0
- `dmm_lib_v2.py`:
  - empty-class convention of Algorithm 1 of the manuscript: a class with no training observation has an
    identically zero density estimate; `LatentKDE` assigns it log-score `-inf` under both the MAP and the ML rule
    (it is never predicted). Ties are broken by the smallest class index. New method `LatentKDE.scores`.
  - `DCAEmbed` drops empty classes before the correspondence analysis (previously a division by zero).
  - `dmm_basis` returns `min(r, numerical rank)` columns (eigenvalues `<= rtol * lambda_max` are treated as zero),
    in agreement with the definition of the spectral coordinates for `r <= rank(rho_D)`. Previously degenerate
    eigenvalues were clamped to `1e-12`, which produced null or noise-amplified coordinates.
- `run_experiments_v2.py`: writes to `results_rerun/` by default (`--out` to change it, so the archived
  `results_v2/` is not overwritten by accident); `--blocks` selects blocks; Block A also records the bound based on
  Hoeffding's inequality and the Lipschitz continuity of the square root cited in the manuscript
  (columns `lipschitz_condition_holds`, `lipschitz_bound` of `A3_rare.csv`, formerly `v1_*`; values unchanged).
- `make_figures_v2.py`: `--results` and `--out` options.
- `environment.yml`: exact versions used to produce the archived results.
- New: `LICENSE` (MIT), `.zenodo.json` (authors and metadata), `test_empty_class.py`.
- No change in any reported result: a full re-run with the pinned environment reproduces all CSV files in
  `results_v2/` exactly (maximum absolute difference 0).

## v2.0.0
- Library (`dmm_lib_v2.py`), experiment (`run_experiments_v2.py`) and figure (`make_figures_v2.py`) scripts, raw results
  (`results_v2/`) and figures (`figures_v2/`) for the current version of the manuscript.

## v1.0.0
- Original notebook with the first synthetic experiments.
