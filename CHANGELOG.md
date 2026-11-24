# Changelog

## v2.0.1
- `dmm_lib_v2.py`: implements the empty-class convention of Algorithm 1 of the manuscript. A class with no training
  observation has an identically zero density estimate; `LatentKDE` assigns it log-score `-inf` under both the MAP and
  the ML rule (it is never predicted). New method `LatentKDE.scores`. `DCAEmbed` drops empty classes before the
  correspondence analysis (previously a division by zero).
- New `test_empty_class.py`.
- No change in any reported result: all experiments use stratified splits, so every class is present in every training
  sample, and the code path for non-empty classes is unchanged (results in `results_v2/` are reproduced exactly).

## v2.0.0
- Library (`dmm_lib_v2.py`), experiment (`run_experiments_v2.py`) and figure (`make_figures_v2.py`) scripts, raw results
  (`results_v2/`) and figures (`figures_v2/`) for the current version of the manuscript.

## v1.0.0
- Original notebook with the first synthetic experiments.
