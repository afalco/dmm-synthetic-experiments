# DMM experiments: density-matrix spectral embeddings for categorical data

Reproducibility material for the manuscript
**"Density-Matrix Spectral Embeddings for Categorical Data: Operator Structure and Stability"**
(R. Bosch-Romeu, A. Falcó, J.-A. Rodríguez-Gallego).

The repository contains a self-contained Python implementation of the Density-Matrix Method (DMM), the baselines,
and the scripts that regenerate every number, table and figure of the numerical section of the manuscript.

## Contents

| File / folder | Description |
|---|---|
| `dmm_lib_v2.py` | Methods: blockwise categorical generator; count-based and class-normalized DMM operators (Gram route); latent Gaussian KDE classifier (MAP / ML rule, Scott bandwidth); baselines (Hellinger PCA+KDE, MCA+KDE, DCA+KDE, PCA+kNN, shrinkage LDA, categorical naive Bayes, logistic regression, random forest). |
| `run_experiments_v2.py` | Runs all experiments with the seeds used in the manuscript and writes CSV files to `results_v2/`. |
| `make_figures_v2.py` | Draws the figures of the numerical section from `results_v2/` into `figures_v2/`. |
| `results_v2/` | Raw results (CSV) reported in the manuscript. |
| `figures_v2/` | Figures of the manuscript (PDF). |
| `dmm_synthetic_experiments.ipynb` | Original exploratory notebook (archived in release v1.0.0). |
| `test_empty_class.py` | Check of the empty-class convention of the classifier (`python test_empty_class.py`). |
| `CHANGELOG.md` | Version history. |
| `environment.yml` | Conda environment. |

## Experiments

`run_experiments_v2.py` contains three blocks:

* **Block A – validation of the theory** (`A1`–`A5`). The count matrix is simulated exactly (multinomial sampling), so the
  empirical operator can be compared with the exact population operator:
  `A1_rate.csv` (rate in *n*), `A2_dimension.csv` (dependence on *d* versus the average cardinality),
  `A3_rare.csv` (rare and empty categories), `A4_imbalance.csv` (class imbalance and spectral gap),
  `A5_classifier.csv` (the two bandwidth regimes of the kernel classifier on an exactly computable model).
* **Block B – synthetic benchmarks** (`S1.csv`–`S4.csv`): controlled separation, high-cardinality irrelevant variables,
  irrelevant blocks and class imbalance; 10 replications, stratified 70/30 splits.
* **Block C – real data** (`real_data.csv`, `real_amazon.csv`): eight categorical UCI data sets and Amazon Employee Access,
  downloaded from OpenML (data ids 40975, 26, 50, 56, 3, 24, 46, 42 and 4135). The download is cached in `data_v2/`
  (not versioned).

Each CSV row contains the setting, the method, the replication index and the metrics (accuracy, macro-F1, balanced accuracy)
or, for Block A, the operator / subspace errors together with the theoretical bounds.

## Usage

```bash
conda env create -f environment.yml
conda activate dmm-synthetic
python run_experiments_v2.py     # about 15-20 minutes on a laptop; Block C needs internet access
python make_figures_v2.py
```

The methods can also be used directly:

```python
import numpy as np, dmm_lib_v2 as L
M = L.synthetic_model(q=20, m_list=[6]*20, k=3, informative=range(5), delta=0.4, seed=0)
codes, X, y = L.sample(M, 3000, np.random.default_rng(0))
emb = L.DMMEmbed(r=3).fit(X, y, k=3, q=20)          # count-based DMM embedding
clf = L.LatentKDE(use_priors=True).fit(emb.transform(X), y, k=3)
```

## Versions

* **v1.0.0** – original notebook with the first synthetic experiments.
* **v2.0.0** – library, experiment and figure scripts, and results for the current version of the manuscript.
* **v2.0.1** – implements the empty-class convention of the classifier (Algorithm 1); reported results unchanged. See `CHANGELOG.md`.
