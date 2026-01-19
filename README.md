# DMM Synthetic Experiments

This repository provides the reproducibility material for the synthetic experiments reported in our manuscript submitted to **MDPI Mathematics**.  
It includes the full Jupyter notebook implementation of the proposed **Density-Matrix Method (DMM)** pipeline and all synthetic generators used to evaluate controlled phenomena in high-cardinality categorical learning (separation, sparsity, irrelevant variables, and class imbalance).

## Contents

- `dmm_synthetic_experiments.ipynb`  
  End-to-end notebook implementing:
  - synthetic categorical-block generator,
  - DMM density-operator construction,
  - spectral embedding and low-rank surrogates,
  - class-conditional KDE classification,
  - baseline comparisons (PCA+KNN, Random Forest),
  - experiments S1–S4 with summary tables.

- `environment.yml`  
  Conda environment specification for reproducible execution.

## Requirements

- Conda (Miniconda/Anaconda)
- VS Code (recommended) with the **Python** and **Jupyter** extensions

## Installation (Conda)

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate dmm-synthetic
python -m ipykernel install --user --name dmm-synthetic --display-name "Python (dmm-synthetic)"