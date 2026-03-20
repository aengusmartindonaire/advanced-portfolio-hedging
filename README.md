# Advanced Portfolio Hedging & Risk Analytics

## Overview

This project implements a risk management system designed for Ultra High Net Worth (UHNW) clients holding concentrated single-stock positions. The core constraint is **tax efficiency**: reducing portfolio risk without triggering capital gains taxes by selling the underlying asset.

We compare two distinct hedging approaches:
1. **Quantitative Factor Model** — A Barra-style risk model using Bloomberg factors (Size, Value, Momentum, Volatility, Profitability, Leverage, Trading Activity) with robust regression.
2. **NLP Semantic Model** — A novel approach using sentence embeddings to identify "fundamental peers" based on semantic business similarity from Wikipedia company descriptions.

## Key Results

A 50-stock stratified backtest (Notebook 03) suggests that **NLP-based hedging can outperform traditional factor models for idiosyncratic companies** where broad sector labels fail to capture specific business risks.

| Strategy | Example | Result |
| :--- | :--- | :--- |
| **NLP Hedge** | Western Digital (WDC) | **+1,017 bps** risk reduction vs. factor hedge |
| **NLP Hedge** | Ovintiv (OVV) | **+992 bps** risk reduction vs. factor hedge |
| **NLP Hedge** | Toll Brothers (TOL) | **+913 bps** risk reduction vs. factor hedge |
| **Factor Hedge** | Ferrari (RACE) | **-396 bps** — factor model superior for globally recognized brands |
| **Factor Hedge** | Apple (AAPL) | **-200 bps** — systematic factors dominate for mega-cap stocks |

**Win Rate:** NLP won 28/50 cases (56%), Factor won 22/50 (44%).

> **Note:** This is a directional result on a 50-stock sample. See [Limitations](#limitations--future-work) for discussion of statistical significance.

## Repository Structure

```text
.
├── data/                   # Raw and processed data (gitignored — see data/README.md)
│   ├── raw/                # Bloomberg factors + Wikipedia embeddings
│   ├── interim/            # Cleaned intermediate outputs
│   └── processed/          # Factor returns, covariance, embeddings
├── notebooks/              # Jupyter notebooks (run in order)
│   ├── 00_exploratory_data_analysis.ipynb
│   ├── 01_factor_model_construction.ipynb
│   ├── 02_nlp_embedding_generation.ipynb
│   ├── 03_hedging_strategy_comparison.ipynb
│   └── 04_ai_revolution_clustering.ipynb
├── scripts/                # Standalone scripts
│   ├── run_hedge_backtest.py
│   └── run_clustering_analysis.py
├── src/adv_hedging/        # Source code package
│   ├── data/               # Data loading and cleaning
│   ├── risk_model/         # Cross-sectional factor engine
│   ├── nlp/                # Text processing and context-aware chunking
│   └── hedging/            # Portfolio optimization and risk metrics
├── tests/                  # Unit tests (pytest)
├── environment.yml         # Conda environment definition
└── pyproject.toml          # Package configuration
```

## Installation

This project uses Conda with Python 3.10. Numba must be installed via conda (not pip) to get LLVM binaries.

```bash
git clone https://github.com/aengusmartindonaire/advanced-portfolio-hedging.git
cd advanced-portfolio-hedging
conda env create -f environment.yml
conda activate hedging_clean
pip install -e .
```

## Data Requirements

The raw data files are not included in the repository. You need:

1. `data/raw/20250930_stk_wiki_em.parquet` — ~650 S&P companies with Wikipedia text and pre-computed embeddings (MPNet, BGE-large, BGE-small).
2. `data/raw/20250928_US_Port.xlsx` — Bloomberg factor exposures (Size, Value, Momentum, Volatility, Profitability, Leverage, Trading Activity).

Place these files in `data/raw/` before running the notebooks. Historical price data is downloaded automatically via yfinance.

## Methodology

### Factor Risk Model (Notebook 01)
- **Data:** 7 Bloomberg risk factors for ~664 companies.
- **Estimation:** Huber Robust Regression (epsilon=1.35) for daily cross-sectional factor return estimation, minimizing the impact of outlier stocks.
- **Output:** Factor return time series and a 7x7 factor covariance matrix on a 2-year rolling window.

### NLP Engine (Notebook 02)
- **Models Tested:** Four embedding models — Nomic v1.5 (custom-generated), MPNet, BGE-large, and BGE-small (pre-computed).
- **Context-Aware Chunking:** Every text chunk includes a metadata header (Title, URL, Sector) to prevent context loss in long documents.
- **Evaluation:** Silhouette scores on GICS sectors showed BGE-large (0.058) provided the best sector separation, followed by MPNet (0.040), BGE-small (0.032), and Nomic (-0.119). The backtest in Notebook 03 uses Nomic embeddings as the student-implemented model, with BGE-large available as an alternative already present in the raw data.

### Hedging Optimization (Notebook 03)
- **Objective:** Minimize Active Risk (Tracking Error) against the target stock.
- **Constraints:** Max 10 positions (cardinality), max 25% per position, 70-130% invested.
- **Two-Stage Approach:** Stage 1 solves the unconstrained dense hedge; Stage 2 re-optimizes on the top 10 positions by weight.
- **NLP Comparison:** Top 10 cosine-similarity neighbors, equal-weighted. This is a deliberately simple heuristic to isolate the value of semantic matching from optimization effects.

### AI Revolution Clustering (Notebook 04)
- **Goal:** Challenge expert "AI Maker vs. User" labels with unsupervised learning.
- **Technique:** UMAP dimensionality reduction + HDBSCAN density clustering with hyperparameter tuning.
- **Insight:** The model identified "hybrid" clusters — Cloud Hyperscalers (AMZN, GOOGL) acting as both Makers and Users, defying the binary classification.

## Usage

Run notebooks in order (00 through 04):

```bash
jupyter notebook notebooks/00_exploratory_data_analysis.ipynb
```

Or run the backtest from the command line:

```bash
python scripts/run_hedge_backtest.py --sample-size 50 --seed 42
```

Run tests:

```bash
pytest tests/
```

## Limitations & Future Work

- **Sample size:** The 50-stock backtest is directional. A full-universe test with bootstrap confidence intervals would strengthen the conclusions.
- **NLP hedge simplicity:** The NLP strategy uses equal-weight top-10 neighbors — a deliberately simple baseline. An optimized NLP hedge (minimizing tracking error using NLP-selected peers) would be a fairer comparison against the factor model's full optimization.
- **Specific risk estimation:** The backtest uses a constant specific variance (0.15) rather than a fully estimated idiosyncratic risk model. A production system would estimate this from regression residuals.
- **Single time period:** Results are based on 2024 returns only. Walk-forward or rolling-window validation across multiple periods would test robustness.
- **Embedding model choice:** BGE-large outperformed Nomic on silhouette scores for sector clustering. Further work could test whether BGE-large also produces better hedging outcomes.
