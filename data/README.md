# Data Directory

This directory is gitignored. To reproduce the analysis, you need the following files:

## Raw Data (`raw/`)

| File | Description |
| :--- | :--- |
| `20250930_stk_wiki_em.parquet` | ~650 S&P companies with Wikipedia text content and pre-computed sentence embeddings (MPNet, BGE-large, BGE-small). |
| `20250928_US_Port.xlsx` | Bloomberg factor exposures: Size, Value, Momentum, Volatility, Profitability, Leverage, Trading Activity. |

## Generated Data

The following are produced by running the notebooks in sequence and do not need to be sourced externally:

| Directory | File | Source |
| :--- | :--- | :--- |
| `interim/` | `wiki_embeddings_cleaned.parquet` | Notebook 00 (data cleaning) |
| `processed/` | `factor_returns_robust.parquet` | Notebook 01 (Huber regression) |
| `processed/` | `factor_covariance.parquet` | Notebook 01 (factor covariance) |
| `processed/` | `nomic_embeddings.parquet` | Notebook 02 (Nomic v1.5 embeddings) |

Historical price data is downloaded automatically via yfinance during notebook execution.
