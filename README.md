# Advanced Portfolio Hedging & Risk Analytics

## 📌 Project Overview
This project implements a sophisticated risk management system designed for Ultra High Net Worth (UHNW) clients holding concentrated single-stock positions. The core constraint is **tax efficiency**: reducing portfolio risk without triggering capital gains taxes by selling the underlying asset.

We compare two distinct hedging approaches:
1.  **Quantitative Factor Model:** A traditional Barra-style risk model using Bloomberg factors (Size, Value, Momentum) and robust regression.
2.  **NLP Semantic Model:** A novel approach using Large Language Model (LLM) embeddings (Nomic v1.5) to identify "fundamental peers" based on semantic business similarity.

## 🚀 Key Results
The backtest results (Notebook 03) reveal that **NLP-based hedging outperforms traditional factor models for idiosyncratic companies** where sector labels are insufficient.

| Strategy | Win Case Example | Rationale |
| :--- | :--- | :--- |
| **NLP Hedge** | **Flextronics (FLEX)** | **+402 bps risk reduction.** NLP correctly identified niche electronics manufacturing peers that generic "Tech" factors missed. |
| **NLP Hedge** | **Mosaic (MOS)** | **+392 bps risk reduction.** Semantic search captured the specific fertilizer/commodity risk better than broad "Materials" sector factors. |
| **Factor Hedge** | **Apple (AAPL)** | **-200 bps.** For mega-cap stocks driven by broad market flows, the systematic factor model proved superior to semantic matching. |

## 📂 Repository Structure

```text
.
├── data/                   # Raw and Processed data (Bloomberg, Wikipedia, Embeddings)
├── notebooks/              # Jupyter Notebooks (Sequential Logic)
│   ├── 00_exploratory_data_analysis.ipynb   # Data Cleaning & Veralto/UMB Fixes
│   ├── 01_factor_model_construction.ipynb   # Huber Robust Regression & Factor Returns
│   ├── 02_nlp_embedding_generation.ipynb    # Nomic v1.5 Embeddings & Context-Aware Chunking
│   ├── 03_hedging_strategy_comparison.ipynb # The Backtest: Factor Optimization vs. NLP
│   └── 04_ai_revolution_clustering.ipynb    # Extra Credit: Unsupervised AI Clustering
├── scripts/                # Production scripts for batch jobs
├── src/                    # Source code package (adv_hedging)
│   ├── hedging/            # Optimization & Metrics
│   ├── nlp/                # Text Processing & Embeddings
│   └── risk_model/         # Factor Engine
├── environment.yml         # Conda environment definition
└── pyproject.toml          # Python dependencies
```

## 🛠 Installation & Setup

This project uses a custom Conda environment (hedging_clean) with Python 3.10.

1. Clone the Repository:

    ```bash
    git clone [https://github.com/your-username/advanced-portfolio-hedging.git](https://github.com/your-username/advanced-portfolio-hedging.git)
cd advanced-portfolio-hedging
    ```

2. Create Environment:

    ```bash
    conda env create -f environment.yml
    conda activate hedging_clean
    ```

3. Install Local Package:

    ```bash
    pip install -e .
    ```

## 🧠 Methodology Details

1. **Factor Risk Model**
    
    - Data: 7 Bloomberg Risk Factors (Size, Value, Momentum, Volatility, Profitability, Leverage, Trading Activity).

    - Estimation: Uses Huber Robust Regression (epsilon=1.35) to estimate daily factor returns, minimizing the impact of outliers (meme stocks).

    - Covariance: Factor covariance matrix estimated on a 2-year rolling window.

2. **NLP Engine**
    - Model: nomic-ai/nomic-embed-text-v1.5 (Matryoshka embeddings).

    - Innovation: Implements Context-Aware Chunking. Every text chunk includes the company metadata header ("Title: Apple Inc...") to prevent context loss in long documents.

    - Evaluation: Validated using Silhouette Scores on GICS sectors, outperforming standard MPNet and BGE models.

3. **Hedging Optimization (Part 3)**
    - Objective: Minimize Active Risk (Tracking Error) against the target stock.

    - Constraints:
    
        - Max 10 positions (Cardinality constraint for operational simplicity).
        - Max weight 25% per position.
        - Hedge Ratio: 100% (Dollar Neutral).

4. AI Revolution Clustering (Extra)

    - Goal: Challenge expert "Maker vs. User" labels using unsupervised learning.
    - Technique: UMAP dimensionality reduction + HDBScan density clustering.
    - Insight: The model identified "Hybrid" clusters (e.g., Cloud Hyperscalers like AMZN/GOOGL) that act as both Makers and Users, defying binary classification.

📊 Usage
To reproduce the full analysis, run the notebooks in order:

1. `00_exploratory_data_analysis.ipynb`: Verifies data integrity.

2. `01_factor_model_construction.ipynb`: Builds the risk model.

3. `02_nlp_embedding_generation.ipynb`: Generates Nomic embeddings (requires GPU/MPS).

4. `03_hedging_strategy_comparison.ipynb`: Runs the 50-stock backtest loop.

Alternatively, use the command-line interface:

```bash
python scripts/run_hedge_backtest.py
```

-- Last modified Dec 20, 2025.