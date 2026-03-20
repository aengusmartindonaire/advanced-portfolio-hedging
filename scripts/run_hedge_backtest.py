"""
scripts/run_hedge_backtest.py
Runs the systematic comparison between Factor Hedges and NLP Hedges.
"""
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from adv_hedging.data.loaders import load_wiki_data, load_risk_factors
from adv_hedging.hedging.optimization import optimize_hedge_weights
from adv_hedging.hedging.metrics import calculate_hedged_volatility
from adv_hedging.config import PROCESSED_DATA_DIR


def load_and_align_data():
    """Load and align the three datasets: returns, factor exposures, and embeddings."""
    # 1. Load factor exposures
    df_factors_raw = load_risk_factors()
    rename_map = {
        'Ticker.1': 'ticker',
        'PORT US Sz Fact Exp:D-1': 'Size',
        'PORT US Val Fact Exp:D-1': 'Value',
        'PORT US Mom Fact Exp:D-1': 'Momentum',
        'PORT US Vol Fact Exp:D-1': 'Volatility',
        'PORT US Prof Fact Exp:D-1': 'Profitability',
        'PORT US Lev Fact Exp:D-1': 'Leverage',
        'PORT US Trd Act Fact Exp:D-1': 'Trading_Activity'
    }
    factor_cols = list(rename_map.values())[1:]
    exposures = df_factors_raw.rename(columns=rename_map).set_index('ticker')[factor_cols].dropna()
    exposures.index = [t.split()[0] for t in exposures.index]

    # 2. Load wiki data and embeddings
    df_wiki = load_wiki_data()
    nomic_path = PROCESSED_DATA_DIR / "nomic_embeddings.parquet"
    if nomic_path.exists():
        df_nomic = pd.read_parquet(nomic_path)
        if 'embedding_nomic' not in df_wiki.columns:
            df_wiki = df_wiki.merge(df_nomic, on='ticker', how='left')
        embedding_col = 'embedding_nomic'
    else:
        embedding_col = 'embedding_mpnet'

    # 3. Find intersection
    common_tickers = list(set(exposures.index) & set(df_wiki['ticker']))

    # 4. Download returns
    print(f"Downloading returns for {len(common_tickers)} tickers...")
    returns_df = yf.download(
        common_tickers,
        start="2024-01-01",
        end="2025-01-01",
        auto_adjust=True,
        progress=False
    )['Close'].pct_change().dropna(how='all')

    # 5. Final alignment
    common_tickers = [t for t in common_tickers if t in returns_df.columns]
    returns_df = returns_df[common_tickers].fillna(0.0)
    exposures = exposures.loc[common_tickers]

    df_wiki = df_wiki[df_wiki['ticker'].isin(common_tickers)].set_index('ticker')
    df_wiki = df_wiki.loc[common_tickers]

    valid_emb_mask = df_wiki[embedding_col].notna()
    df_wiki = df_wiki[valid_emb_mask]
    embedding_matrix = np.stack(df_wiki[embedding_col].values)

    print(f"Aligned universe: {len(df_wiki)} stocks")
    return returns_df, exposures, df_wiki, embedding_matrix


def run_backtest(returns_df, exposures, df_wiki, embedding_matrix, sample_size=50, seed=42):
    """Run factor vs NLP hedge comparison on a random sample."""
    df_cov = pd.read_parquet(PROCESSED_DATA_DIR / "factor_covariance.parquet")
    ticker_map = {tick: i for i, tick in enumerate(df_wiki.index)}
    spec_risk = pd.Series(0.15, index=exposures.index)

    np.random.seed(seed)
    n_targets = min(sample_size, len(df_wiki))
    test_targets = np.random.choice(df_wiki.index, size=n_targets, replace=False)

    results = []
    for target in tqdm(test_targets, desc="Backtesting"):
        # --- Factor Hedge ---
        try:
            hedge_universe = exposures.drop(target)
            target_exp = exposures.loc[target]
            w_factor = optimize_hedge_weights(
                target_exp, hedge_universe, df_cov,
                spec_risk.drop(target), max_positions=10
            )
            hedge_ret_series = returns_df[hedge_universe.index]
            vol_factor = calculate_hedged_volatility(
                returns_df[target], hedge_ret_series, w_factor
            )
        except Exception:
            vol_factor = np.nan

        # --- NLP Hedge ---
        try:
            idx = ticker_map[target]
            target_emb = embedding_matrix[idx].reshape(1, -1)
            sims = cosine_similarity(target_emb, embedding_matrix)[0]
            sims[idx] = -1  # Exclude self
            top_10_idx = np.argsort(sims)[-10:]
            nlp_tickers = df_wiki.index[top_10_idx]
            w_nlp = pd.Series(0.1, index=nlp_tickers)
            nlp_ret_series = returns_df[nlp_tickers]
            vol_nlp = calculate_hedged_volatility(
                returns_df[target], nlp_ret_series, w_nlp
            )
        except Exception:
            vol_nlp = np.nan

        results.append({
            'Ticker': target,
            'Factor_Vol': vol_factor,
            'NLP_Vol': vol_nlp,
            'Sector': df_wiki.loc[target, 'sector']
        })

    res_df = pd.DataFrame(results).dropna()
    res_df['Winner'] = np.where(res_df['NLP_Vol'] < res_df['Factor_Vol'], 'NLP', 'Factor')
    res_df['Risk_Reduction_bps'] = (res_df['Factor_Vol'] - res_df['NLP_Vol']) * 10000
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Factor vs NLP Hedge Backtest")
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Number of stocks to test (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='hedge_comparison_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    print("Loading and aligning datasets...")
    returns_df, exposures, df_wiki, embedding_matrix = load_and_align_data()

    print(f"\nRunning backtest on {args.sample_size} stocks (seed={args.seed})...")
    res_df = run_backtest(
        returns_df, exposures, df_wiki, embedding_matrix,
        sample_size=args.sample_size, seed=args.seed
    )

    # Summary
    print(f"\n{'='*50}")
    print(f"Results ({len(res_df)} stocks):")
    print(res_df['Winner'].value_counts().to_string())
    print(f"\nMean Risk Reduction (NLP vs Factor): {res_df['Risk_Reduction_bps'].mean():.1f} bps")

    print("\nTop 5 NLP Wins:")
    print(res_df.nlargest(5, 'Risk_Reduction_bps')[
        ['Ticker', 'Sector', 'Risk_Reduction_bps']
    ].to_string(index=False))

    print("\nTop 5 Factor Wins:")
    print(res_df.nsmallest(5, 'Risk_Reduction_bps')[
        ['Ticker', 'Sector', 'Risk_Reduction_bps']
    ].to_string(index=False))

    res_df.to_csv(args.output, index=False)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
