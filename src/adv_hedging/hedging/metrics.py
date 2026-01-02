"""
src/adv_hedging/hedging/metrics.py
Performance evaluation metrics for hedged portfolios.
"""
import numpy as np
import pandas as pd

def calculate_portfolio_variance(weights, cov_matrix):
    """
    Calculates variance: w' * Sigma * w
    """
    return weights @ cov_matrix @ weights

def calculate_hedged_volatility(
    target_returns: pd.Series,
    hedge_returns: pd.DataFrame,
    hedge_weights: pd.Series
) -> float:
    """
    Calculates the realized volatility of the hedged position.
    
    Position = Long Target + Short Hedge
    """
    # Align dates
    common_dates = target_returns.index.intersection(hedge_returns.index)
    t_ret = target_returns.loc[common_dates]
    h_ret = hedge_returns.loc[common_dates]
    
    # Calculate daily returns of the hedge basket
    # (Matrix Multiply: Days x Assets @ Assets x 1 = Days x 1)
    hedge_basket_returns = h_ret @ hedge_weights
    
    # Net returns (Long Target - Short Hedge)
    net_returns = t_ret - hedge_basket_returns
    
    # Annualized Volatility (assuming daily data)
    return net_returns.std() * np.sqrt(252)

def calculate_risk_reduction(unhedged_vol: float, hedged_vol: float) -> float:
    """Returns the percentage reduction in volatility."""
    if unhedged_vol == 0:
        return 0.0
    return 1 - (hedged_vol / unhedged_vol)