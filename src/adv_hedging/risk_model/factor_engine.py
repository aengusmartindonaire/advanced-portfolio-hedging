"""
src/adv_hedging/risk_model/factor_engine.py
Calculates factor returns using cross-sectional regression.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from tqdm import tqdm

def calculate_factor_returns(
    returns_df: pd.DataFrame, 
    exposures_df: pd.DataFrame, 
    method: str = 'huber'
) -> pd.DataFrame:
    """
    Performs cross-sectional regression for each day to estimate factor returns.
    
    Equation: R_i = Beta * F + epsilon
    We solve for F (Factor Returns).
    
    Args:
        returns_df: DataFrame of asset returns (Index=Date, Cols=Tickers)
        exposures_df: DataFrame of factor exposures (Index=Tickers, Cols=Factors)
                      (Must be aligned with returns columns)
        method: 'ols' or 'huber' (robust)
    """
    # Align tickers
    common_tickers = returns_df.columns.intersection(exposures_df.index)
    returns_aligned = returns_df[common_tickers]
    exposures_aligned = exposures_df.loc[common_tickers]
    
    factor_returns = []
    dates = returns_aligned.index
    
    # Pre-initialize the regressor
    if method == 'huber':
        # epsilon=1.35 is standard for 95% efficiency
        model = HuberRegressor(epsilon=1.35) 
    else:
        model = LinearRegression()
        
    print(f"Estimating factor returns using {method.upper()} regression...")
    
    for date in tqdm(dates):
        # Get returns for this day
        day_ret = returns_aligned.loc[date]
        
        # Filter out missing returns for this specific day (e.g., halted stocks)
        valid_mask = ~day_ret.isna()
        
        if valid_mask.sum() < exposures_aligned.shape[1] + 10:
            # Skip if not enough data points to regress
            continue
            
        X = exposures_aligned.loc[valid_mask]
        y = day_ret.loc[valid_mask]
        
        try:
            model.fit(X, y)
            
            # Store coefficients (Factor Returns)
            # Create a Series indexed by factor names
            f_ret = pd.Series(model.coef_, index=exposures_aligned.columns)
            f_ret.name = date
            factor_returns.append(f_ret)
            
        except Exception as e:
            # In case Huber fails to converge
            print(f"Regression failed for {date}: {e}")
            
    return pd.DataFrame(factor_returns)