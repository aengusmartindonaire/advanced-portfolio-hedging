"""
tests/test_factor_engine.py
Tests for the cross-sectional factor return estimation.
"""
import pytest
import pandas as pd
import numpy as np
from adv_hedging.risk_model.factor_engine import calculate_factor_returns

def test_factor_return_shape(mock_returns_df):
    """
    Verifies that the output dataframe has the correct dimensions:
    Rows = Dates
    Columns = Factors
    """
    # 1. Setup Data
    # mock_returns_df comes from conftest.py (100 days x 20 stocks)
    tickers = mock_returns_df.columns
    factors = ['Size', 'Value', 'Momentum']
    
    # Create random factor exposures for these 20 stocks
    np.random.seed(42)
    exposures_df = pd.DataFrame(
        np.random.randn(len(tickers), len(factors)),
        index=tickers,
        columns=factors
    )
    
    # 2. Run the Engine
    factor_returns = calculate_factor_returns(
        returns_df=mock_returns_df,
        exposures_df=exposures_df,
        method='ols' # Test simple OLS first
    )
    
    # 3. Assertions
    # Should have returns for every day
    assert len(factor_returns) == len(mock_returns_df)
    # Should have columns for every factor
    assert list(factor_returns.columns) == factors
    # Values should be floats, not NaNs (unless data was fully missing)
    assert not factor_returns.isna().all().all()

def test_robust_regression_vs_ols(mock_returns_df):
    """
    Ensures that Huber regression runs and produces different results 
    than OLS (implying the robust logic is actually triggering).
    """
    tickers = mock_returns_df.columns
    factors = ['Size']
    
    exposures_df = pd.DataFrame(
        np.random.randn(len(tickers), 1),
        index=tickers,
        columns=factors
    )
    
    # Introduce a massive outlier in returns for one stock on the first day
    # This should cause OLS to skew, but Huber to ignore it
    mock_returns_df.iloc[0, 0] = 1000.0  # Massive 100,000% return
    
    # Run OLS
    f_ret_ols = calculate_factor_returns(mock_returns_df, exposures_df, method='ols')
    
    # Run Huber
    f_ret_huber = calculate_factor_returns(mock_returns_df, exposures_df, method='huber')
    
    # Check the first day (where we added the outlier)
    ols_val = f_ret_ols.iloc[0, 0]
    huber_val = f_ret_huber.iloc[0, 0]
    
    # They should be significantly different
    assert abs(ols_val - huber_val) > 0.01
    
    # Huber should be "smaller" (closer to 0) because it ignores the outlier
    # OLS tries to fit the 1000.0 return, pulling the line up/down drastically
    assert abs(huber_val) < abs(ols_val)

def test_ticker_alignment():
    """
    Verifies that the engine correctly intersects tickers if the
    Returns DF and Exposures DF don't match perfectly.
    """
    # Create disjoint datasets with ENOUGH overlap to pass the min_obs check (needs > 11)
    dates = pd.date_range('2024-01-01', periods=5)
    
    # Create a universe of 20 stocks
    all_tickers = [f"Stock_{i}" for i in range(20)]
    
    # Returns has stocks 0-14 (15 stocks)
    ret_df = pd.DataFrame(
        np.random.randn(5, 15), 
        index=dates, 
        columns=all_tickers[:15]
    )
    
    # Exposures has stocks 5-19 (15 stocks)
    # Overlap is indices 5-14 (10 stocks) -> Wait, strict check is shape[1] + 10 = 1 + 10 = 11.
    # We need 11+ stocks. Let's increase overlap.
    
    # Returns: 0 to 17 (18 stocks)
    ret_df = pd.DataFrame(
        np.random.randn(5, 18), 
        index=dates, 
        columns=all_tickers[:18]
    )
    
    # Exposures: 5 to 19 (15 stocks)
    # Overlap: 5 to 17 (13 stocks). 13 > 11, so this should pass.
    exp_df = pd.DataFrame(
        np.random.randn(15, 1), 
        index=all_tickers[5:], 
        columns=['Factor1']
    )
    
    # Run calculation
    f_ret = calculate_factor_returns(ret_df, exp_df)
    
    # Should run without error and produce results
    assert not f_ret.empty
    assert len(f_ret) == 5