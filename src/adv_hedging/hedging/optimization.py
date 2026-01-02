"""
src/adv_hedging/hedging/optimization.py
Core optimization logic for portfolio hedging.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def objective_tracking_error(weights, target_exposures, universe_exposures, factor_cov_matrix, specific_variances):
    """
    Objective function: Minimize Active Risk (Tracking Error).
    Active Risk^2 = Systemic_Risk^2 + Specific_Risk^2
    
    Systemic Risk = (w_target - w_hedge)' * Factor_Cov * (w_target - w_hedge)
    """
    # 1. Calculate Net Exposure (Target - Hedge)
    # weights is shape (N,), universe_exposures is (N, F)
    # hedge_exposure will be (F,)
    hedge_exposure = weights @ universe_exposures
    net_exposure = target_exposures - hedge_exposure
    
    # 2. Systemic Risk Component (Variance)
    # (F,) @ (F,F) @ (F,) -> scalar
    systemic_variance = net_exposure @ factor_cov_matrix @ net_exposure
    
    # 3. Specific Risk Component
    # We only care about the specific risk of the HEDGE portfolio here for minimization
    # (assuming target specific risk is constant/unhedgeable by other stocks)
    specific_variance = np.sum((weights ** 2) * specific_variances)
    
    return systemic_variance + specific_variance

def optimize_hedge_weights(
    target_exposures: pd.Series,
    universe_exposures: pd.DataFrame,
    factor_cov_matrix: pd.DataFrame,
    specific_variances: pd.Series,
    max_positions: int = 10
) -> pd.Series:
    """
    Calculates optimal hedge weights subject to constraints.
    Uses a two-stage approach to handle cardinality (max 10 stocks).
    """
    num_assets = len(universe_exposures)
    
    # --- STAGE 1: Relaxed Optimization (Find the best "dense" hedge) ---
    
    # Constraints: 
    # 1. Fully invested hedge (sum of weights between 0.7 and 1.3 as per project specs)
    #    Eq constraint: sum(w) - 1.0 = 0 (Softened to bounds below)
    cons = [
        {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.7}, # Sum >= 0.7
        {'type': 'ineq', 'fun': lambda w: 1.3 - np.sum(w)}, # Sum <= 1.3
    ]
    
    # Bounds: 0 <= w <= 0.25 (as per project specs)
    bounds = [(0.0, 0.25) for _ in range(num_assets)]
    
    # Initial Guess: Equal weight
    init_guess = np.ones(num_assets) / num_assets
    
    # Matrix alignment for numpy math
    univ_exp_vals = universe_exposures.values
    targ_exp_vals = target_exposures.values
    cov_vals = factor_cov_matrix.values
    spec_vals = specific_variances.values
    
    result = minimize(
        objective_tracking_error,
        init_guess,
        args=(targ_exp_vals, univ_exp_vals, cov_vals, spec_vals),
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': False, 'maxiter': 100}
    )
    
    if not result.success:
        print(f"Warning: Stage 1 Optimization failed: {result.message}")
        return pd.Series(0, index=universe_exposures.index)

    # --- STAGE 2: Cardinality Constraint (Pick Top N) ---
    
    full_weights = pd.Series(result.x, index=universe_exposures.index)
    
    # Sort by weight and pick top N
    top_tickers = full_weights.sort_values(ascending=False).head(max_positions).index
    
    # Subset data for re-optimization
    subset_univ = universe_exposures.loc[top_tickers]
    subset_spec = specific_variances.loc[top_tickers]
    num_subset = len(top_tickers)
    
    # Re-run optimization on just these N stocks
    # Update bounds and constraints for the smaller set
    subset_bounds = [(0.0, 0.25) for _ in range(num_subset)]
    subset_init = np.ones(num_subset) / num_subset
    
    # Same constraints (Sum >= 0.7, Sum <= 1.3)
    subset_cons = [
        {'type': 'ineq', 'fun': lambda w: np.sum(w) - 0.7},
        {'type': 'ineq', 'fun': lambda w: 1.3 - np.sum(w)},
    ]
    
    result_stage2 = minimize(
        objective_tracking_error,
        subset_init,
        args=(targ_exp_vals, subset_univ.values, cov_vals, subset_spec.values),
        method='SLSQP',
        bounds=subset_bounds,
        constraints=subset_cons
    )
    
    # Construct final Series
    final_weights = pd.Series(0.0, index=universe_exposures.index)
    final_weights.loc[top_tickers] = result_stage2.x
    
    return final_weights