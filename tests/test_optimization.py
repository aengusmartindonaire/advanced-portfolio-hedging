"""
tests/test_optimization.py
"""
import pandas as pd
import numpy as np
from adv_hedging.hedging.optimization import optimize_hedge_weights

def test_cardinality_constraint():
    # Setup: 20 assets, target is asset 0
    assets = [f"S_{i}" for i in range(20)]
    
    # Fake exposures (random)
    universe_exposures = pd.DataFrame(
        np.random.randn(20, 3), 
        index=assets, 
        columns=['Size', 'Value', 'Mom']
    )
    target_exp = universe_exposures.iloc[0] * -1 # Target is opposite
    
    # Fake Covariance
    cov = pd.DataFrame(np.eye(3), index=['Size', 'Value', 'Mom'], columns=['Size', 'Value', 'Mom'])
    spec_risk = pd.Series(0.1, index=assets)
    
    # Run optimization with max 5 positions
    weights = optimize_hedge_weights(
        target_exp, universe_exposures, cov, spec_risk, max_positions=5
    )
    
    # Count non-zero weights (allow for floating point noise)
    non_zero = (weights > 1e-4).sum()
    
    assert non_zero <= 5
    assert non_zero > 0 # Should have bought something