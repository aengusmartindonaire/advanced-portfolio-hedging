"""
src/adv_hedging/hedging/core.py
Abstract base class for hedging strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd

class HedgeEngine(ABC):
    """
    Interface for different hedging strategies (Factor vs Embedding).
    """
    
    def __init__(self, universe_data):
        self.universe_data = universe_data
        
    @abstractmethod
    def calculate_hedge(self, target_ticker: str) -> pd.Series:
        """
        Must return a Series of weights for the hedge portfolio.
        Index = Tickers, Values = Weights.
        """
        pass
        
    @abstractmethod
    def get_hedge_rationale(self) -> str:
        """
        Returns a string explaining why this hedge was chosen.
        """
        pass