"""
tests/conftest.py
Pytest fixtures for generating mock data.
"""
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def mock_returns_df():
    """Generates random returns for 20 assets over 100 days."""
    dates = pd.date_range(start='2024-01-01', periods=100)
    tickers = [f"STOCK_{i}" for i in range(20)]
    data = np.random.normal(0, 0.01, size=(100, 20))
    return pd.DataFrame(data, index=dates, columns=tickers)

@pytest.fixture
def mock_wiki_row():
    """Generates a single fake row of Wikipedia data."""
    return {
        'ticker': 'TEST',
        'title': 'Test Company',
        'URL': 'http://wiki.test/Test_Company',
        'sector': 'Technology',
        'content': 'Word ' * 1000  # 1000 words of dummy text
    }