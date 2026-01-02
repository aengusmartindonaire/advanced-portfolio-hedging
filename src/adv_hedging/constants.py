"""
src/adv_hedging/constants.py
Stores static lists and definitions used across the project.
"""

# Defined by 'Dr. Chen' in the project requirements
AI_MAKERS = [
    'NVDA', 'TSM', 'ASML', 'AVGO', 'AMD', 'ANET', 'MU', 'MRVL',
    'ARM', 'SMCI', 'LRCX', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'PLTR', 'CLS', 'VRT', 'ORCL', 'SNOW', 'CRWD', 'WDC', 'STX'
]

AI_USERS = [
    'TSLA', 'PLTR', 'CRWD', 'CRM', 'NOW', 'JPM', 'COF', 'COIN',
    'SHOP', 'APP', 'XYZ', 'SOFI', 'PYPL', 'HOOD', 'UNH', 'V',
    'MA', 'BLK', 'PGR', 'ISRG', 'MRNA', 'GS', 'MS', 'WFC',
    'BAC', 'C', 'USB', 'WMT', 'HD', 'NKE', 'WDAY', 'ADBE',
    'GM', 'F', 'AAPL'
]

# Mapping for the plotting/clustering analysis
AI_CATEGORIES = {ticker: 'Maker' for ticker in AI_MAKERS}
AI_CATEGORIES.update({ticker: 'User' for ticker in AI_USERS})
# Handle overlaps (PLTR, CRWD appear in both, notebook says context matters)
# For now, we will default to 'Both' or handle purely based on the list lookup
overlap = set(AI_MAKERS).intersection(set(AI_USERS))
for ticker in overlap:
    AI_CATEGORIES[ticker] = 'Both'