"""
tests/test_data_cleaning.py
"""
import pandas as pd
from adv_hedging.data.cleaning import clean_wiki_data

def test_veralto_fix():
    # Create a fake bad dataframe
    bad_df = pd.DataFrame([{
        'ticker': 'VLTO',
        'URL': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', # Bad URL
        'content': 'Generic list content...'
    }])
    
    cleaned_df = clean_wiki_data(bad_df)
    
    # Assert the URL was fixed to the specific company page
    assert cleaned_df.loc[0, 'URL'] == "https://en.wikipedia.org/wiki/Veralto"