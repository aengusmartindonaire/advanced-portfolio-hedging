import pandas as pd
from adv_hedging.config import RAW_DATA_DIR, INTERIM_DATA_DIR, WIKI_PARQUET_FILE, BLOOMBERG_EXCEL_FILE
from adv_hedging.data.cleaning import clean_wiki_data # We will define this next

def load_wiki_data(force_clean=False):
    """
    Loads the Wikipedia embeddings data.
    Checks 'data/interim' first for a cleaned version. 
    If not found, loads raw, cleans it, saves it, and returns it.
    """
    clean_path = INTERIM_DATA_DIR / "wiki_embeddings_cleaned.parquet"
    
    if clean_path.exists() and not force_clean:
        return pd.read_parquet(clean_path)
    
    print("Loading raw data and applying fixes (Veralto, UMB, etc.)...")
    if not WIKI_PARQUET_FILE.exists():
        raise FileNotFoundError(f"Raw file not found at {WIKI_PARQUET_FILE}. Please put it in data/raw/")
        
    df_raw = pd.read_parquet(WIKI_PARQUET_FILE)
    
    # Apply the cleaning logic (defined in cleaning.py)
    df_clean = clean_wiki_data(df_raw)
    
    # Save for next time
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(clean_path)
    
    return df_clean

def load_risk_factors():
    """Loads the Bloomberg Excel file."""
    if not BLOOMBERG_EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file not found at {BLOOMBERG_EXCEL_FILE}")
    
    # Skip the first 2 rows as per notebook instructions
    return pd.read_excel(BLOOMBERG_EXCEL_FILE, skiprows=2)