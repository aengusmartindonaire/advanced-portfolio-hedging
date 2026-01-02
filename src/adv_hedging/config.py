import os
from pathlib import Path

# Get the project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Filenames (match exactly what you uploaded)
WIKI_PARQUET_FILE = RAW_DATA_DIR / "20250930_stk_wiki_em.parquet"
BLOOMBERG_EXCEL_FILE = RAW_DATA_DIR / "20250928_US_Port.xlsx"