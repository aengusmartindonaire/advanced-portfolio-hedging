"""
src/adv_hedging/utils.py
Utilities for caching and logging.
"""
import os
from joblib import Memory
from adv_hedging.config import DATA_DIR

# Create a cache directory inside data/
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Initialize joblib memory
# verbose=0 means it won't print every time it reads from cache
memory = Memory(location=CACHE_DIR, verbose=0)

def get_cache_decorator():
    """
    Returns the @memory.cache decorator to wrap expensive functions.
    Usage:
        from adv_hedging.utils import memory
        @memory.cache
        def expensive_func(x): ...
    """
    return memory.cache