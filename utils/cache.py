import sqlite3
import hashlib
import pandas as pd
import json
from typing import Tuple, Any
import streamlit as st # For st.cache_data

# === Caching ===
# Function to create a hash for a DataFrame based on its content
@st.cache_data # Cache the hash generation itself
def get_df_hash(df: pd.DataFrame) -> str:
    """Generates a SHA256 hash for a DataFrame based on its content."""
    if df is None:
        return "empty_df"
    # Using a sample of the dataframe can be faster for very large dataframes,
    # but for full accuracy, hash the whole dataframe.
    # For performance with large DFs, consider df.sample(frac=0.1).to_csv() or similar
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def init_cache():
    conn = sqlite3.connect("cache.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS cache (
        query_hash TEXT PRIMARY KEY,
        code TEXT,
        result TEXT
    )""")
    conn.commit()
    return conn

def cache_result(query: str, code: str, result: str):
    conn = init_cache()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    c = conn.cursor()
    # Ensure result is stored as a string, potentially JSON for complex objects
    if not isinstance(result, str):
        try:
            result_str = json.dumps(result)
        except TypeError: # Handle non-serializable objects if necessary
            result_str = str(result)
    else:
        result_str = result
    c.execute("INSERT OR REPLACE INTO cache (query_hash, code, result) VALUES (?, ?, ?)",
              (query_hash, code, result_str))
    conn.commit()
    conn.close()

def get_cached_result(query: str) -> Tuple[str, Any]: # Changed result type to Any
    conn = init_cache()
    query_hash = hashlib.md5(query.encode()).hexdigest()
    c = conn.cursor()
    c.execute("SELECT code, result FROM cache WHERE query_hash = ?", (query_hash,))
    res_tuple = c.fetchone()
    conn.close()
    if res_tuple:
        code, result_str = res_tuple
        try:
            # Attempt to parse result string as JSON
            result_obj = json.loads(result_str)
            return code, result_obj
        except json.JSONDecodeError:
            # If not JSON, return as string (original behavior for simple strings)
            return code, result_str
        except TypeError: # If result_str is None or not a string
             return code, None
    return None, None
