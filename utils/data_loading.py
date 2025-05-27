import pandas as pd
import streamlit as st

# === Helpers ===
@st.cache_data # Cache the data loading
def load_data(uploaded_file) -> pd.DataFrame | None:
    """Loads data from an uploaded file (CSV or Excel) into a pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# === DataFrameSummaryTool ===
def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate summary prompt for LLM.
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
    """
    prompt = f"""
    Given a dataset with {len(df)} rows and {len(df.columns)} columns:
    Columns: {', '.join(df.columns)}
    Data types: {df.dtypes.to_dict()}
    Missing values: {df.isnull().sum().to_dict()}
    Provide:
    1. A brief description of what this dataset contains (1-2 sentences).
    2. 3-4 possible data analysis questions that could be asked of this dataset.
    Keep it very concise and focused. Return ONLY the text, no JSON, no markdown.
    """
    return prompt

# === MissingValueSummaryTool ===
def MissingValueSummaryTool(df: pd.DataFrame) -> str:
    """Generate missing value summary prompt for LLM."""
    missing = df.isnull().sum().to_dict()
    total_missing = sum(missing.values())
    prompt = f"""
    Dataset: {len(df)} rows, {len(df.columns)} columns
    Missing values: {missing}
    Total missing: {total_missing}
    Provide a brief summary (2-3 sentences) of missing values and their potential impact on analysis.
    Return ONLY the textual summary. No JSON, no markdown.
    """
    return prompt
