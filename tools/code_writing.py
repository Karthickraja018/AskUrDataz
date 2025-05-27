import os
from openai import OpenAI
import pandas as pd
import streamlit as st # For st.error
import re
from typing import List, Dict, Tuple, Any
import json
import pandas as pd

from utils.cache import get_df_hash, get_cached_result, cache_result # Absolute import

# === Configuration ===
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

# Use a capable model for code generation
ADVANCED_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# === Generic Code Writing Tool ===
def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for pandas-only code."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas to answer: "{query}"
    
    Rules:
    1. Use pandas operations on `df` only (no plotting libraries).
    2. For missing values queries: use df.isnull().sum(), df.info(), or create summary DataFrames.
    3. For duplicates queries: use df.duplicated().sum(), df[df.duplicated()], or df.drop_duplicates().
    4. For statistical queries: use df.describe(), df.nunique(), df.value_counts(), etc.
    5. For data info queries: use df.shape, df.dtypes, df.columns, etc.
    6. Assign the final result to `result` variable.
    7. If creating a summary, make it a DataFrame or Series for better display.
    8. Return inside a single ```python fence.
    
    Examples:
    - For "show missing values": result = df.isnull().sum().to_frame(name='Missing_Count')
    - For "check duplicates": result = f"Total duplicates: {{df.duplicated().sum()}}"
    - For "data summary": result = df.describe()
    """

# === Preprocessing Code Generation Tool ===
def PreprocessingCodeGeneratorTool(cols: List[str], query: str) -> Tuple[str, Dict]:
    """Generate preprocessing parameters and code from query."""
    params = {
        "missing_strategy": "None", 
        "encode_categorical": False, 
        "scale_features": False, 
        "target_columns": None,
        "scaling_strategy": "standard", 
        "constant_value_impute": None,
        "one_hot_encode_columns": None,
        "outlier_strategy": None,
        "outlier_columns": None,
        "feature_engineering": None
    }
    query_lower = query.lower()

    impute_match = re.match(r"impute column '([^']+)' with (mean|median|mode|constant|forward_fill|backward_fill)(?: \(value: (.+)\))?", query_lower)
    encode_match = re.match(r"(label_encoding|one_hot_encoding) for column '([^']+)'", query_lower)
    scale_match = re.match(r"(standard_scaling|min_max_scaling|robust_scaling) for columns: (.+)", query_lower)
    outlier_match = re.match(r"apply (remove|cap) outlier handling to columns: (.+)", query_lower)
    poly_match = re.match(r"add polynomial features \(degree (\d+)\) for columns: (.+)", query_lower)
    date_match = re.match(r"extract date components \(year, month, day\) from columns: (.+)", query_lower)

    action_taken = False
    if impute_match:
        action_taken = True
        params["target_columns"] = [impute_match.group(1)]
        strategy = impute_match.group(2)
        params["missing_strategy"] = strategy
        if strategy == "constant" and impute_match.group(3):
            params["constant_value_impute"] = impute_match.group(3)
    elif encode_match:
        action_taken = True
        strategy = encode_match.group(1)
        column = encode_match.group(2)
        params["target_columns"] = [column]
        if strategy == "label_encoding":
            params["encode_categorical"] = True
        elif strategy == "one_hot_encoding":
            params["one_hot_encode_columns"] = [column]
            params["encode_categorical"] = True
    elif scale_match:
        action_taken = True
        strategy = scale_match.group(1)
        columns_str = scale_match.group(2)
        params["target_columns"] = [col.strip() for col in columns_str.split(',')]
        params["scale_features"] = True
        params["scaling_strategy"] = strategy.replace("_scaling", "")
    elif outlier_match:
        action_taken = True
        strategy = outlier_match.group(1)
        columns_str = outlier_match.group(2)
        params["outlier_strategy"] = strategy
        params["outlier_columns"] = [col.strip() for col in columns_str.split(',')]
    elif poly_match:
        action_taken = True
        degree = int(poly_match.group(1))
        columns_str = poly_match.group(2)
        params["feature_engineering"] = {
            "polynomial_cols": [col.strip() for col in columns_str.split(',')],
            "polynomial_degree": degree
        }
    elif date_match:
        action_taken = True
        columns_str = date_match.group(1)
        params["feature_engineering"] = {
            "date_cols": [col.strip() for col in columns_str.split(',')]
        }
    else:
        # Fallback to NLP-based parameter detection
        if "impute" in query_lower:
            action_taken = True
            if "mean" in query_lower: params["missing_strategy"] = "mean"
            elif "median" in query_lower: params["missing_strategy"] = "median"
            elif "most frequent" in query_lower or "mode" in query_lower: params["missing_strategy"] = "most_frequent"
            elif "forward fill" in query_lower or "ffill" in query_lower: params["missing_strategy"] = "forward_fill"
            elif "backward fill" in query_lower or "bfill" in query_lower: params["missing_strategy"] = "backward_fill"
        if "encode" in query_lower or "categorical" in query_lower:
            action_taken = True
            params["encode_categorical"] = True
        if "scale" in query_lower or "normalize" in query_lower:
            action_taken = True
            params["scale_features"] = True

    # If no specific preprocessing action identified, return empty code
    if not action_taken:
        return "", params

    # Construct the call to PreprocessingTool
    code_lines = [
        "df_processed = PreprocessingTool(",
        "    df=df,",
        f"    missing_strategy='{params['missing_strategy']}',",
        f"    encode_categorical={params['encode_categorical']},",
        f"    scale_features={params['scale_features']},",
        f"    target_columns={params['target_columns']},",
        f"    scaling_strategy='{params['scaling_strategy']}',",
        f"    constant_value_impute={repr(params['constant_value_impute'])},",
        f"    one_hot_encode_columns={params['one_hot_encode_columns']},",
        f"    outlier_strategy={repr(params['outlier_strategy'])},",
        f"    outlier_columns={params['outlier_columns']},",
        f"    feature_engineering={params['feature_engineering']}",
        ")",
        "result = df_processed"
    ]
    final_code = "\n".join(code_lines)
    
    return final_code, params