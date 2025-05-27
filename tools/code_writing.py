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
def CodeWritingTool(df_columns: List[str], user_query: str, intent: str = "analyze", code_type: str = "python") -> str:
    """Generates Python code for data analysis or manipulation based on user query.
       This is a more generic version compared to PlotCodeGeneratorTool.
    """
    # Create a hash for caching based on columns, query, intent, and code_type
    cache_key_query = f"codewriting_{intent}_{code_type}_{get_df_hash(pd.DataFrame(columns=df_columns))}_{user_query}"
    cached_code, _ = get_cached_result(cache_key_query)
    if cached_code: 
        return cached_code

    prompt = f"""Given a pandas DataFrame named `df` with the following columns: {df_columns}
User query: \"{user_query}\"
Intent: {intent}

Generate {code_type} code to achieve the user's goal.

Instructions:
1. Assume the DataFrame is available as `df`.
2. Import necessary libraries (e.g., `pandas as pd`, `numpy as np`).
3. The code should perform the requested analysis or manipulation.
4. If the intent is 'analyze' or involves calculation, store the final result (e.g., a value, a Series, a DataFrame) in a variable named `result`.
5. If the intent is 'preprocess', the code should modify `df` in place or return a new DataFrame assigned to `df_processed` or similar, which will then be stored in `result`.
6. Return ONLY the {code_type} code block (e.g., within ```python ... ```).

Example (intent: 'analyze', query: "Calculate the average of column 'value'"):
```python
import pandas as pd

# Assuming df is already loaded
average_value = df['value'].mean()
result = average_value
```

Example (intent: 'preprocess', query: "Drop column 'temp_id'"):
```python
import pandas as pd

# Assuming df is already loaded
df_processed = df.drop(columns=['temp_id'])
result = df_processed
```
"""
    try:
        response = client.chat.completions.create(
            model=ADVANCED_MODEL_NAME,
            messages=[{"role": "system", "content": f"You are a {code_type} code generation assistant for pandas DataFrames."},
                      {"role": "user", "content": prompt}],
            temperature=0.05, # Low temperature for more deterministic code
            max_tokens=1024 
        )
        code_block = response.choices[0].message.content.strip()
        
        match = re.search(r'```python\n(.*?)\n```', code_block, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
        else:
            # If no markdown, check if it's just a block of code (heuristic)
            if "import" in code_block or "def" in code_block or "df[" in code_block:
                 generated_code = code_block
            else: # Unsure, wrap it to be safe for exec, or could raise error
                 generated_code = f"# LLM returned non-standard code format:\n# {code_block}"
        
        cache_result(cache_key_query, prompt, generated_code)
        return generated_code
    except Exception as e:
        st.error(f"Error generating code with CodeWritingTool: {e}")
        error_code = f"# Error generating code: {e}"
        cache_result(cache_key_query, prompt, error_code)
        return error_code

# === Preprocessing Code Generation Tool ===
def PreprocessingCodeGeneratorTool(df_columns: List[str], preprocess_params: Dict[str, Any]) -> str:
    """Generates Python code for preprocessing based on structured parameters."""
    # Create a hash for caching based on columns and sorted preprocess_params items
    # Sorting items ensures dict order doesn't change hash
    params_str = json.dumps(preprocess_params, sort_keys=True)
    cache_key_query = f"preprocess_code_{get_df_hash(pd.DataFrame(columns=df_columns))}_{params_str}"
    cached_code, _ = get_cached_result(cache_key_query)
    if cached_code: 
        return cached_code

    # Construct a more detailed prompt for preprocessing
    prompt_parts = ["Generate Python code to preprocess a pandas DataFrame `df`.",
                    f"Available columns: {df_columns}",
                    "Instructions:",
                    "1. Import `pandas as pd` and any necessary sklearn modules.",
                    "2. Apply the following preprocessing steps based on the parameters:"]

    # Add parameter-specific instructions to the prompt
    if preprocess_params.get('missing_strategy') and preprocess_params.get('target_columns'):
        strat = preprocess_params['missing_strategy']
        cols = preprocess_params['target_columns']
        impute_strat = preprocess_params.get('imputation_strategy', 'simple')
        if impute_strat == 'knn':
            k = preprocess_params.get('knn_neighbors', 5)
            prompt_parts.append(f"   - Impute missing values in columns {cols} using KNNImputer (n_neighbors={k}). Remember to import KNNImputer from sklearn.impute.")
        else:
            const_val = preprocess_params.get('constant_value_impute')
            fill_method_map = {
                'mean': f"fillna(df[{cols}].mean()) for numeric, or mode for categoric in {cols}",
                'median': f"fillna(df[{cols}].median()) for numeric, or mode for categoric in {cols}",
                'most_frequent': f"fillna(df[{cols}].mode()[0]) in {cols}",
                'mode': f"fillna(df[{cols}].mode()[0]) in {cols}", # Same as most_frequent
                'constant': f"fillna({const_val}) in {cols}" if const_val is not None else f"fillna(a_suitable_constant) in {cols}",
                'forward_fill': f"ffill() on columns {cols}",
                'backward_fill': f"bfill() on columns {cols}"
            }
            prompt_parts.append(f"   - Handle missing values in columns {cols} using strategy: '{strat}'. ({fill_method_map.get(strat, 'specified strategy')}). Import SimpleImputer if needed for mean/median/mode.")

    if preprocess_params.get('encode_categorical') and (preprocess_params.get('target_columns') or df_columns):
        # If target_columns for encoding is given, use that, else assume all categoricals
        cols_to_encode = preprocess_params.get('target_columns', df_columns)
        prompt_parts.append(f"   - Label encode categorical columns: {cols_to_encode}. Import LabelEncoder from sklearn.preprocessing.")

    if preprocess_params.get('one_hot_encode_columns'):
        ohe_cols = preprocess_params['one_hot_encode_columns']
        prompt_parts.append(f"   - One-hot encode columns: {ohe_cols} using `pd.get_dummies(df, columns={ohe_cols})`.")

    if preprocess_params.get('scale_features') and (preprocess_params.get('target_columns') or df_columns):
        scaling_strat = preprocess_params.get('scaling_strategy', 'standard')
        # If target_columns for scaling is given, use that, else assume all numerics
        cols_to_scale = preprocess_params.get('target_columns', df_columns)
        scaler_map = {'standard': "StandardScaler", 'min_max': "MinMaxScaler", 'robust': "RobustScaler"}
        prompt_parts.append(f"   - Scale numerical features in columns {cols_to_scale} using {scaler_map.get(scaling_strat, 'StandardScaler')}. Import from sklearn.preprocessing.")
    
    if preprocess_params.get('outlier_strategy') and preprocess_params.get('outlier_columns'):
        out_strat = preprocess_params['outlier_strategy']
        out_cols = preprocess_params['outlier_columns']
        prompt_parts.append(f"   - Handle outliers in columns {out_cols} using IQR method ({out_strat} them).")
        
    if preprocess_params.get('datetime_columns'):
        dt_cols = preprocess_params['datetime_columns']
        prompt_parts.append(f"   - Convert columns {dt_cols} to datetime objects using `pd.to_datetime(df[col], errors=\'coerce\')`.")

    if preprocess_params.get('feature_engineering'):
        fe_params = preprocess_params['feature_engineering']
        if fe_params.get('polynomial_cols') and fe_params.get('polynomial_degree'):
            poly_cols = fe_params['polynomial_cols']
            poly_deg = fe_params['polynomial_degree']
            prompt_parts.append(f"   - Create polynomial features (degree {poly_deg}) for columns {poly_cols}. Import PolynomialFeatures.")
        if fe_params.get('date_cols'):
            date_ext_cols = fe_params['date_cols']
            prompt_parts.append(f"   - For datetime columns {date_ext_cols}, extract year, month, and day into new columns (e.g., df['{date_ext_cols[0]}_year'] = df['{date_ext_cols[0]}'].dt.year).")

    if preprocess_params.get('multi_label_columns'):
        ml_cols = preprocess_params['multi_label_columns']
        prompt_parts.append(f"   - Multi-label binarize columns {ml_cols}. Assume values are strings with delimiters like comma or pipe. Import MultiLabelBinarizer.")

    prompt_parts.append("3. The final processed DataFrame should be assigned to a variable `result`.")
    prompt_parts.append("4. Return ONLY the Python code block.")
    
    final_prompt = "\n".join(prompt_parts)

    try:
        response = client.chat.completions.create(
            model=ADVANCED_MODEL_NAME,
            messages=[{"role": "system", "content": "You are a Python code generation assistant specializing in pandas and scikit-learn for data preprocessing."},
                      {"role": "user", "content": final_prompt}],
            temperature=0.0, # Minimal temperature for precise code
            max_tokens=1536 # Increased for potentially longer preprocessing scripts
        )
        code_block = response.choices[0].message.content.strip()
        match = re.search(r'```python\n(.*?)\n```', code_block, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
        else:
            generated_code = code_block # Assume plain code if no markdown
        
        cache_result(cache_key_query, final_prompt, generated_code)
        return generated_code
    except Exception as e:
        st.error(f"Error generating preprocessing code: {e}")
        error_code = f"# Error generating preprocessing code: {e}"
        cache_result(cache_key_query, final_prompt, error_code)
        return error_code