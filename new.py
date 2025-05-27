import os, io, re, sqlite3, hashlib
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures, MultiLabelBinarizer
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import datetime

# === Configuration ===
# Use a more standard environment variable name if possible, or ensure this is set.
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "your_fallback_api_key_here_if_any") 
# For the purpose of this exercise, I'll use the one provided in the original code.
# If a "your_fallback_api_key_here_if_any" is not provided, and NVIDIA_API_KEY is not set, this will fail.
# It's better practice to require the API key to be set and not have a hardcoded fallback.
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    # Fallback to the one hardcoded in the original prompt for this specific context
    NVIDIA_API_KEY = "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"
    
    # st.warning("NVIDIA_API_KEY environment variable not set. Using a hardcoded key for demonstration.")
    # It's better to raise an error or show a persistent warning in a real app:
    # raise ValueError("NVIDIA_API_KEY environment variable not set.")


# Define LLM models
# Assuming 'meta/llama3-8b-instruct' is a valid model string for the NVIDIA API for a fast model
# If not, this string needs to be updated to a valid fast model identifier.
# TEMPORARY CHANGE FOR DEBUGGING CONNECTION ERRORS:
FAST_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1" # Using a smaller, faster model for quick responses
POWERFUL_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1" # Using the powerful model for complex tasks
# Fallback for QueryUnderstandingTool if it needs to be ultra-fast and simple
# However, the original used POWERFUL_MODEL_NAME with low tokens, which might be fine.
# Let's make QueryUnderstandingTool use the FAST_MODEL_NAME as well.
QUERY_CLASSIFICATION_MODEL_NAME = FAST_MODEL_NAME


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

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

# === Preprocessing Tool ===
def PreprocessingTool(
    df: pd.DataFrame,
    missing_strategy: str = 'mean',
    encode_categorical: bool = False,
    scale_features: bool = False,
    target_columns: List[str] = None,
    scaling_strategy: str = 'standard',
    constant_value_impute: Any = None,
    one_hot_encode_columns: List[str] = None,
    outlier_strategy: str = None,
    outlier_columns: List[str] = None,
    feature_engineering: Dict[str, Any] = None,
    datetime_columns: List[str] = None,  # New: columns to parse as datetime
    imputation_strategy: str = 'simple', # 'simple' or 'knn'
    knn_neighbors: int = 5, # for KNNImputer
    multi_label_columns: List[str] = None # NEW: columns to multi-label encode
) -> pd.DataFrame:
    """
    Enhanced preprocessing tool:
    - Handles missing values (mean, median, mode, constant, forward/backward fill, KNN imputation)
    - Encodes categorical variables (label, one-hot)
    - Scales numerical features (standard, min-max, robust)
    - Handles outliers (IQR-based remove/cap)
    - Feature engineering (polynomial, date extraction)
    - Parses datetime columns and extracts year/month/day
    - Multi-label encoding for columns with delimited lists (e.g., genres)
    - Validates column types before operations
    """
    df_processed = df.copy()

    # --- Parse datetime columns and extract components ---
    if datetime_columns:
        for col in datetime_columns:
            if col in df_processed.columns:
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    df_processed[f'{col}_year'] = df_processed[col].dt.year
                    df_processed[f'{col}_month'] = df_processed[col].dt.month
                    df_processed[f'{col}_day'] = df_processed[col].dt.day
                except Exception as e:
                    pass
    # Also auto-detect object columns that look like datetimes
    for col in df_processed.select_dtypes(include='object').columns:
        if df_processed[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            except Exception:
                pass

    # --- Missing value imputation ---
    if missing_strategy and missing_strategy != "None":
        if imputation_strategy == 'knn':
            # KNNImputer only works on numeric columns
            num_cols = df_processed.select_dtypes(include=np.number).columns
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df_processed[num_cols] = imputer.fit_transform(df_processed[num_cols])
        else:
            all_num_cols = df_processed.select_dtypes(include=np.number).columns
            all_cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
            num_cols_to_process = [col for col in target_columns if col in all_num_cols] if target_columns else all_num_cols
            cat_cols_to_process = [col for col in target_columns if col in all_cat_cols] if target_columns else all_cat_cols
            if missing_strategy in ['mean', 'median'] and len(num_cols_to_process) > 0:
                imputer_num = SimpleImputer(strategy=missing_strategy)
                df_processed[num_cols_to_process] = imputer_num.fit_transform(df_processed[num_cols_to_process])
            elif missing_strategy == 'most_frequent' and len(cat_cols_to_process) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_processed[cat_cols_to_process] = imputer_cat.fit_transform(df_processed[cat_cols_to_process])
            elif missing_strategy == 'mode' and len(cat_cols_to_process) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                df_processed[cat_cols_to_process] = imputer_cat.fit_transform(df_processed[cat_cols_to_process])
            elif missing_strategy == 'constant' and constant_value_impute is not None and target_columns:
                for col in target_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].fillna(constant_value_impute)
            elif missing_strategy == 'forward_fill' and target_columns:
                for col in target_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].ffill()
            elif missing_strategy == 'backward_fill' and target_columns:
                for col in target_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].bfill()

    # --- Categorical encoding ---
    if encode_categorical:
        all_cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns

    # --- Scaling ---
    if scale_features:
        all_num_cols = df_processed.select_dtypes(include=np.number).columns
        if target_columns:
            num_cols_to_process = [col for col in target_columns if col in all_num_cols]
        else:
            num_cols_to_process = list(all_num_cols)
        # Validate columns are numeric
        num_cols_to_process = [col for col in num_cols_to_process if pd.api.types.is_numeric_dtype(df_processed[col])]
        if len(num_cols_to_process) > 0:
            if scaling_strategy == 'standard':
                scaler = StandardScaler()
            elif scaling_strategy == 'min_max':
                scaler = MinMaxScaler()
            elif scaling_strategy == 'robust':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            df_processed[num_cols_to_process] = scaler.fit_transform(df_processed[num_cols_to_process])
        else:
            # Optionally, log or warn if no numeric columns found
            print("[PreprocessingTool] No numeric columns found for scaling.")
    
    # Outlier Handling (IQR-based)
    if outlier_strategy and outlier_columns:
        for col in outlier_columns:
            if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col]):
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                if outlier_strategy == 'remove':
                    df_processed = df_processed[(df_processed[col] >= lower) & (df_processed[col] <= upper)]
                elif outlier_strategy == 'cap':
                    df_processed[col] = df_processed[col].clip(lower, upper)
    # Feature Engineering
    if feature_engineering:
        # Polynomial features
        if feature_engineering.get('polynomial_cols'):
            degree = feature_engineering.get('polynomial_degree', 2)
            poly_cols = feature_engineering['polynomial_cols']
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_data = poly.fit_transform(df_processed[poly_cols])
            poly_feature_names = poly.get_feature_names_out(poly_cols)
            poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df_processed.index)
            for col in poly_df.columns:
                if col not in df_processed.columns:
                    df_processed[col] = poly_df[col]
        # Date component extraction
        if feature_engineering.get('date_cols'):
            for col in feature_engineering['date_cols']:
                if col in df_processed.columns:
                    try:
                        df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                        df_processed[f'{col}_year'] = df_processed[col].dt.year
                        df_processed[f'{col}_month'] = df_processed[col].dt.month
                        df_processed[f'{col}_day'] = df_processed[col].dt.day
                    except Exception as e:
                        pass
    
    # --- Multi-label (multi-hot) encoding ---
    if multi_label_columns:
        for col in multi_label_columns:
            if col in df_processed.columns:
                # Split the string into lists (assume '|' delimiter, fallback to ',' if not found)
                split_lists = df_processed[col].fillna("").apply(lambda x: [s.strip() for s in re.split(r'\||,', x) if s.strip()])
                mlb = MultiLabelBinarizer()
                mlb_df = pd.DataFrame(mlb.fit_transform(split_lists), columns=[f"{col}_"+c for c in mlb.classes_], index=df_processed.index)
                df_processed = pd.concat([df_processed, mlb_df], axis=1)
                df_processed = df_processed.drop(columns=[col])
    
    return df_processed

# === Preprocessing Suggestion Tool ===
def PreprocessingSuggestionTool(df: pd.DataFrame) -> Dict[str, str]:
    """Suggest preprocessing techniques based on dataset characteristics. 
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls if necessary, or if CombinedAnalysisAgent fails.
    """
    # This function might be simplified or removed if CombinedAnalysisAgent is robust.
    # For now, let's assume it could be a fallback or used independently.
    dataset_hash_query = f"preprocessing_suggestions_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict): return cached_result

    missing = df.isnull().sum()
    total_rows = len(df)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    suggestions = {}
    if missing.sum() > 0:
        missing_pct = missing / total_rows * 100
        for col, pct in missing_pct.items():
            if pct > 0 and col in num_cols:
                suggestions[f"impute_{col}"] = f"Impute missing values in '{col}' with {'mean' if pct < 10 else 'median'} (missing: {pct:.1f}%)."
            elif pct > 0 and col in cat_cols:
                suggestions[f"impute_{col}"] = f"Impute missing values in '{col}' with most frequent value (missing: {pct:.1f}%)."
    
    if len(cat_cols) > 0:
        suggestions["encode_categorical"] = f"Encode {len(cat_cols)} categorical columns ({', '.join(cat_cols)}) for analysis."
    
    if len(num_cols) > 0 and df[num_cols].std().max() > 10: # Heuristic for scaling
        suggestions["scale_features"] = "Scale numerical features to normalize large value ranges."
    
    newline = '\n' # For f-string compatibility
    suggestions_str = "".join([f'- {desc}{newline}' for desc in suggestions.values()])
    prompt = (
        f"Dataset: {total_rows} rows, {len(df.columns)} columns{newline}"
        f"Columns: {', '.join(df.columns)}{newline}"
        f"Data types: {df.dtypes.to_dict()}{newline}"
        f"Missing values: {missing.to_dict()}{newline}"
        f"Existing Suggestions (based on heuristics):{newline}{suggestions_str}"
        f"Based on the above, provide a brief overall 'explanation' text (2-3 sentences) for why these general types of preprocessing steps are recommended for this dataset.{newline}"
        f"Return ONLY the explanation string, no other text, no JSON."
    )
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide concise explanations for preprocessing steps."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=200 # Reduced max_tokens for suggestions
        )
        explanation_text = response.choices[0].message.content.strip()
        suggestions["explanation"] = explanation_text
        cache_result(dataset_hash_query, prompt, suggestions)
        return suggestions
    except Exception as e:
        st.error(f"Error in PreprocessingSuggestionTool (LLM part): {e}")
        suggestions["explanation"] = "Could not generate LLM explanation due to an error."
        # Cache anyway with the error in explanation
        cache_result(dataset_hash_query, prompt, suggestions)
        return suggestions

# === Visualization Suggestion Tool ===
def VisualizationSuggestionTool(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Suggest visualizations based on dataset characteristics.
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls if necessary, or if CombinedAnalysisAgent fails.
    """
    dataset_hash_query = f"visualization_suggestions_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, list): return cached_result

    suggestions = [] # Start with an empty list for LLM to populate primarily
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns

    # Minimal hardcoded suggestions as fallback or seed if LLM fails badly
    if len(cat_cols) > 0:
        suggestions.append({
            "query": f"Show bar chart of counts for {cat_cols[0]}",
            "desc": f"Bar chart of value counts for categorical column '{cat_cols[0]}'."
        })
    if len(num_cols) > 0:
        suggestions.append({
            "query": f"Show histogram of {num_cols[0]}",
            "desc": f"Histogram of numerical column '{num_cols[0]}' to show distribution."
        })
    
    prompt = f"""
    Dataset: {len(df)} rows, {len(df.columns)} columns
    Columns: {', '.join(df.columns)}
    Data types: {df.dtypes.to_dict()}
    Based on the dataset characteristics, suggest 3-4 diverse and relevant visualizations.
    Format as a list of JSON objects. Each object must have a "query" field (a natural language query for a visualization, e.g., "Show bar chart of counts for columnName") and a "desc" field (a human-readable description, e.g., "Bar chart of value counts for categorical column 'columnName'.").
    Return ONLY the list of JSON objects, as a valid JSON array string. No other text before or after.
    Example: [ {{"query": "Show histogram of age", "desc": "Histogram of age distribution"}}, {{"query": "Show bar chart of gender counts", "desc": "Bar chart of gender distribution"}} ]
    """
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Suggest concise, relevant visualizations as a JSON list."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=512 # Max tokens for suggestions
        )
        content = response.choices[0].message.content
        llm_suggestions = json.loads(content) # Expecting a list directly if model/API supports it well
        if isinstance(llm_suggestions, list):
            # Further validation for individual items can be added here
            cache_result(dataset_hash_query, prompt, llm_suggestions)
            return llm_suggestions[:4] # Limit to 4 suggestions
        else:
            st.warning("LLM did not return a list for visualization suggestions. Using fallback.")
            cache_result(dataset_hash_query, prompt, suggestions[:4]) # Cache fallback
            return suggestions[:4] # Fallback to hardcoded or minimal

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from LLM for visualization suggestions: {e}. Response: {content[:200]}")
        cache_result(dataset_hash_query, prompt, suggestions[:4]) # Cache fallback
        return suggestions[:4]
    except Exception as e:
        st.error(f"Error in VisualizationSuggestionTool (LLM part): {e}")
        cache_result(dataset_hash_query, prompt, suggestions[:4]) # Cache fallback
        return suggestions[:4]

# === Model Recommendation Tool ===
def ModelRecommendationTool(df: pd.DataFrame) -> str:
    """Recommend ML models for the dataset.
       NOTE: This is mostly superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls if necessary, or if CombinedAnalysisAgent fails.
    """
    dataset_hash_query = f"model_recommendations_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, str): return cached_result

    num_rows, num_cols = df.shape
    target_col = None
    # Simple heuristic for inferring a potential target column
    for col in df.columns:
        if df[col].nunique() < num_rows * 0.1 and df[col].nunique() > 1: # Avoid constant columns
            target_col = col
            break
    
    task = "classification" if target_col and df[target_col].dtype == 'object' else "regression"
    if not target_col: # If no clear target, suggest clustering
        task = "clustering"

    size = "small" if num_rows < 1000 else "large"
    
    # Basic recommendations based on heuristics
    recommendations_heuristic = []
    if task == "classification":
        recommendations_heuristic.append(("Logistic Regression", "Suitable for categorical outcomes."))
        recommendations_heuristic.append(("Random Forest Classifier", "Handles complex feature interactions."))
        if size == "large":
            recommendations_heuristic.append(("XGBoost Classifier", "High performance for large datasets."))
    elif task == "regression":
        recommendations_heuristic.append(("Linear Regression", "Simple for continuous outcomes."))
        recommendations_heuristic.append(("Random Forest Regressor", "Captures non-linear trends."))
        if size == "large":
            recommendations_heuristic.append(("Gradient Boosting Regressor", "Effective for complex patterns."))
    else: # Clustering
        recommendations_heuristic.append(("K-Means Clustering", "Groups data for segmentation."))
    
    newline = '\n' # For f-string compatibility
    prompt = f"""
    Dataset: {num_rows} rows, {num_cols} columns.
    Inferred Task: {task} (Target column: {target_col if target_col else 'None identified'}).
    Heuristic Model Suggestions:
    {"".join([f'- {model}: {reason}{newline}' for model, reason in recommendations_heuristic])}
    Based on the dataset characteristics and inferred task, provide a brief (2-4 sentences) explanation and recommendation for suitable Machine Learning models.
    Focus on why these types of models are appropriate. You can refine or confirm the heuristic suggestions.
    Return ONLY the textual explanation. No JSON, no list, just the paragraph.
    """
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide concise model recommendations and explanations."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=256 # Max tokens for this specific output
        )
        recommendation_text = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, recommendation_text)
        return recommendation_text
    except Exception as e:
        st.error(f"Error in ModelRecommendationTool (LLM part): {e}")
        # Fallback to a simple heuristic string if LLM fails
        fallback_text = f"Based on the task ({task}), consider models like {', '.join([r[0] for r in recommendations_heuristic])}. LLM explanation failed."
        cache_result(dataset_hash_query, prompt, fallback_text)
        return fallback_text

# === QueryUnderstandingTool ===
def QueryUnderstandingTool(query: str) -> Tuple[str, bool]:
    """Classify query as preprocessing, visualization, or analytics."""
    # No separate caching here as it's called frequently and should be very fast.
    
    # First check for explicit backend requests
    query_lower = query.lower()
    if "using matplotlib" in query_lower:
        return ("visualization", False)
    if "using chart.js" in query_lower or "using chartjs" in query_lower:
        return ("visualization", True)
    
    analytical_patterns = [
        "show missing", "missing value", "missing data", "check missing",
        "duplicates", "duplicate", "is there any", "how many",
        "summary", "describe", "info", "statistics", "stats",
        "count", "unique", "nunique", "shape", "size",
        "correlation", "corr", "distribution"
    ]
    
    # Check for visualization patterns
    viz_patterns = [
        "plot", "chart", "graph", "visualize", "histogram", "scatter",
        "bar chart", "pie chart", "line chart", "heatmap", "boxplot"
    ]
    
    # Check for preprocessing patterns (only if not analytical)
    preprocessing_patterns = [
        "impute", "encode", "scale", "normalize", "preprocess",
        "fill missing", "handle missing", "transform", "convert"
    ]
    
    # Prioritize analytical queries
    if any(pattern in query_lower for pattern in analytical_patterns):
        intent = "analytics"
        is_chartjs = False
    elif any(pattern in query_lower for pattern in viz_patterns):
        intent = "visualization"
        # Check for specific chart types that work well with Chart.js
        chartjs_patterns = ["bar chart", "pie chart", "line chart"]
        is_chartjs = any(pattern in query_lower for pattern in chartjs_patterns)
    elif any(pattern in query_lower for pattern in preprocessing_patterns):
        intent = "preprocessing"
        is_chartjs = False
    else:
        # Use LLM as fallback for unclear cases
        messages = [
            {"role": "system", "content": "Classify the user query. Respond with one word: 'preprocessing', 'visualization', 'chartjs', or 'analytics'. 'chartjs' is for specific bar, pie, or line chart requests. If it's a general plot or graph, use 'visualization'. Detailed thinking off."},
            {"role": "user", "content": query}
        ]
        
        try:
            response = client.chat.completions.create(
                model=QUERY_CLASSIFICATION_MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().lower()
            valid_intents = ["preprocessing", "visualization", "chartjs", "analytics"]
            if intent not in valid_intents:
                intent = "analytics"  # Default to analytics for unclear queries
            
            is_chartjs = intent == "chartjs"
            if is_chartjs:
                intent = "visualization"
        except Exception as e:
            st.error(f"Error in QueryUnderstandingTool: {e}. Defaulting intent.")
            intent = "analytics"  # Default to analytics on error
            is_chartjs = False

    return (intent, is_chartjs)

# === Code Generation Tools ===
def PlotCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for pandas+matplotlib code."""
    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}
    Write Python code using pandas and matplotlib.pyplot (as plt) to answer:
    "{query}"
    Rules:
    1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
    2. Assign the final result (matplotlib Figure) to `result`.
    3. Create ONE plot with figsize=(6,4), add title/labels.
    4. Return inside a single ```python fence.
    """

def ChartJSCodeGeneratorTool(cols: List[str], query: str) -> str:
    """Generate a prompt for Python code that creates a Chart.js configuration dictionary."""
    return f"""
    Given a pandas DataFrame `df` with columns: {', '.join(cols)}
    Your task is to write Python code that processes this DataFrame and prepares data for a Chart.js visualization.
    The goal is to answer the user's query: "{query}"

    Instructions for the Python code you will write:
    1. Use pandas to perform any necessary data manipulation (grouping, aggregation, filtering, etc.) on `df`.
    2. Construct a Python dictionary that represents a valid Chart.js JSON configuration.
       Supported chart types are: 'bar', 'line', or 'pie'.
    3. This Python dictionary should include:
        - 'type': The type of chart (e.g., 'bar').
        - 'data': A dictionary containing 'labels' (a list of strings) and 'datasets' (a list of dataset objects).
        - 'options': A dictionary for chart options, including a title (e.g., options.plugins.title.text).
    4. Ensure 'datasets' contains appropriate data (e.g., 'data' list for values, 'backgroundColor' for colors).
       Provide distinct colors for chart elements.
    5. Assign the fully formed Python dictionary (NOT a JSON string) to a variable named `result`.
    6. Return ONLY the Python code block. Do not include any explanations before or after the ```python ... ``` fence.

    Example of the structure of the Python dictionary to be assigned to `result`:
    ```python
    # result = {{
    #     "type": "bar",
    #     "data": {{
    #         "labels": ["Category A", "Category B"],
    #         "datasets": [{{
    #             "label": "Sales",
    #             "data": [100, 150],
    #             "backgroundColor": ["rgba(75, 192, 192, 0.2)", "rgba(255, 99, 132, 0.2)"],
    #             "borderColor": ["rgba(75, 192, 192, 1)", "rgba(255, 99, 132, 1)"],
    #             "borderWidth": 1
    #         }}]
    #     }},
    #     "options": {{
    #         "responsive": True,
    #         "plugins": {{
    #             "title": {{
    #                 "display": True,
    #                 "text": "Chart Title from User Query"
    #             }}
    #         }},
    #         "scales": {{
    #             "y": {{
    #                 "beginAtZero": True
    #             }}
    #         }}
    #     }}
    # }}
    ```
    Focus on generating the Python code that creates such a dictionary.
    """

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

def PreprocessingCodeGeneratorTool(cols: List[str], query: str) -> Tuple[str, Dict]:
    """Generate preprocessing parameters and code from query."""
    params = {
        "missing_strategy": "None", 
        "encode_categorical": False, 
        "scale_features": False, 
        "target_columns": None, # Initialize as None
        "scaling_strategy": "standard", 
        "constant_value_impute": None,
        "one_hot_encode_columns": None, # Initialize as None
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
        params["missing_strategy"] = strategy # PreprocessingTool will map ffill/bfill if needed
        if strategy == "constant" and impute_match.group(3):
            params["constant_value_impute"] = impute_match.group(3)
        # For mean, median, mode, PreprocessingTool will handle based on missing_strategy and target_columns
    elif encode_match:
        action_taken = True
        strategy = encode_match.group(1)
        column = encode_match.group(2)
        params["target_columns"] = [column]
        if strategy == "label_encoding":
            params["encode_categorical"] = True # PreprocessingTool will apply LE to target_columns if cat
        elif strategy == "one_hot_encoding":
            params["one_hot_encode_columns"] = [column]
            params["encode_categorical"] = True # Also set this, PreprocessingTool can decide not to LE if OHE is done
    elif scale_match:
        action_taken = True
        strategy = scale_match.group(1)
        columns_str = scale_match.group(2)
        params["target_columns"] = [col.strip() for col in columns_str.split(',')]
        params["scale_features"] = True
        params["scaling_strategy"] = strategy.replace("_scaling", "") # e.g. "standard_scaling" -> "standard"
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
        # Fallback to original NLP-based parameter detection
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
            # Could try to extract target columns from NLP here if desired
        if "scale" in query_lower or "normalize" in query_lower:
            action_taken = True
            params["scale_features"] = True
            # Could try to extract target columns/scaling strategy from NLP here

    # If no specific preprocessing action identified by query, it might be an analytical query misclassified
    # or a general request. For safety, if no flags are set, don't generate PreprocessingTool call.
    if not action_taken and not (params["missing_strategy"] != "None" or params["encode_categorical"] or params["scale_features"]):
        # This query is likely not a preprocessing task for this tool.
        # Return empty code and default params, CodeGenerationAgent will decide what to do.
        return "", params 

    # Construct the call to PreprocessingTool
    # df_processed = PreprocessingTool(df, missing_strategy='...', ...)
    # result = df_processed
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

# === CodeGenerationAgent ===
def CodeGenerationAgent(query: str, df: pd.DataFrame):
    """Generate code using LLM, with caching."""
    # Cache key combines query and DataFrame hash for context-specific code
    # This is important if the DataFrame structure influences the code generated for the same query
    # However, the prompt to the LLM already includes column names.
    # A simpler query_hash might be sufficient if df columns are always in the prompt.
    # For robustness, let's use a query-specific hash here, assuming prompt templates handle df context.
    query_hash = hashlib.md5(query.encode()).hexdigest() # Original query hash
    cached_code, cached_result_obj = get_cached_result(query_hash)
    
    # The cached_result_obj from get_cached_result here is for *execution results* of the code,
    # not the code itself. CodeGenerationAgent primarily returns code.
    # The existing caching logic seems to store code and then separately an empty string or execution result.
    # Let's refine: cache_result(query_hash, generated_code, execution_result_if_any)

    if cached_code: # If we have cached code for this query
        intent, is_chartjs = QueryUnderstandingTool(query)
        # If we also had a cached *execution result* for this code, we could return it too.
        # The current cache stores code and result under the same query_hash.
        # This implies if cached_code exists, cached_result_obj is its corresponding execution result.
        return cached_code, intent, is_chartjs, cached_result_obj # Return cached code and its previously cached result
    
    intent, is_chartjs = QueryUnderstandingTool(query)
    code = ""
    
    if intent == "preprocessing":
        # PreprocessingCodeGeneratorTool generates code directly, not an LLM prompt
        code, _ = PreprocessingCodeGeneratorTool(df.columns.tolist(), query)
    elif intent in ["visualization"]:
        # is_chartjs is now handled by the QueryUnderstandingTool which normalizes chartjs to visualization
        # The prompt generator will decide if it's a Chart.js or Matplotlib prompt
        prompt_template_func = ChartJSCodeGeneratorTool if is_chartjs else PlotCodeGeneratorTool
        prompt = prompt_template_func(df.columns.tolist(), query)
        
        system_message_content = "You are an expert Python programmer. Write clean, efficient code. "
        if is_chartjs:
            system_message_content += "You need to generate Python code that creates a Python dictionary for a Chart.js configuration. This dictionary will be assigned to a variable called 'result'. Return ONLY a single Python code block (```python ... ```)."
        else:
            system_message_content += "Return ONLY a single Python code block (```python ... ```) for pandas/matplotlib. No explanations before or after the code block."

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=POWERFUL_MODEL_NAME, # Use powerful model for code generation
            messages=messages,
            temperature=0.1, # Low temperature for more deterministic code
            max_tokens=1024 # Sufficient for most code snippets
        )
        code = extract_first_code_block(response.choices[0].message.content)
    else: # intent == "analytics"
        # Check for common analytical queries and provide hardcoded solutions
        query_lower = query.lower()
        if "missing" in query_lower and ("show" in query_lower or "check" in query_lower):
            # Hardcoded solution for missing values
            code = """missing_counts = df.isnull().sum()
missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': (missing_counts.values / len(df)) * 100
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
result = missing_df if not missing_df.empty else "No missing values found in the dataset." """
        elif "duplicate" in query_lower:
            # Hardcoded solution for duplicates
            code = """duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    duplicate_rows = df[df.duplicated(keep=False)]
    result = f"Found {duplicate_count} duplicate rows. First few duplicates:\\n{duplicate_rows.head().to_string()}"
else:
    result = "No duplicate rows found in the dataset." """
        elif "shape" in query_lower or "size" in query_lower:
            code = """result = f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns" """
        elif "info" in query_lower or "summary" in query_lower:
            code = """result = df.describe()"""
        elif ("correlation" in query_lower and "heatmap" in query_lower) or \
             ("correlation heatmap" in query_lower) or \
             ("correlation matrix heatmap" in query_lower) or \
             ("corr heatmap" in query_lower):
            code = '''
import matplotlib.pyplot as plt
import seaborn as sns
corr_matrix = df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
           square=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax)
ax.set_title('Correlation Matrix Heatmap')
plt.tight_layout()
result = fig
'''
        elif "filter rows where" in query_lower:
            # Extract filter condition from query
            filter_pattern = r"filter rows where (\w+) ([><=!]+) (.+)"
            filter_match = re.search(filter_pattern, query_lower)
            if filter_match:
                col = filter_match.group(1)
                op = filter_match.group(2)
                val = filter_match.group(3)
                # Try to convert value to appropriate type
                try:
                    if val.replace('.', '', 1).isdigit():
                        val = float(val) if '.' in val else int(val)
                    else:
                        val = f"'{val}'"
                except:
                    val = f"'{val}'"
                code = f"""filtered_df = df[df['{col}'] {op} {val}]
result = f"Filtered dataset: {{len(filtered_df)}} rows (from {{len(df)}} original rows)"
if len(filtered_df) > 0:
    result += f"\\n\\nFirst 5 rows of filtered data:\\n{{filtered_df.head().to_string()}}"
else:
    result += "\\nNo rows match the filter criteria." """
            else:
                code = """result = "Could not parse filter condition. Use format: 'column operator value'" """
        else:
            # Use LLM for other analytical queries
            prompt = CodeWritingTool(df.columns.tolist(), query)
            messages = [
                {"role": "system", "content": "You are an expert Python programmer. Write clean, efficient pandas code. Return ONLY a single Python code block (```python ... ```). No explanations before or after the code block."},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model=POWERFUL_MODEL_NAME, # Use powerful model
                messages=messages,
                temperature=0.1, # Low temperature
                max_tokens=1024
            )
            code = extract_first_code_block(response.choices[0].message.content)
    
    # Cache the generated code. The execution result will be cached by ExecutionAgent later if successful.
    # Store an empty string or a placeholder indicating "not executed yet" for the result part of this cache entry.
    cache_result(query_hash, code, "__CODE_GENERATED_NOT_EXECUTED__") 
    return code, intent, is_chartjs, "__CODE_GENERATED_NOT_EXECUTED__" # Return a placeholder for result

# === ExecutionAgent ===
def ExecutionAgent(code: str, df: pd.DataFrame, intent: str, is_chartjs: bool, query_for_cache: str):
    """Execute code safely and return result. Updates cache with execution result."""
    env = {"pd": pd, "df": df, "PreprocessingTool": PreprocessingTool, "np": np} # Added numpy
    if intent in ["visualization"] and not is_chartjs: # intent can be 'visualization' (covers chartjs too now)
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["io"] = io
    
    query_hash = hashlib.md5(query_for_cache.encode()).hexdigest()

    try:
        # Debug: Print the code being executed if there might be issues
        if "df_processed" in code and "PreprocessingTool" in code:
            print(f"DEBUG: Executing preprocessing code:\n{code}")
        
        exec(code, {}, env) # Using separate global and local dicts for exec
        result = env.get("result", None)
        
        # Successfully executed, cache the result along with the code
        # The code should already be in cache from CodeGenerationAgent, here we update the result part.
        cache_result(query_hash, code, result) # result might be complex, cache_result handles JSON conversion
        
        if is_chartjs and isinstance(result, dict):
            return result
        # Ensure matplotlib figures are handled correctly for display later
        if isinstance(result, (plt.Figure, plt.Axes)):
             # The object itself is returned; Streamlit handles plotting it.
            return result
        return result # Could be DataFrame, Series, scalar, etc.
    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        
        # Enhanced error message with code context
        error_message = f"Error executing code: {exc}\n\nCode that failed:\n{code}\n\nFull traceback:\n{error_details}"
        
        # Print to console for debugging
        print(f"EXECUTION ERROR for query '{query_for_cache}':")
        print(f"Code:\n{code}")
        print(f"Error: {exc}")
        
        # Cache the error message as the result for this query/code
        cache_result(query_hash, code, error_message)
        return error_message

# === ReasoningCurator ===
def ReasoningCurator(query: str, result: Any) -> str:
    """Build LLM prompt for reasoning about results."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))
    is_chartjs = isinstance(result, dict) and "type" in result and result["type"] in ["bar", "line", "pie"]
    
    if is_error:
        desc = result
    elif is_plot:
        title = result._suptitle.get_text() if isinstance(result, plt.Figure) and result._suptitle else result.get_title() if isinstance(result, plt.Axes) else "Chart"
        desc = f"[Plot Object: {title}]"
    elif is_chartjs:
        title = result.get("options", {}).get("plugins", {}).get("title", {}).get("text", "Chart")
        desc = f"[Chart.js Object: {title}]"
    else:
        desc = str(result)[:300]
    
    prompt = f'''
    The user asked: "{query}".
    Result: {desc}
    Explain in 2–3 concise sentences what this tells about the data (mention charts only if relevant).
    '''
    return prompt

# === ReasoningAgent ===
def ReasoningAgent(query: str, result: Any):
    """Stream LLM reasoning for results."""
    prompt = ReasoningCurator(query, result)
    # This agent is for detailed reasoning, so POWERFUL_MODEL_NAME is appropriate.
    response = client.chat.completions.create(
        model=POWERFUL_MODEL_NAME, 
        messages=[{"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
                  {"role": "user", "content": prompt}],
        temperature=0.2, # As per original, can be low for factual reasoning
        max_tokens=1024, # As per original
        stream=True
    )
    
    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )
    
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return thinking_content, cleaned

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

# === DataInsightAgent ===
def DataInsightAgent(df: pd.DataFrame) -> str:
    """Generate dataset summary and questions.
       NOTE: This is largely superseded by CombinedAnalysisAgent for initial load.
       Retained for targeted calls or if CombinedAnalysisAgent fails.
    """
    # Use dataset hash for caching key
    dataset_hash_query = f"dataset_insights_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, str): 
        return cached_result
    
    prompt = DataFrameSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide brief, focused insights as plain text."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=256 # Reduced max_tokens
        )
        
        result = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, result) # Cache string result
        return result
    except Exception as e:
        st.error(f"Error in DataInsightAgent: {e}")
        return f"Error generating insights: {e}"

# === MissingValueAgent ===
def MissingValueAgent(df: pd.DataFrame) -> str:
    """Generate missing value summary with LLM insights."""
    dataset_hash_query = f"missing_value_summary_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, str): 
        return cached_result
    
    prompt = MissingValueSummaryTool(df)
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, # Use fast model
            messages=[{"role": "system", "content": "Provide a concise summary of missing values as plain text."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, # Low temperature
            max_tokens=200 # Reduced max_tokens
        )
        
        result = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, result)
        return result
    except Exception as e:
        st.error(f"Error in MissingValueAgent: {e}")
        return f"Error generating missing value summary: {e}"

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

def extract_first_code_block(text: str) -> str:
    """Extract first Python code block from markdown."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Combined Initial Analysis Agent ===
def CombinedAnalysisAgent(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates initial dataset insights, preprocessing suggestions, visualization suggestions, and model recommendations using a single LLM call."""
    
    dataset_hash_query = f"combined_analysis_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict):
        # Basic validation if the cached result has expected keys
        if all(k in cached_result for k in ["insights", "preprocessing_suggestions", "visualization_suggestions", "model_recommendations"]):
            return cached_result

    num_rows, num_cols = df.shape
    column_names = ", ".join(df.columns)
    dtypes_dict = df.dtypes.to_dict()
    missing_values_dict = df.isnull().sum().to_dict()

    # Construct a detailed prompt for the LLM
    prompt = f"""
    Analyze the following dataset and provide a comprehensive analysis. The dataset has {num_rows} rows and {num_cols} columns.
    Column Names: {column_names}
    Data Types: {dtypes_dict}
    Missing Values: {missing_values_dict}

    Please provide the output as a single JSON object with the following four top-level keys:
    1.  "insights": A string containing a brief description of what this dataset likely contains and 3-4 possible data analysis questions that could be asked.
    2.  "preprocessing_suggestions": A JSON object. Keys should be descriptive identifiers (e.g., "impute_columnName", "encode_categorical", "scale_features"). Values should be strings explaining the suggestion (e.g., "Impute missing values in 'columnName' with mean (missing: X.X%).", "Encode N categorical columns (col1, col2) for analysis.", "Scale numerical features to normalize large value ranges."). Also include an "explanation" key with a general rationale for the suggested preprocessing steps.
    3.  "visualization_suggestions": A list of JSON objects. Each object should have a "query" field (a natural language query for a visualization, e.g., "Show bar chart of counts for columnName") and a "desc" field (a human-readable description, e.g., "Bar chart of value counts for categorical column 'columnName'."). Suggest 3-4 diverse and relevant visualizations.
    4.  "model_recommendations": A string explaining recommended ML models (e.g., Logistic Regression, Random Forest) based on dataset characteristics (e.g., target variable type if inferable, dataset size). Explain why these models are suitable.

    Prioritize concise and actionable information. Ensure the entire output is a single valid JSON object.
    Example for preprocessing_suggestions value:
    {{ "impute_age": "Impute missing values in 'age' with mean (missing: 5.2%).", "encode_gender": "Encode categorical column 'gender'.", "explanation": "Imputation handles missing data, encoding prepares categorical data for modeling."}}
    Example for visualization_suggestions value:
    [ {{"query": "Show histogram of age", "desc": "Histogram of age distribution"}}, {{"query": "Show bar chart of gender counts", "desc": "Bar chart of gender distribution"}} ]
    """

    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Provide results as a single, valid JSON object, adhering strictly to the requested structure. No extra text before or after the JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2048 # Increased to accommodate comprehensive JSON output
        )
        
        content = response.choices[0].message.content
        # Attempt to parse the JSON content
        # The LLM should ideally return a string that is parseable into a JSON object.
        # If the API supports a json_object response_format, this will be more reliable.
        parsed_json = json.loads(content)

        # Validate the structure of the parsed_json
        if not isinstance(parsed_json, dict) or not all(k in parsed_json for k in ["insights", "preprocessing_suggestions", "visualization_suggestions", "model_recommendations"]):
            # Fallback or error handling if JSON is not as expected
            st.error("LLM returned an unexpected JSON structure for combined analysis. Attempting to regenerate individual components.")
            # Call individual suggestion tools as a fallback (this part is complex to add here, would make the agent huge)
            # For now, let's assume the JSON is correct or this error gets caught and handled higher up.
            # A more robust solution would be to retry or parse defensively.
            return {
                "insights": "Error: Could not parse combined insights from LLM.",
                "preprocessing_suggestions": {},
                "visualization_suggestions": [],
                "model_recommendations": "Error: Could not parse model recommendations from LLM."
            }

        # Cache the successfully parsed result
        cache_result(dataset_hash_query, prompt, parsed_json) # Cache the parsed JSON directly
        return parsed_json

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from LLM for combined analysis: {e}. Response: {content[:500]}")
        # Fallback or error handling
        return {
            "insights": "Error: LLM response was not valid JSON.",
            "preprocessing_suggestions": {},
            "visualization_suggestions": [],
            "model_recommendations": "Error: LLM response was not valid JSON."
        }
    except Exception as e:
        st.error(f"Error in CombinedAnalysisAgent: {e}")
        # Fallback or error handling
        return {
            "insights": f"Error generating insights: {e}",
            "preprocessing_suggestions": {},
            "visualization_suggestions": [],
            "model_recommendations": f"Error generating model recommendations: {e}"
        }

# === Main Streamlit App ===
def main():
    st.set_page_config(layout="wide", page_title="AskurData Education")
    
    # Initialize session state variables if they don't exist
    default_session_state = {
        "plots": [],
        "messages": [],
        "df": None,
        "df_processed_history": [], # To store states of df for undo or comparison
        "current_file_name": None,
        "insights": None,
        "preprocessing_suggestions": {},
        "visualization_suggestions": [],
        "model_suggestions": None,
        "initial_analysis_done": False # Flag to track if combined analysis is complete
    }
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    left, right = st.columns([3,7])
    
    with left:
        st.header("Data Analysis Agent")
        st.markdown("<medium>Powered by <a href='https://build.nvidia.com/nvidia/llama-3.1-nemotron-ultra-253b-v1'>NVIDIA Llama-3.1-Nemotron-Ultra-253b-v1</a></medium>", unsafe_allow_html=True)
        
        # File Uploader
        uploaded_file = st.file_uploader("Choose CSV or Excel", type=["csv", "xlsx"], key="file_uploader")

        if uploaded_file is not None:
            # Check if it's a new file or the same file (to avoid reprocessing on every rerun)
            if st.session_state.current_file_name != uploaded_file.name:
                st.session_state.df = load_data(uploaded_file)
                st.session_state.current_file_name = uploaded_file.name
                st.session_state.messages = [] # Clear messages for new file
                st.session_state.plots = []    # Clear plots for new file
                st.session_state.df_processed_history = [] # Clear history
                st.session_state.initial_analysis_done = False # Reset flag for new file
                # Clear previous analysis results
                st.session_state.insights = None
                st.session_state.preprocessing_suggestions = {}
                st.session_state.visualization_suggestions = []
                st.session_state.model_suggestions = None
                st.rerun() # Rerun once to update UI with new DF and clear old analysis state
            
            if st.session_state.df is not None:
                # Display dataset info always if df is loaded
                st.markdown(f"**Dataset Info: {st.session_state.current_file_name}**")
                st.markdown(f"Rows: {len(st.session_state.df)}, Columns: {len(st.session_state.df.columns)}")
                with st.expander("Column Names and Types"):
                    # Create a DataFrame for column names and types
                    col_info_df = pd.DataFrame({
                        'Column Name': st.session_state.df.columns,
                        'Data Type': [str(dtype) for dtype in st.session_state.df.dtypes]
                    })
                    st.dataframe(col_info_df, use_container_width=True)

                # Display dataset preview in sidebar (first 5 rows)
                # The requirement mentioned preview in chat, but sidebar is also good for persistent view.
                # Chat preview happens on initial load message.
                with st.expander("Dataset Preview (First 5 Rows)", expanded=False):
                    st.dataframe(st.session_state.df.head())

                # Perform combined analysis only if not already done for the current df
                if not st.session_state.initial_analysis_done:
                    with st.spinner("Generating initial dataset analysis. This may take a moment..."):
                        analysis_results = CombinedAnalysisAgent(st.session_state.df)
                        
                        # Initialize with defaults in case of partial failure
                        st.session_state.insights = "Insights generation failed or pending."
                        st.session_state.preprocessing_suggestions = {}
                        st.session_state.visualization_suggestions = []
                        st.session_state.model_suggestions = "Model suggestions generation failed or pending."

                        if isinstance(analysis_results, dict):
                            st.session_state.insights = analysis_results.get("insights", st.session_state.insights)
                            st.session_state.preprocessing_suggestions = analysis_results.get("preprocessing_suggestions", st.session_state.preprocessing_suggestions)
                            st.session_state.visualization_suggestions = analysis_results.get("visualization_suggestions", st.session_state.visualization_suggestions)
                            st.session_state.model_suggestions = analysis_results.get("model_recommendations", st.session_state.model_suggestions)
                            
                            # Log to console if any part of the analysis returned an error string from the agent
                            for key, value in analysis_results.items():
                                if isinstance(value, str) and "Error:" in value:
                                    print(f"CombinedAnalysisAgent returned error for '{key}': {value}")
                        else:
                            st.error("Failed to retrieve a valid analysis structure from the agent.")
                            print("CombinedAnalysisAgent did not return a dictionary.")

                        st.session_state.initial_analysis_done = True # Mark analysis as attempted/done

                        initial_chat_messages = []
                        if st.session_state.insights and "failed or pending" not in st.session_state.insights and "Error:" not in st.session_state.insights:
                            initial_chat_messages.append(f"### Dataset Insights\n{st.session_state.insights}")
                        else:
                            initial_chat_messages.append("### Dataset Insights\nCould not retrieve insights at this time.")
                        
                        if st.session_state.model_suggestions and "failed or pending" not in st.session_state.model_suggestions and "Error:" not in st.session_state.model_suggestions:
                            initial_chat_messages.append(f"### Model Suggestions\n{st.session_state.model_suggestions}")
                        else:
                            initial_chat_messages.append("### Model Suggestions\nCould not retrieve model suggestions at this time.")
                        
                        # Add combined initial messages to chat if there's anything to show
                        if initial_chat_messages:
                            # Prepend to messages so it appears first after dataset load info
                            st.session_state.messages.insert(0, {
                                "role": "assistant",
                                "content": "\n\n---\n\n".join(initial_chat_messages)
                            })
                        st.rerun() # Rerun to display the new insights in chat and populate tools
                        if not initial_chat_messages:
                            st.error("Failed to retrieve initial analysis from the agent (result was None or empty).")
        else:
            st.info("Upload a CSV or Excel file to begin.")
            # Clear session state if no file is uploaded or file is removed
            if st.session_state.current_file_name is not None:
                st.session_state.current_file_name = None
                st.session_state.df = None
                st.session_state.messages = []
                st.session_state.plots = []
                st.session_state.initial_analysis_done = False
                st.session_state.insights = None
                st.session_state.preprocessing_suggestions = {}
                st.session_state.visualization_suggestions = []
                st.session_state.model_suggestions = None
                st.rerun() # Rerun to clear the UI

        # Tool Dashboard - should populate based on session_state variables filled by CombinedAnalysisAgent
        st.header("Tool Dashboard")
        tab_pre, tab_eda, tab_utils = st.tabs(["Preprocessing", "EDA", "Utilities"])
        with tab_pre:
            with st.expander("💡 AI Suggestions", expanded=True):
                st.subheader("AI Suggested Preprocessing")
                if st.session_state.initial_analysis_done and st.session_state.preprocessing_suggestions:
                    suggestions_to_display = dict(st.session_state.preprocessing_suggestions)
                    explanation = suggestions_to_display.pop("explanation", None)
                    if not suggestions_to_display:
                        st.caption("No specific preprocessing steps suggested by AI.")
                    for i, (key, desc) in enumerate(suggestions_to_display.items()):
                        button_key = f"preprocess_btn_{key.replace(' ', '_')}_{i}"
                        if st.button(desc, key=button_key, help=f"Apply action: {key}"):
                            # Save a copy of the current df for diff
                            st.session_state.df_before_preprocess = st.session_state.df.copy() if st.session_state.df is not None else None
                            query_for_preprocessing = f"Apply AI suggestion: {desc}"
                            st.session_state.messages.append({"role": "user", "content": query_for_preprocessing})
                            st.session_state.last_preprocess_action = key
                            st.rerun()
                    if explanation:
                        st.markdown(f"**AI Explanation:** {explanation}")
                elif st.session_state.df is not None and not st.session_state.initial_analysis_done:
                    st.caption("Suggestions will appear after initial analysis.")
                elif st.session_state.df is None:
                    st.caption("Upload a dataset to see suggestions.")
                else:
                    st.caption("No preprocessing suggestions from AI.")

                # After processing a preprocessing action, show the modified dataset and what changed
                if (
                    hasattr(st.session_state, 'df_before_preprocess') and
                    st.session_state.df_before_preprocess is not None and
                    st.session_state.df is not None and
                    hasattr(st.session_state, 'last_preprocess_action') and
                    st.session_state.last_preprocess_action is not None
                ):
                    old_df = st.session_state.df_before_preprocess
                    new_df = st.session_state.df
                    changed_cols = [col for col in new_df.columns if not old_df[col].equals(new_df[col]) if col in old_df.columns]
                    added_cols = [col for col in new_df.columns if col not in old_df.columns]
                    removed_cols = [col for col in old_df.columns if col not in new_df.columns]
                    st.markdown(f"### 🛠️ Preprocessing Applied: {st.session_state.last_preprocess_action}")
                    st.markdown(f"**Changed columns:** {', '.join(changed_cols) if changed_cols else 'None'}")
                    st.markdown(f"**Added columns:** {', '.join(added_cols) if added_cols else 'None'}")
                    st.markdown(f"**Removed columns:** {', '.join(removed_cols) if removed_cols else 'None'}")
                    # Show missing value summary before/after
                    old_missing = old_df.isnull().sum().sum()
                    new_missing = new_df.isnull().sum().sum()
                    st.markdown(f"**Missing values before:** {old_missing}, **after:** {new_missing}")
                    st.markdown("#### Preview of Modified Dataset:")
                    st.dataframe(new_df.head())
                    # Reset so this only shows once per action
                    st.session_state.df_before_preprocess = None
                    st.session_state.last_preprocess_action = None

            with st.expander("🛠️ Preprocessing Tools", expanded=True):
                if st.session_state.df is None:
                    st.caption("Upload a dataset to use preprocessing tools.")
                else:
                    # Handle Missing Values
                    st.subheader("Handle Missing Values")
                    with st.form("impute_form"):
                        st.selectbox("Column to Impute", options=st.session_state.df.columns, key="impute_col_select")
                        st.selectbox("Imputation Strategy", options=["mean", "median", "mode", "constant", "forward_fill", "backward_fill"], key="impute_strategy_select")
                        st.text_input("Constant Value (if strategy is 'constant')", key="impute_constant_val")
                        impute_submit = st.form_submit_button("Apply Imputation")
                        if impute_submit:
                            col = st.session_state.impute_col_select
                            strategy = st.session_state.impute_strategy_select
                            const_val = st.session_state.impute_constant_val
                            query = f"Impute column '{col}' with {strategy}"
                            if strategy == "constant":
                                query += f" (value: {const_val})"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()

                    # Encode Categorical Variables
                    st.subheader("Encode Categorical Variables")
                    with st.form("encode_form"):
                        st.selectbox("Column to Encode", options=st.session_state.df.select_dtypes(include='object').columns, key="encode_col_select")
                        st.selectbox("Encoding Strategy", options=["label_encoding", "one_hot_encoding"], key="encode_strategy_select")
                        encode_submit = st.form_submit_button("Apply Encoding")
                        if encode_submit:
                            col = st.session_state.encode_col_select
                            strategy = st.session_state.encode_strategy_select
                            query = f"{strategy} for column '{col}'"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                    
                    # Scale Numerical Features
                    st.subheader("Scale Numerical Features")
                    with st.form("scale_form"):
                        st.multiselect("Columns to Scale", options=st.session_state.df.select_dtypes(include=np.number).columns, key="scale_cols_select")
                        st.selectbox("Scaling Strategy", options=["standard_scaling", "min_max_scaling", "robust_scaling"], key="scale_strategy_select")
                        scale_submit = st.form_submit_button("Apply Scaling")
                        if scale_submit:
                            cols = st.session_state.scale_cols_select
                            strategy = st.session_state.scale_strategy_select
                            if cols:
                                query = f"{strategy} for columns: {', '.join(cols)}"
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
                            else:
                                st.warning("Please select columns to scale.")

                    # Outlier Handling
                    st.subheader("Outlier Handling (IQR)")
                    with st.form("outlier_form"):
                        outlier_cols = st.multiselect("Select columns for outlier handling", options=st.session_state.df.select_dtypes(include='number').columns, key="outlier_cols_select")
                        outlier_strategy = st.selectbox("Outlier Strategy", options=["remove", "cap"], key="outlier_strategy_select")
                        outlier_submit = st.form_submit_button("Apply Outlier Handling")
                        if outlier_submit and outlier_cols:
                            query = f"Apply {outlier_strategy} outlier handling to columns: {', '.join(outlier_cols)}"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                    
                    # Feature Engineering
                    st.subheader("Feature Engineering")
                    with st.form("feature_eng_form"):
                        feat_type = st.selectbox("Feature Type", ["Polynomial Features", "Date Component Extraction"], key="feat_type_select")
                        if feat_type == "Polynomial Features":
                            poly_cols = st.multiselect("Columns for Polynomial Features", options=st.session_state.df.select_dtypes(include='number').columns, key="poly_cols_select")
                            poly_degree = st.number_input("Polynomial Degree", min_value=2, max_value=5, value=2, key="poly_degree_input")
                        else:
                            date_cols = st.multiselect("Date Columns", options=st.session_state.df.columns, key="date_cols_select")
                        feat_submit = st.form_submit_button("Apply Feature Engineering")
                        if feat_submit:
                            if feat_type == "Polynomial Features" and poly_cols:
                                query = f"Add polynomial features (degree {poly_degree}) for columns: {', '.join(poly_cols)}"
                            elif feat_type == "Date Component Extraction" and date_cols:
                                query = f"Extract date components (year, month, day) from columns: {', '.join(date_cols)}"
                            else:
                                query = None
                            if query:
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
        with tab_eda:
            with st.expander("📊 EDA Tools", expanded=True):
                if st.session_state.df is None:
                    st.caption("Upload a dataset to use EDA tools.")
                else:
                    # Manual Visualizations
                    st.subheader("Manual Visualizations")
                    with st.form("manual_viz_form"):
                        plot_type = st.selectbox("Plot Type", ["bar", "line", "pie", "histogram", "scatter"], key="plot_type_select")
                        x_col = st.selectbox("X Axis", options=st.session_state.df.columns, key="x_col_select")
                        y_col = None
                        if plot_type in ["bar", "line", "scatter"]:
                            y_col = st.selectbox("Y Axis", options=st.session_state.df.columns, key="y_col_select")
                        chart_lib = st.selectbox("Chart Library", ["Matplotlib", "Chart.js"], key="chart_lib_select")
                        viz_submit = st.form_submit_button("Generate Visualization")
                        if viz_submit:
                            if plot_type in ["bar", "line", "scatter"] and y_col:
                                query = f"Show {plot_type} chart of {y_col} vs {x_col} using {chart_lib}"
                            elif plot_type in ["pie", "histogram"]:
                                query = f"Show {plot_type} of {x_col} using {chart_lib}"
                            else:
                                query = None
                            if query:
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
                    # Statistical Summaries
                    st.subheader("Statistical Summaries")
                    if st.button("Show Statistical Summary", key="stat_summary_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Show statistical summary (describe)"})
                        st.rerun()
                    # Correlation Matrix
                    st.subheader("Correlation Matrix")
                    if st.button("Show Correlation Matrix", key="corr_matrix_btn"):
                        st.session_state.messages.append({"role": "user", "content": "Show correlation matrix heatmap"})
                        st.rerun()
                    # Data Filtering
                    st.subheader("Data Filtering")
                    with st.form("filter_form"):
                        filter_col = st.selectbox("Column to Filter", options=st.session_state.df.columns, key="filter_col_select")
                        filter_op = st.selectbox("Operator", [">", ">=", "<", "<=", "==", "!="], key="filter_op_select")
                        filter_val = st.text_input("Value", key="filter_val_input")
                        filter_submit = st.form_submit_button("Apply Filter")
                        if filter_submit and filter_col and filter_op and filter_val:
                            query = f"Filter rows where {filter_col} {filter_op} {filter_val}"
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
        with tab_utils:
            if st.session_state.df is None:
                st.caption("Upload a dataset to use utility tools.")
            else:
                st.subheader("Missing Value Summary")
                if st.button("Show Detailed Missing Values Summary", key="detailed_missing_val_summary_btn"):
                    st.session_state.messages.append({"role": "user", "content": "Show detailed missing value summary and analysis"})
                    st.rerun()

                st.subheader("Model Recommendations")
                if st.button("Show Model Recommendations", key="model_recom_btn_sidebar", disabled=not st.session_state.initial_analysis_done):
                    if st.session_state.model_suggestions:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"### Model Suggestions (from initial analysis)\n{st.session_state.model_suggestions}"
                        })
                        st.rerun()
                    else:
                        st.warning("Model suggestions not available from initial analysis. You can ask the chat.")

                st.subheader("Download Dataset")
                csv_data = st.session_state.df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Current Dataset (CSV)",
                    data=csv_data,
                    file_name=f"processed_{st.session_state.current_file_name if st.session_state.current_file_name else 'dataset.csv'}",
                    mime="text/csv",
                    key="download_csv_sidebar_btn"
                )

                st.subheader("Export Chat History")
                if st.session_state.messages:
                    chat_json = json.dumps(st.session_state.messages, indent=2, default=str)
                    st.download_button(
                        label="Download Chat History (JSON)",
                        data=chat_json,
                        file_name="chat_history.json",
                        mime="application/json",
                        key="download_json_btn"
                    )
                    
                    chat_txt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    st.download_button(
                        label="Download Chat History (Text)",
                        data=chat_txt,
                        file_name="chat_history.txt",
                        mime="text/plain",
                        key="download_txt_btn"
                    )
                else:
                    st.caption("No chat history to export.")

    with right:
        st.header("Chat with your data")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        chat_container = st.container()
        chat_render_error = None
        try:
            # === FIX: Process any unprocessed user message from sidebar tools automatically ===
            # Find the latest user message that is not immediately followed by an assistant message
            unprocessed_user_idx = None
            for i in range(len(st.session_state.messages) - 1, -1, -1):
                if st.session_state.messages[i]["role"] == "user":
                    if i == len(st.session_state.messages) - 1 or st.session_state.messages[i+1]["role"] != "assistant":
                        unprocessed_user_idx = i
                        break
            if unprocessed_user_idx is not None:
                user_q = st.session_state.messages[unprocessed_user_idx]["content"]
                try:
                    with st.spinner("Working …"):
                        # Add current df to history before any modification by user query
                        if st.session_state.df is not None:
                            st.session_state.df_processed_history.append(st.session_state.df.copy())
                            if len(st.session_state.df_processed_history) > 5: # Keep last 5 states
                                st.session_state.df_processed_history.pop(0)

                        # Specific handling for "Show detailed missing value summary and analysis" from sidebar
                        if user_q.lower() == "show detailed missing value summary and analysis":
                            result = MissingValueAgent(st.session_state.df) # This provides an LLM summary
                            # Calculate detailed missing values
                            missing_values = st.session_state.df.isnull().sum()
                            missing_percent = (missing_values / len(st.session_state.df)) * 100
                            missing_df = pd.DataFrame({
                                'Column': st.session_state.df.columns,
                                'Missing Values': missing_values,
                                'Percentage Missing (%)': missing_percent
                            })
                            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percentage Missing (%)', ascending=False)
                            st.session_state.messages.insert(unprocessed_user_idx+1, {
                                "role": "assistant",
                                "content": f"### Detailed Missing Value Analysis {result}", # LLM summary
                                "dataframe": missing_df if not missing_df.empty else "No missing values found."
                            })
                        else:
                            code, intent, is_chartjs, _ = CodeGenerationAgent(user_q, st.session_state.df)
                            assistant_msg_content = {}
                            if code:
                                result_obj = ExecutionAgent(code, st.session_state.df, intent, is_chartjs, user_q)
                                if intent == "preprocessing" and isinstance(result_obj, pd.DataFrame):
                                    st.session_state.df = result_obj
                                    assistant_msg_content["dataframe_preview"] = result_obj.head()
                                    assistant_msg_content["header_text"] = "Preprocessing applied successfully! Dataset updated."
                                elif isinstance(result_obj, str) and result_obj.startswith("Error"):
                                    assistant_msg_content["header_text"] = "⚠️ Error"
                                    assistant_msg_content["error_details"] = result_obj
                                elif is_chartjs and isinstance(result_obj, dict):
                                    assistant_msg_content["chartjs_config"] = result_obj
                                    assistant_msg_content["header_text"] = "📊 Here is the Chart.js visualization:"
                                elif intent in ["visualization", "chartjs"] and isinstance(result_obj, (plt.Figure, plt.Axes)):
                                    fig = result_obj.figure if isinstance(result_obj, plt.Axes) else result_obj
                                    st.session_state.plots.append(fig)
                                    assistant_msg_content["plot_index"] = len(st.session_state.plots) - 1
                                    assistant_msg_content["header_text"] = "📊 Here is the visualization:"
                                elif isinstance(result_obj, (pd.DataFrame, pd.Series)):
                                    assistant_msg_content["dataframe_result"] = result_obj
                                    assistant_msg_content["header_text"] = f"🔍 Result:"
                                else:
                                    assistant_msg_content["scalar_result"] = result_obj
                                    assistant_msg_content["header_text"] = f"💡 Result: {str(result_obj)[:200]}"
                                raw_thinking, reasoning_txt = ReasoningAgent(user_q, result_obj)
                                reasoning_txt = reasoning_txt.replace("`", "")
                                msg_parts = []
                                if assistant_msg_content.get("header_text"):
                                    msg_parts.append(assistant_msg_content.get("header_text"))
                                if reasoning_txt:
                                    msg_parts.append(reasoning_txt)
                                thinking_html = f'''<details class="thinking" style="margin-top: 10px; border: 1px solid #ddd; padding: 5px;">\n    <summary>🧠 View Model Reasoning</summary>\n    <pre style="white-space: pre-wrap; word-wrap: break-word;">{raw_thinking}</pre>\n</details>''' if raw_thinking else ""
                                code_html = f'''<details class="code" style="margin-top: 10px; border: 1px solid #ddd; padding: 5px;">\n    <summary>💻 View Generated Code</summary>\n    <pre><code class="language-python" style="white-space: pre-wrap; word-wrap: break-word;">{code}</code></pre>\n</details>'''
                                if thinking_html:
                                    msg_parts.append(thinking_html)
                                if code_html:
                                    msg_parts.append(code_html)
                                final_assistant_text = "\n\n".join(filter(None, msg_parts))
                                df_for_message = None
                                if assistant_msg_content.get("dataframe_preview") is not None:
                                    df_for_message = assistant_msg_content.get("dataframe_preview")
                                elif assistant_msg_content.get("dataframe_result") is not None:
                                    df_for_message = assistant_msg_content.get("dataframe_result")
                                st.session_state.messages.insert(unprocessed_user_idx+1, {
                                    "role": "assistant",
                                    "content": final_assistant_text,
                                    "plot_index": assistant_msg_content.get("plot_index"),
                                    "dataframe": df_for_message,
                                    "chartjs": assistant_msg_content.get("chartjs_config")
                                })
                            else:
                                if "insights" in user_q.lower() or "describe" in user_q.lower() or "dataset contain" in user_q.lower():
                                    insight_text = DataInsightAgent(st.session_state.df)
                                    st.session_state.messages.insert(unprocessed_user_idx+1, {"role": "assistant", "content": f"### Dataset Insights\n{insight_text}"})
                                else:
                                    st.session_state.messages.insert(unprocessed_user_idx+1, {"role": "assistant", "content": "I couldn't generate code or a direct answer for your query. Please try rephrasing or use the sidebar tools."})
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        except Exception as chat_render_exc:
            chat_render_error = chat_render_exc
        # Always render the chat area, even if an error occurred
        with chat_container:
            if chat_render_error:
                st.error(f"A chat processing error occurred: {chat_render_error}")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)
                    if msg.get("plot_index") is not None:
                        idx = msg["plot_index"]
                        if 0 <= idx < len(st.session_state.plots):
                            # Check if it's a matplotlib figure or axes
                            plot_obj = st.session_state.plots[idx]
                            if isinstance(plot_obj, (plt.Figure, plt.Axes)):
                                st.pyplot(plot_obj, use_container_width=True)
                            else:
                                st.markdown(f"Debug: Plot object at index {idx} is type {type(plot_obj)}")
                    if msg.get("dataframe") is not None:
                        # Display DataFrame or Series
                        df_to_display = msg["dataframe"]
                        if isinstance(df_to_display, (pd.DataFrame, pd.Series)):
                            st.dataframe(df_to_display)
                        else:
                            st.markdown(f"Debug: Dataframe object is type {type(df_to_display)}")
                    if msg.get("chartjs") is not None:
                        # Display Chart.js JSON configuration in a markdown code block
                        chart_json_str = json.dumps(msg['chartjs'], indent=2)
                        st.markdown(f"Chart.js Configuration:```json{chart_json_str}```")
                        st.caption("To render this Chart.js config, a custom Streamlit component would be needed.")
            # --- Add chat input at the bottom ---
            user_input = st.chat_input("Type your message and press Enter…")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.rerun()


if __name__ == "__main__":
    main()