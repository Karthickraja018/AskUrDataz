import streamlit as st
from openai import OpenAI
from typing import Tuple, Dict, List, Union, Any
import os
import pandas as pd
import json
import re
from utils.cache import get_df_hash, get_cached_result, cache_result

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
QUERY_CLASSIFICATION_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

def QueryUnderstandingTool(df: pd.DataFrame, query: str) -> Dict[str, Any]:
    """Analyzes user query to extract intent, parameters, and suggest actions."""
    dataset_hash_query = f"query_understanding_{get_df_hash(df)}_{query}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict): return cached_result

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    all_cols = df.columns.tolist()

    prompt = f"""Analyze the user query: \"{query}\" in the context of a dataset with these columns:
Numeric: {numeric_cols}
Categorical: {categorical_cols}
Datetime: {datetime_cols}
All Columns: {all_cols}

Determine the user's intent and extract relevant parameters. 
Possible intents: 'preprocess', 'visualize', 'analyze', 'model', 'clarify', 'unknown'.

Output a JSON object with the following fields:
- `intent`: (string) The primary intent.
- `chart_type`: (string, optional) If visualize, the type of chart (e.g., 'histogram', 'bar', 'scatter', 'line', 'boxplot', 'heatmap', 'chartjs_bar', 'chartjs_line', etc.). Infer if not explicit.
- `target_columns`: (list of strings, optional) Columns relevant to the query.
- `preprocess_params`: (dict, optional) If preprocess, parameters like `missing_strategy`, `encode_categorical`, `scale_features`, `scaling_strategy`, `one_hot_encode_columns`, `outlier_strategy`, `outlier_columns`, `feature_engineering` (with `polynomial_cols`, `polynomial_degree`, `date_cols`), `datetime_columns`, `imputation_strategy`, `knn_neighbors`, `multi_label_columns`.
- `plot_description`: (string, optional) If visualize, a concise description for generating the plot.
- `analysis_type`: (string, optional) If analyze, type like 'summary', 'correlation', 'missing_values', 'descriptive_stats'.
- `model_params`: (dict, optional) If model, parameters like `target_variable`.
- `needs_clarification`: (boolean) True if the query is too vague or needs more info.
- `clarification_question`: (string, optional) If needs_clarification, a question to ask the user.

Examples:
1. Query: "Impute age with mean and one-hot encode gender"
   {{"intent": "preprocess", "preprocess_params": {{"missing_strategy": "mean", "target_columns": ["age"], "one_hot_encode_columns": ["gender"]}}}}
2. Query: "Show histogram of salary"
   {{"intent": "visualize", "chart_type": "histogram", "target_columns": ["salary"], "plot_description": "Histogram of salary"}}
3. Query: "What is the correlation between price and size?"
   {{"intent": "analyze", "analysis_type": "correlation", "target_columns": ["price", "size"]}}
4. Query: "Tell me about the data"
   {{"intent": "clarify", "needs_clarification": true, "clarification_question": "Could you be more specific about what you'd like to know about the data? For example, are you interested in a summary, missing values, or something else?"}}
5. Query: "Convert joined_date to datetime and extract year"
   {{"intent": "preprocess", "preprocess_params": {{"datetime_columns": ["joined_date"], "feature_engineering": {{"date_cols": ["joined_date"]}}}}}}
6. Query: "Handle missing values in rating using KNN with 3 neighbors"
   {{"intent": "preprocess", "preprocess_params": {{"imputation_strategy": "knn", "knn_neighbors": 3, "target_columns": ["rating"], "missing_strategy": "knn"}}}}
7. Query: "Multi-label encode the genres column, it's comma-separated"
   {{"intent": "preprocess", "preprocess_params": {{"multi_label_columns": ["genres"]}}}}

Return ONLY the JSON object.
"""
    try:
        response = client.chat.completions.create(
            model=QUERY_CLASSIFICATION_MODEL_NAME,
            messages=[{"role": "system", "content": "You are an expert at understanding data analysis queries and extracting structured information."},
                      {"role": "user", "content": prompt}],
            temperature=0.05, # Low temperature for deterministic output
            max_tokens=1000, # Allow for detailed JSON
            response_format={"type": "json_object"} # If API supports, for direct JSON output
        )
        
        content = response.choices[0].message.content.strip()
        
        # The API should ideally return valid JSON with response_format. If not, try to parse.
        parsed_result = json.loads(content)
        
        # Basic validation of intent
        if "intent" not in parsed_result or parsed_result["intent"] not in ['preprocess', 'visualize', 'analyze', 'model', 'clarify', 'unknown']:
            parsed_result["intent"] = "unknown"
            parsed_result["needs_clarification"] = True
            if "clarification_question" not in parsed_result:
                 parsed_result["clarification_question"] = "I'm not sure I understood your request. Could you please rephrase?"

        cache_result(dataset_hash_query, prompt, parsed_result)
        return parsed_result
    except json.JSONDecodeError as e_json:
        # Attempt to extract JSON from markdown if parsing fails directly
        match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if match:
            try:
                parsed_result = json.loads(match.group(1))
                # Basic validation of intent
                if "intent" not in parsed_result or parsed_result["intent"] not in ['preprocess', 'visualize', 'analyze', 'model', 'clarify', 'unknown']:
                    parsed_result["intent"] = "unknown"
                cache_result(dataset_hash_query, prompt, parsed_result)
                return parsed_result
            except json.JSONDecodeError as e_json_markdown:
                st.error(f"Error decoding JSON from query understanding (even after markdown extraction): {e_json_markdown}. Response: {content[:500]}")
        else:
            st.error(f"Error decoding JSON from query understanding: {e_json}. Response: {content[:500]}")
        
        error_res = {"intent": "unknown", "needs_clarification": True, "clarification_question": "There was an issue understanding your query. Please try rephrasing."}
        cache_result(dataset_hash_query, prompt, error_res)
        return error_res
    except Exception as e_gen:
        st.error(f"Error in QueryUnderstandingTool: {e_gen}")
        error_res = {"intent": "unknown", "needs_clarification": True, "clarification_question": "An unexpected error occurred while processing your query."}
        cache_result(dataset_hash_query, prompt, error_res)
        return error_res