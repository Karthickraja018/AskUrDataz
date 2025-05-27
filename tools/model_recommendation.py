import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from utils.cache import get_df_hash, get_cached_result, cache_result
from typing import List, Dict

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

FAST_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1"

def ModelRecommendationTool(df: pd.DataFrame, user_query: str, target_variable: str = None) -> List[Dict[str, str]]:
    """Recommends ML models based on data, query, and target variable."""
    dataset_hash_query = f"model_recommend_{get_df_hash(df)}_{user_query}_{target_variable or ''}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, list): return cached_result

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Determine problem type (classification or regression) if target is specified
    problem_type = "unknown"
    if target_variable and target_variable in df.columns:
        if df[target_variable].dtype in ['object', 'category'] or df[target_variable].nunique() < 15: # Heuristic for classification
            problem_type = "classification"
        elif pd.api.types.is_numeric_dtype(df[target_variable]):
            problem_type = "regression"
    
    target_info = f'Target variable: {target_variable} (Problem type: {problem_type})\n' if target_variable else ''
    prompt = f"""User query: "{user_query}"
Dataset characteristics:
Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
{target_info}

Based on the user query and dataset, recommend 3-4 suitable machine learning models (e.g., Linear Regression, Logistic Regression, Decision Tree, Random Forest, SVM, K-Means, etc.).
For each model, provide:
1. `model_name`: The common name of the model.
2. `description`: A brief (1-2 sentences) explanation of why it might be suitable for this problem/query.
3. `use_case`: A typical use case or type of problem it solves (e.g., regression, classification, clustering).

Format the output as a JSON list of objects.
Example: 
[{{"model_name": "Linear Regression", "description": "Good for predicting continuous values when there is a linear relationship.", "use_case": "Regression"}}]
Return ONLY the JSON list.
"""
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME,
            messages=[{"role": "system", "content": "You are an expert in machine learning model recommendation."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500 
        )
        recommendations_str = response.choices[0].message.content.strip()
        
        # LLM might return a string that is not perfectly a list, attempt to parse
        try:
            recommendations = eval(recommendations_str) # eval can be risky, but useful if LLM returns Python list string
            if not isinstance(recommendations, list):
                # Fallback if eval doesn't produce a list (e.g. if it's a string representation of a list)
                recommendations = [{"model_name": "Error", "description": "Could not parse recommendations correctly.", "use_case": "Error"}]
        except:
             recommendations = [{"model_name": "Error", "description": "Failed to parse LLM response for model recommendations.", "use_case": "Error"}]

        cache_result(dataset_hash_query, prompt, recommendations)
        return recommendations
    except Exception as e:
        st.error(f"Error recommending models: {e}")
        # Cache the error information
        error_info = [{"model_name": "Error", "description": str(e), "use_case": "Error"}]
        cache_result(dataset_hash_query, prompt, error_info)
        return error_info