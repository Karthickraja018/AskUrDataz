import os
from openai import OpenAI
import pandas as pd
import streamlit as st
import json
import re
from typing import Dict, List, Any

# Absolute imports
from utils.cache import get_df_hash, get_cached_result, cache_result
from utils.data_loading import DataFrameSummaryTool, MissingValueSummaryTool

# === Configuration ===
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    # Fallback or error if not set, depending on desired behavior
    NVIDIA_API_KEY = "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" 

# Model for generating insights and summaries
INSIGHT_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
FAST_INSIGHT_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1" # For simpler summaries

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# === Data Insight Agent ===
def DataInsightAgent(df: pd.DataFrame, query: str) -> str:
    """Generates textual insights about the data based on a specific query."""
    dataset_hash_query = f"data_insight_{get_df_hash(df)}_{query}"
    cached_insight, _ = get_cached_result(dataset_hash_query)
    if cached_insight: 
        return cached_insight

    # Basic context for the LLM
    cols = df.columns.tolist()
    num_rows, num_cols = df.shape
    # Sample data for context, ensure it is not too large for the prompt
    sample_data_str = df.head(3).to_string() if not df.empty else "Dataset is empty."

    prompt = f"""User query: \"{query}\"
Dataset context:
- Columns: {cols}
- Shape: {num_rows} rows, {num_cols} columns
- Sample data (first 3 rows):
{sample_data_str}

Based on the user query and dataset context, provide a concise textual insight (2-4 sentences).
Focus on directly answering the query if possible, or providing relevant observations.
Return ONLY the textual insight. No markdown, no JSON.
"""
    try:
        response = client.chat.completions.create(
            model=INSIGHT_MODEL_NAME, # Use a capable model for nuanced insights
            messages=[{"role": "system", "content": "You are a data analyst providing insights based on data context and user queries."},
                      {"role": "user", "content": prompt}],
            temperature=0.3, 
            max_tokens=300 
        )
        insight = response.choices[0].message.content.strip()
        cache_result(dataset_hash_query, prompt, insight)
        return insight
    except Exception as e:
        st.error(f"Error generating data insight: {e}")
        error_msg = f"Could not generate insight for query '{query}' due to an error."
        cache_result(dataset_hash_query, prompt, error_msg)
        return error_msg

# === Missing Value Agent ===
def MissingValueAgent(df: pd.DataFrame) -> Dict[str, Any]:
    """Provides a summary and suggestions for missing values."""
    dataset_hash_query = f"missing_value_analysis_{get_df_hash(df)}"
    cached_analysis, _ = get_cached_result(dataset_hash_query)
    if cached_analysis and isinstance(cached_analysis, dict):
        return cached_analysis

    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df)) * 100
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing_count': missing_summary,
        'missing_percentage': missing_pct
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_count', ascending=False)

    if missing_df.empty:
        result = {"summary": "No missing values found in the dataset.", "suggestions": [], "dataframe": None}
        cache_result(dataset_hash_query, "No missing values", result)
        return result

    # Use MissingValueSummaryTool for prompt generation if preferred for consistency
    # llm_prompt = MissingValueSummaryTool(df)
    # Or build prompt directly:
    llm_prompt = f"""Dataset Missing Value Analysis:
Total rows: {len(df)}
Columns with missing values:
{missing_df.to_string()}

Provide:
1. A brief textual summary (2-3 sentences) of the missing data situation.
2. A list of 2-3 actionable suggestions for handling these missing values (e.g., "Impute 'column_name' with mean due to low missing percentage (X%)", "Consider dropping 'column_name' if missing > Y% and not critical", "Use mode for categorical column 'cat_col_name'").

Format the output as a JSON object with keys "summary" (string) and "suggestions" (list of strings).
Example: {{"summary": "The dataset has significant missing data in columns X and Y...", "suggestions": ["Impute X with median.", "Drop Y due to high missingness."]}}
Return ONLY the JSON object.
"""
    try:
        response = client.chat.completions.create(
            model=FAST_INSIGHT_MODEL_NAME, # Faster model for structured summary
            messages=[{"role": "system", "content": "You provide summaries and suggestions for missing data in a dataset."},
                      {"role": "user", "content": llm_prompt}],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"} # Request JSON output if API supports
        )
        content = response.choices[0].message.content.strip()
        
        # Try to parse JSON directly, or from markdown block
        try:
            analysis_result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if match:
                analysis_result = json.loads(match.group(1))
            else:
                raise # Reraise if no JSON found
        
        # Add the missing_df to the result for display
        analysis_result['dataframe'] = missing_df.to_dict('records') # Send as records for easier st.dataframe or st.table
        
        cache_result(dataset_hash_query, llm_prompt, analysis_result)
        return analysis_result
    except Exception as e:
        st.error(f"Error in MissingValueAgent: {e}")
        error_result = {
            "summary": f"Error analyzing missing values: {e}", 
            "suggestions": ["Manual review of missing data is recommended."],
            "dataframe": missing_df.to_dict('records') # Still provide the basic stats
        }
        cache_result(dataset_hash_query, llm_prompt, error_result)
        return error_result

# === Combined Analysis Agent (Initial Load) ===
def CombinedAnalysisAgent(df: pd.DataFrame) -> Dict[str, Any]:
    """Provides an initial overview of the dataset: summary, missing values, and first analysis questions."""
    dataset_hash = get_df_hash(df)
    cache_key = f"combined_analysis_{dataset_hash}"
    cached_data, _ = get_cached_result(cache_key)
    if cached_data and isinstance(cached_data, dict):
        return cached_data

    # Prepare data characteristics for the prompt
    num_rows, num_cols_count = df.shape
    column_names = df.columns.tolist()
    data_types = df.dtypes.apply(lambda x: x.name).to_dict()
    missing_values_summary = df.isnull().sum().to_dict()
    # Only include columns with missing values in this part of the summary
    missing_values_summary = {k: v for k, v in missing_values_summary.items() if v > 0}
    
    # Get a sample of the data (first 3 rows, or fewer if df is small)
    sample_data_head = df.head(min(3, len(df))).to_string()

    prompt = f"""Perform an initial combined analysis of the uploaded dataset.

Dataset Characteristics:
- Shape: {num_rows} rows, {num_cols_count} columns
- Column Names: {column_names}
- Data Types: {data_types}
- Missing Values (counts per column, only if >0): {missing_values_summary if missing_values_summary else 'No missing values'}
- Data Sample (first few rows):
{sample_data_head}

Provide the following in a single JSON object:
1.  `dataset_description`: (string) A brief, 1-2 sentence high-level description of what this dataset might contain based on its structure and column names.
2.  `data_quality_summary`: (string) A 2-3 sentence summary focusing on data quality aspects like completeness (missing values), potential data type issues (e.g., numbers as objects), and any obvious concerns from the sample data.
3.  `missing_value_overview`: (dict) A dictionary with:
    -   `summary_text`: (string) A 1-2 sentence summary about missing values (e.g., "Column X has many missing values, column Y has a few."). If no missing values, state that.
    -   `columns_affected`: (list of strings) List of column names that have missing values.
4.  `suggested_analysis_questions`: (list of strings) 3-4 diverse and insightful data analysis questions that could be asked of this dataset. These should go beyond simple summarization if possible (e.g., relationships, trends, predictions).
5.  `initial_preprocessing_suggestions`: (list of strings) 2-3 high-level preprocessing steps that might be considered based on the initial overview (e.g., "Handle missing values in column X", "Encode categorical column Y", "Scale numerical features if ranges are large").

Example JSON structure:
{{
  "dataset_description": "This dataset appears to be about customer transactions, including product details and purchase amounts.",
  "data_quality_summary": "The data seems mostly complete, though 'purchase_date' is an object and may need conversion. Some missing values in 'promo_code'.",
  "missing_value_overview": {{
    "summary_text": "'promo_code' has some missing entries; other columns are complete.",
    "columns_affected": ["promo_code"]
  }},
  "suggested_analysis_questions": [
    "What is the average purchase amount by product category?",
    "Is there a relationship between customer age and total spending?",
    "Which promotion codes are most effective?"
  ],
  "initial_preprocessing_suggestions": [
    "Convert 'purchase_date' to datetime objects.",
    "Decide on a strategy for missing 'promo_code' values (e.g., impute with 'None', or use mode)."
  ]
}}

Return ONLY the JSON object.
"""
    try:
        response = client.chat.completions.create(
            model=INSIGHT_MODEL_NAME, # Use capable model for this comprehensive analysis
            messages=[{"role": "system", "content": "You are an AI data analyst providing a comprehensive initial overview of a dataset."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, 
            max_tokens=1500, # Increased token limit for detailed JSON
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content.strip()
        
        # LLM should return JSON directly due to response_format, but add fallback parsing
        try:
            analysis = json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract from markdown block if any
            match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
            if match:
                analysis = json.loads(match.group(1))
            else:
                st.error(f"CombinedAnalysisAgent: Failed to parse JSON response. Raw: {response_content[:300]}")
                raise # Re-raise to be caught by outer try-except

        # Basic validation of expected top-level keys
        expected_keys = ["dataset_description", "data_quality_summary", "missing_value_overview", "suggested_analysis_questions", "initial_preprocessing_suggestions"]
        if not all(key in analysis for key in expected_keys):
            st.warning("CombinedAnalysisAgent: LLM response missing some expected keys.")
            # Fill missing keys with default/error messages if necessary
            for k in expected_keys: analysis.setdefault(k, "Info not generated.")
            if "missing_value_overview" not in analysis or not isinstance(analysis["missing_value_overview"], dict):
                analysis["missing_value_overview"] = {"summary_text": "Info not generated.", "columns_affected": []}

        cache_result(cache_key, prompt, analysis)
        return analysis
    except Exception as e:
        st.error(f"Error in CombinedAnalysisAgent: {e}")
        # Construct a default error response structure
        error_analysis = {
            "dataset_description": "Error: Could not generate dataset description.",
            "data_quality_summary": f"Error during analysis: {e}",
            "missing_value_overview": {"summary_text": "Error fetching missing value info.", "columns_affected": []},
            "suggested_analysis_questions": ["Could not generate questions due to error."],
            "initial_preprocessing_suggestions": ["Review data manually due to analysis error."]
        }
        cache_result(cache_key, prompt, error_analysis)
        return error_analysis