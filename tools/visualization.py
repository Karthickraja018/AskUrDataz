import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from typing import List, Dict, Any
import os
from utils.cache import get_df_hash, get_cached_result, cache_result
import re
import matplotlib.pyplot as plt
import seaborn as sns

# === Configuration ===
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

# Select appropriate model based on availability and performance needs
FAST_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1" # Faster, for suggestions
ADVANCED_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1" # More capable, for code generation

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# === Visualization Suggestion Tool ===
def VisualizationSuggestionTool(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Suggest visualizations based on dataset characteristics."""
    suggestions = []
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns

    # Basic suggestions based on data types
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
    if len(num_cols) >= 2:
        suggestions.append({
            "query": f"Show scatter plot of {num_cols[0]} vs {num_cols[1]}",
            "desc": f"Scatter plot showing relationship between '{num_cols[0]}' and '{num_cols[1]}'."
        })
    if len(date_cols) > 0 and len(num_cols) > 0:
        suggestions.append({
            "query": f"Show line chart of {num_cols[0]} over {date_cols[0]}",
            "desc": f"Line chart showing trend of '{num_cols[0]}' over time."
        })

    return suggestions[:4]  # Return at most 4 suggestions

# === Plot Code Generation Tool ===
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

# === Chart.js Code Generation Tool ===
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