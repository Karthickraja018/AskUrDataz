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
def VisualizationSuggestionTool(df: pd.DataFrame, user_query: str = None) -> Dict[str, str]:
    """Suggests visualization types based on data and optionally a user query."""
    dataset_hash_query = f"viz_suggestions_{get_df_hash(df)}_{user_query or ''}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict): return cached_result

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    user_query_part = f'User query: "{user_query}"\n' if user_query else ''
    prompt = f"""Given a dataset with:
Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
Datetime columns: {datetime_cols}

{user_query_part}Suggest 3-5 diverse and appropriate visualization types (e.g., histogram, scatter plot, bar chart, line chart, box plot, heatmap). 
For each, provide a brief title and a 1-sentence description of what it would show for this data.
Format as a JSON object where keys are suggested titles and values are descriptions.
Example: {{"Histogram of Age": "Shows the distribution of age.", "Scatter Plot of Salary vs Experience": "Shows the relationship between salary and experience."}}
Return ONLY the JSON object.
"""
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME, 
            messages=[{"role": "system", "content": "You are a helpful assistant that suggests data visualizations."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300 
        )
        suggestions_json_str = response.choices[0].message.content.strip()
        
        # Try to find JSON within ```json ... ``` if present
        match = re.search(r'```json\n(.*?)\n```', suggestions_json_str, re.DOTALL)
        if match:
            suggestions_json_str = match.group(1)

        suggestions = json.loads(suggestions_json_str)
        cache_result(dataset_hash_query, prompt, suggestions) # Cache the dict
        return suggestions
    except Exception as e:
        st.error(f"Error suggesting visualizations: {e}")
        # Cache an empty dict or error message
        cache_result(dataset_hash_query, prompt, {"error": str(e)})
        return {"Error": "Could not generate suggestions."}

# === Plot Code Generation Tool ===
def PlotCodeGeneratorTool(df: pd.DataFrame, plot_description: str, chart_type: str = None) -> str:
    """Generates Python code (Matplotlib/Seaborn) for a requested plot."""
    dataset_hash_query = f"plot_code_{get_df_hash(df)}_{plot_description}_{chart_type or ''}"
    cached_code, _ = get_cached_result(dataset_hash_query) # Result not needed here
    if cached_code: return cached_code

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    chart_type_hint = f'Chart Type Hint: {chart_type}\n' if chart_type else ''
    prompt = f"""Generate Python code using Matplotlib or Seaborn to create a plot based on the following description:
Description: "{plot_description}"
{chart_type_hint}Dataset columns available:
Numeric: {numeric_cols}
Categorical: {categorical_cols}

Instructions:
1. Assume the DataFrame is named `df`.
2. Import necessary libraries (e.g., `import matplotlib.pyplot as plt`, `import seaborn as sns`).
3. Generate complete, runnable code for the plot.
4. Include `plt.xlabel()`, `plt.ylabel()`, and `plt.title()` as appropriate.
5. Crucially, ensure the final plot object is stored in a variable named `fig` (e.g., `fig, ax = plt.subplots()`, or `fig = sns_plot.figure`).
6. If using Seaborn, get the Matplotlib figure using `fig = plot_object.get_figure()` if the plot object itself isn't a Figure.
7. Do not use `plt.show()`. The figure will be handled by the calling environment.
8. Return ONLY the Python code block (e.g., within triple backticks with python).

Example for a histogram of 'age':
Triple backticks with python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='age', kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
fig = plt.gcf() # Get current figure
Triple backticks
"""
    try:
        response = client.chat.completions.create(
            model=ADVANCED_MODEL_NAME, # Use more capable model for code
            messages=[{"role": "system", "content": "You are a Python code generation assistant specializing in Matplotlib and Seaborn plots."},
                      {"role": "user", "content": prompt}],
            temperature=0.1, 
            max_tokens=800 
        )
        code_block = response.choices[0].message.content.strip()
        
        # Extract code from markdown block if present
        match = re.search(r'```python\n(.*?)\n```', code_block, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
        else:
            generated_code = code_block # Assume it's already plain code if no markdown
            
        cache_result(dataset_hash_query, prompt, generated_code)
        return generated_code
    except Exception as e:
        st.error(f"Error generating plot code: {e}")
        cache_result(dataset_hash_query, prompt, f"# Error generating code: {e}")
        return f"# Error generating plot code: {e}"

# === Chart.js Code Generation Tool ===
def ChartJSCodeGeneratorTool(df: pd.DataFrame, plot_description: str, chart_type: str) -> Dict[str, Any]:
    """Generates Chart.js configuration for a requested plot."""
    dataset_hash_query = f"chartjs_code_{get_df_hash(df)}_{plot_description}_{chart_type}"
    cached_config, _ = get_cached_result(dataset_hash_query)
    if cached_config and isinstance(cached_config, dict): return cached_config
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    prompt = f"""Generate a Chart.js configuration object for the following plot request:
Plot Description: "{plot_description}"
Chart Type: "{chart_type}"

Dataset columns available:
Numeric: {numeric_cols}
Categorical: {categorical_cols}
Datetime: {datetime_cols}

Instructions:
1. Based on the plot description and chart type, select appropriate columns from the DataFrame `df` for labels, data, and other relevant properties.
2. You MUST process the `df` (pandas DataFrame) to extract the necessary data for the Chart.js config. 
   For example, if a bar chart of counts per category is requested for column 'category_col':
   `labels = df['category_col'].value_counts().index.tolist()`
   `data_values = df['category_col'].value_counts().values.tolist()`
3. Create a complete Chart.js JSON configuration object. This includes `type`, `data` (with `labels` and `datasets`), and `options`.
4. For `datasets`, ensure `label` and `data` fields are correctly populated. Include `backgroundColor` and `borderColor` with appropriate colors (can be single or list).
5. If it is a time-series or scatter plot, ensure data is formatted as an array of {{x: value, y: value}} objects if appropriate for the chart type.
6. For options, include a responsive and maintainAspectRatio: false setting. Add a title using `options.plugins.title.text` and `options.plugins.title.display = true`.
7. Return ONLY the Chart.js JSON configuration object, enclosed in triple backticks with json.

Example for a bar chart of counts for a column named 'category':
Triple backticks with json
{{
  "type": "bar",
  "data": {{
    "labels": ["Category A", "Category B"], // These would be derived from df['category'].value_counts().index
    "datasets": [{{
      "label": "Count of Category",
      "data": [10, 20], // These would be derived from df['category'].value_counts().values
      "backgroundColor": "rgba(75, 192, 192, 0.2)",
      "borderColor": "rgba(75, 192, 192, 1)",
      "borderWidth": 1
    }}]
  }},
  "options": {{
    "responsive": true,
    "maintainAspectRatio": false,
    "plugins": {{
      "title": {{
        "display": true,
        "text": "Distribution of Categories"
      }}
    }},
    "scales": {{
      "y": {{
        "beginAtZero": true
      }}
    }}
  }}
}}
Triple backticks
"""
    try:
        # This tool is more complex as it needs to generate code to process data AND the JSON config.
        # For now, the LLM is asked to infer data processing. A more robust solution might involve
        # a Python execution step to get data, then a second LLM call for JSON config based on that data.
        response = client.chat.completions.create(
            model=ADVANCED_MODEL_NAME, # Use more capable model for complex JSON and implied data processing
            messages=[{"role": "system", "content": "You are an expert in Chart.js and pandas. Generate Chart.js configuration JSON, including deriving data from a pandas DataFrame `df` as per the user's plot request."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500 # Increased for potentially complex configs
        )
        
        config_json_str = response.choices[0].message.content.strip()
        match = re.search(r'```json\n(.*?)\n```', config_json_str, re.DOTALL)
        if match:
            config_json_str = match.group(1)
        
        chart_js_config = json.loads(config_json_str)
        cache_result(dataset_hash_query, prompt, chart_js_config)
        return chart_js_config
    except Exception as e:
        st.error(f"Error generating Chart.js config: {e}")
        cache_result(dataset_hash_query, prompt, {"error": str(e)})
        return {"error": f"Error generating Chart.js config: {e}"}