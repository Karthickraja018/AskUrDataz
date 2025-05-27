import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple
from openai import OpenAI
from utils.cache import get_df_hash, get_cached_result, cache_result
import re

def ExecutionAgent(code: str, df: pd.DataFrame, intent: str, is_chartjs: bool, query: str) -> Any:
    """Execute generated code and handle the result based on intent."""
    try:
        local_vars = {"df": df}
        exec(code, globals(), local_vars)
        result = local_vars.get("result", None)
        
        if result is None:
            return "Error: Code execution produced no result"
        
        if intent == "visualization":
            if is_chartjs and isinstance(result, dict) and "type" in result:
                return result
            elif not is_chartjs and isinstance(result, (plt.Figure, plt.Axes)):
                return result
            return "Error: Invalid visualization result"
        
        if intent == "preprocessing" and not isinstance(result, pd.DataFrame):
            return "Error: Preprocessing did not return a DataFrame"
        
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
ADVANCED_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1"
FAST_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1"

def ReasoningCurator(query: str, result: Any) -> str:
    """Generate a natural language explanation of the result based on the query."""
    result_str = str(result) if not isinstance(result, str) else result
    key_points = []
    
    if isinstance(result, pd.DataFrame):
        key_points.extend([
            f"The result contains {result.shape[0]} rows and {result.shape[1]} columns",
            f"Columns included: {', '.join(result.columns)}",
            "This shows a structured view of the data"
        ])
    elif isinstance(result, pd.Series):
        key_points.extend([
            f"The result is a series of {len(result)} values",
            f"Series name: {result.name or 'unnamed'}",
            f"Data type: {result.dtype}"
        ])
    elif isinstance(result, (int, float)):
        key_points.extend([
            f"The result is a numeric value: {result}",
            "This represents a single calculated metric"
        ])
    elif isinstance(result, dict):
        if "type" in result:
            key_points.extend([
                f"Generated a {result['type']} chart configuration",
                "The visualization will help understand the data patterns"
            ])
        else:
            key_points.extend([
                f"The result contains {len(result)} key-value pairs",
                "This provides a structured summary of the analysis"
            ])
    elif "error" in result_str.lower():
        key_points.extend([
            "The operation encountered an error",
            f"Error details: {result_str}"
        ])
    else:
        key_points.extend([
            "The result provides a text explanation",
            f"Length of explanation: {len(result_str)} characters"
        ])
    
    explanation = []
    explanation.append(f"Based on your query: \"{query}\"")
    explanation.append("\nKey observations:")
    for point in key_points:
        explanation.append(f"- {point}")
    
    return "\n".join(explanation)

def ReasoningAgent(query: str, result: Any) -> Tuple[str, str]:
    """Generate reasoning about a query result using LLM."""
    cache_key_query = f"reasoning_{get_df_hash(pd.DataFrame())}_{query}_{str(result)[:100]}"
    cached_prompt, cached_result = get_cached_result(cache_key_query)
    if cached_result and isinstance(cached_result, str): return cached_prompt, cached_result
    
    prompt = ReasoningCurator(query, result)
    
    try:
        response = client.chat.completions.create(
            model=FAST_MODEL_NAME,
            messages=[{"role": "system", "content": "You are a data analysis assistant that explains results clearly and concisely."},
                      {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        reasoning = response.choices[0].message.content.strip()
        cache_result(cache_key_query, prompt, reasoning)
        return prompt, reasoning
    except Exception as e:
        error_msg = f"Error generating reasoning: {e}"
        cache_result(cache_key_query, prompt, error_msg)
        return prompt, error_msg

def CodeGenerationAgent(query: str, df: pd.DataFrame) -> Tuple[str, str, str, Any]:
    """Generate and execute code based on user query."""
    cache_key_query = f"code_gen_{get_df_hash(df)}_{query}"
    cached_prompt, cached_result = get_cached_result(cache_key_query)
    if cached_result and isinstance(cached_result, tuple): return cached_result
    
    prompt = f"""Given a pandas DataFrame `df` with columns: {df.columns.tolist()}
User query: "{query}"

Write Python code to answer the query. Rules:
1. Use pandas operations on `df`.
2. For missing values: use df.isnull().sum(), df.info(), or create summary DataFrames.
3. For duplicates: use df.duplicated().sum(), df[df.duplicated()], or df.drop_duplicates().
4. For statistics: use df.describe(), df.nunique(), df.value_counts(), etc.
5. For data info: use df.shape, df.dtypes, df.columns, etc.
6. Assign final result to `result` variable.
7. Return inside a single ```python fence.

Examples:
- For "show missing values": result = df.isnull().sum().to_frame(name='Missing_Count')
- For "check duplicates": result = f"Total duplicates: {{df.duplicated().sum()}}"
- For "data summary": result = df.describe()
"""
    
    try:
        response = client.chat.completions.create(
            model=ADVANCED_MODEL_NAME,
            messages=[{"role": "system", "content": "You are a Python code generation assistant specializing in pandas."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )
        code_block = response.choices[0].message.content.strip()
        
        match = re.search(r'```python\n(.*?)\n```', code_block, re.DOTALL)
        if match:
            generated_code = match.group(1).strip()
        else:
            generated_code = code_block
        
        try:
            local_vars = {"df": df}
            exec(generated_code, globals(), local_vars)
            result = local_vars.get("result", None)
            success_msg = "Code executed successfully."
        except Exception as e:
            result = f"Error executing code: {e}"
            success_msg = f"Code execution failed: {e}"
        
        cache_result(cache_key_query, prompt, (generated_code, prompt, success_msg, result))
        return generated_code, prompt, success_msg, result
    except Exception as e:
        error_msg = f"Error generating code: {e}"
        cache_result(cache_key_query, prompt, ("", prompt, error_msg, None))
        return "", prompt, error_msg, None 