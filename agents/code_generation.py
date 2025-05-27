import os
from openai import OpenAI
import pandas as pd
import numpy as np # For ExecutionAgent
import matplotlib.pyplot as plt # For ExecutionAgent
import seaborn as sns # For ExecutionAgent
import io # For ExecutionAgent
import hashlib # For ExecutionAgent
import streamlit as st # For st.error
from typing import Dict, List, Any
import json # For ReasoningAgent if it handles JSON output
import re # For extracting code

# Absolute imports from other modules in the project
from utils.cache import get_df_hash, get_cached_result, cache_result
from tools.preprocessing import PreprocessingTool # For ExecutionAgent
from tools.code_writing import CodeWritingTool, PreprocessingCodeGeneratorTool
from tools.visualization import PlotCodeGeneratorTool, ChartJSCodeGeneratorTool
from utils.helpers import extract_first_code_block

# === Configuration ===
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

# Choose appropriate models for different agent tasks
REASONING_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1" # Faster for reasoning/planning
CODE_GEN_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1" # More capable for code generation

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# === Reasoning Agent ===
def ReasoningAgent(df_cols: List[str], user_query: str, conversation_history: List[Dict[str,str]]) -> Dict[str, Any]:
    """Determines the next step or tool to use based on the query and history."""
    cache_key = f"reasoning_{get_df_hash(pd.DataFrame(columns=df_cols))}_{user_query}_{json.dumps(conversation_history[-3:])}" # Cache based on recent history
    cached_plan, _ = get_cached_result(cache_key)
    if cached_plan and isinstance(cached_plan, dict):
        return cached_plan

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])

    prompt = f"""User query: \"{user_query}\"
Available columns: {df_cols}
Conversation history (last 5 turns):
{history_str}

Based on the query and conversation, decide the next action. 
Possible actions:
1.  `call_tool`: If a specific tool can address the query (e.g., `PreprocessingTool`, `PlotCodeGeneratorTool`, `CodeWritingTool`, `ChartJSCodeGeneratorTool`, `DataInsightAgent`, `MissingValueAgent`, `CombinedAnalysisAgent`).
2.  `clarify`: If the query is ambiguous or needs more information.
3.  `answer_directly`: If the query can be answered from context or general knowledge (rare for this app).

Output a JSON object:
{{
  "action": "call_tool" | "clarify" | "answer_directly",
  "tool_name": "<tool_function_name>" (if action is call_tool, e.g., "PlotCodeGeneratorTool"),
  "tool_params": {{...}} (parameters for the tool, derived from query and context),
  "clarification_question": "<question_to_user>" (if action is clarify),
  "direct_answer": "<answer>" (if action is answer_directly),
  "explanation": "Brief reason for this decision (1-2 sentences)."
}}

Example - Query: "Show me a histogram of age"
{{"action": "call_tool", "tool_name": "PlotCodeGeneratorTool", "tool_params": {{"plot_description": "Histogram of age", "chart_type": "histogram"}}, "explanation": "User asked for a histogram, which PlotCodeGeneratorTool can create."}}

Example - Query: "Clean the data"
{{"action": "clarify", "clarification_question": "Could you be more specific about how you'd like to clean the data? For example, what columns are you concerned about, and what kind of cleaning (e.g., missing values, outliers)?", "explanation": "Query is too vague for direct action."}}

Return ONLY the JSON object.
"""
    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL_NAME,
            messages=[{"role": "system", "content": "You are an AI assistant that plans the next step to answer a user's data analysis query."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=700,
            response_format={"type": "json_object"} 
        )
        plan_str = response.choices[0].message.content.strip()
        plan = json.loads(plan_str)
        cache_result(cache_key, prompt, plan)
        return plan
    except Exception as e:
        st.error(f"Reasoning Agent Error: {e}")
        fallback_plan = {"action": "clarify", "clarification_question": "I had trouble understanding your request. Could you please rephrase?", "explanation": f"Error during reasoning: {e}"}
        cache_result(cache_key, prompt, fallback_plan)
        return fallback_plan

# === Code Generation Agent ===
def CodeGenerationAgent(df_cols: List[str], user_query: str, intent_details: Dict[str, Any]) -> str:
    """Generates code based on a specific intent and parameters (wrapper around tools)."""
    intent = intent_details.get("intent", "unknown")
    
    cache_key_parts = ["codegen", intent, get_df_hash(pd.DataFrame(columns=df_cols)), user_query]
    if intent == "visualize":
        cache_key_parts.append(intent_details.get("chart_type"))
        cache_key_parts.append(intent_details.get("plot_description"))
    elif intent == "preprocess":
        cache_key_parts.append(json.dumps(intent_details.get("preprocess_params"), sort_keys=True))
    cache_key_query = "_".join(filter(None, cache_key_parts))

    cached_code, _ = get_cached_result(cache_key_query)
    if cached_code:
        return cached_code

    generated_code = "# No code generated or intent not actionable for direct code generation."

    if intent == "visualize":
        plot_desc = intent_details.get("plot_description", user_query)
        chart_type = intent_details.get("chart_type")
        if chart_type and "chartjs" in chart_type.lower():
            generated_code = f"# Chart.js visualization requested for: {plot_desc}. Use ChartJSCodeGeneratorTool directly."
        else:
            # Ensure df is passed correctly if PlotCodeGeneratorTool expects a DataFrame instance
            # For now, assuming it can work with just column names for prompt generation as per original structure
            generated_code = PlotCodeGeneratorTool(pd.DataFrame(columns=df_cols), plot_desc, chart_type)
    elif intent == "preprocess":
        params = intent_details.get("preprocess_params")
        if params:
            generated_code = PreprocessingCodeGeneratorTool(df_cols, params)
        else:
            generated_code = "# Preprocessing requested but no parameters found."
    elif intent == "analyze":
        generated_code = CodeWritingTool(df_cols, user_query, intent="analyze")
    else:
        generated_code = f"# Intent '{intent}' not directly handled by CodeGenerationAgent for Python code generation."

    cache_result(cache_key_query, f"Intent: {intent}, Details: {intent_details}", generated_code)
    return generated_code

# === Execution Agent ===
def ExecutionAgent(code: str, df: pd.DataFrame, query_for_cache: str) -> Any:
    """Executes generated Python code in a controlled environment."""
    df_h = get_df_hash(df) if df is not None else "no_df"
    code_hash = hashlib.md5(code.encode()).hexdigest()
    exec_cache_key = f"exec_result_{df_h}_{code_hash}"

    cached_code_output, cached_result_val = get_cached_result(exec_cache_key)
    if cached_code_output == code: 
        st.info("Retrieved cached execution result.")
        return cached_result_val

    result_val = None
    error_message = None
    full_env = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "io": io,
        "df": df.copy() if df is not None else None, 
        "PreprocessingTool": PreprocessingTool,
        "result": None 
    }
    try:
        clean_code = extract_first_code_block(code)
        if not clean_code: 
            clean_code = code

        exec_globals = {}
        exec_globals.update(full_env)

        exec(clean_code, exec_globals)
        
        result_val = exec_globals.get("result")
        
        if "fig" in exec_globals and isinstance(exec_globals["fig"], plt.Figure):
            fig = exec_globals["fig"]
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            result_val = buf 
            plt.close(fig) 
        
        elif result_val is None and "df_processed" in exec_globals and isinstance(exec_globals["df_processed"], pd.DataFrame):
            result_val = exec_globals["df_processed"]
        elif result_val is None and "df" in exec_globals and df is not None and not df.equals(full_env["df"]):
            result_val = exec_globals["df"]

        if result_val is not None:
            cache_result(exec_cache_key, code, result_val)
        else:
            cache_result(exec_cache_key, code, "# Execution produced no explicit 'result' or 'fig'.")
            result_val = "Code executed, but no specific output variable 'result' or Matplotlib 'fig' was found."
            
    except Exception as e:
        code_snippet = code[:500] + ('...' if len(code) > 500 else '')
        error_message = f"Execution Error: {str(e)}\n--- Code ---\n{code_snippet}"
        st.error(error_message)
        cache_result(exec_cache_key, code, error_message)
        result_val = error_message 
    finally:
        plt.close('all') 
        
    return result_val