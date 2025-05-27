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
QUERY_CLASSIFICATION_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1"

def QueryUnderstandingTool(query: str) -> Dict[str, Any]:
    """Classify query as preprocessing, visualization, or analytics.
    Returns a dictionary with:
    - intent (str): 'preprocessing', 'visualization', or 'analytics'
    - is_chartjs (bool): Whether to use Chart.js for visualization
    - needs_clarification (bool): Whether query needs clarification
    - clarification_question (str | None): Question to ask user if clarification needed
    """
    # No separate caching here as it's called frequently and should be very fast.
    
    # First check for explicit backend requests
    query_lower = query.lower()
    if "using matplotlib" in query_lower:
        return {
            "intent": "visualization",
            "is_chartjs": False,
            "needs_clarification": False,
            "clarification_question": None
        }
    if "using chart.js" in query_lower or "using chartjs" in query_lower:
        return {
            "intent": "visualization",
            "is_chartjs": True,
            "needs_clarification": False,
            "clarification_question": None
        }
    
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
            
            intent_from_llm = response.choices[0].message.content.strip().lower()
            valid_intents = ["preprocessing", "visualization", "chartjs", "analytics"]
            if intent_from_llm not in valid_intents:
                intent = "analytics"  # Default to analytics for unclear queries
            else:
                intent = intent_from_llm
            
            is_chartjs = intent == "chartjs"
            if is_chartjs:
                intent = "visualization"  # Normalize "chartjs" intent to "visualization"
        except Exception as e:
            # St.error not available here, will add logging later
            print(f"Error in QueryUnderstandingTool LLM fallback: {e}. Defaulting intent.")
            intent = "analytics"  # Default to analytics on error
            is_chartjs = False
            
            # If LLM failed, the query might need clarification
            return {
                "intent": intent,
                "is_chartjs": is_chartjs,
                "needs_clarification": True,
                "clarification_question": "I'm not sure how to process your query. Could you rephrase it or be more specific?"
            }

    # Check if query needs clarification based on intent and patterns
    needs_clarification = False
    clarification_question = None
    
    # For preprocessing queries, check if column or method is specified
    if intent == "preprocessing":
        if not any(col_pattern in query_lower for col_pattern in ["column", "columns", "field", "fields"]):
            needs_clarification = True
            clarification_question = "Which column(s) would you like to preprocess?"
    
    # For visualization queries, check if data to visualize is specified
    elif intent == "visualization" and not any(col_pattern in query_lower for col_pattern in ["of", "for", "between", "vs", "versus"]):
        needs_clarification = True
        clarification_question = "What data would you like to visualize?"

    return {
        "intent": intent,
        "is_chartjs": is_chartjs,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question
    }

# Note: The original QueryUnderstandingTool in tools/query_understanding.py which returned a Dict
# has been replaced. If other functions in this file depended on it, they might break.
# However, the user's request is to align with new.py.