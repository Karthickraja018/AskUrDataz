import streamlit as st
import pandas as pd
import os
from openai import OpenAI
from utils.cache import get_df_hash, get_cached_result, cache_result
from typing import List, Dict, Any

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

FAST_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1"

def ModelRecommendationTool(df: pd.DataFrame) -> str:
    """Suggest ML models based on dataset characteristics."""
    # Get basic dataset info
    n_samples = df.shape[0]
    n_features = df.shape[1]
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Build recommendation text
    recommendation = []
    
    # Basic dataset characteristics
    recommendation.append(f"Dataset has {n_samples} samples and {n_features} features.")
    recommendation.append(f"- {len(numeric_cols)} numeric features")
    recommendation.append(f"- {len(categorical_cols)} categorical features")
    
    # Model suggestions based on data characteristics
    recommendation.append("\nRecommended models:")
    
    if len(numeric_cols) > 0:
        recommendation.append("1. Linear Regression")
        recommendation.append("   - Good for numeric prediction tasks")
        recommendation.append("   - Works well with continuous features")
        recommendation.append("   - Easy to interpret")
        
        recommendation.append("\n2. Random Forest")
        recommendation.append("   - Handles both numeric and categorical features")
        recommendation.append("   - Good for complex relationships")
        recommendation.append("   - Less prone to overfitting")
        
        if n_samples >= 1000:
            recommendation.append("\n3. XGBoost")
            recommendation.append("   - High performance on large datasets")
            recommendation.append("   - Good with mixed feature types")
            recommendation.append("   - Handles missing values well")
    
    if len(categorical_cols) > 0:
        recommendation.append("\n4. Decision Tree")
        recommendation.append("   - Excellent for categorical features")
        recommendation.append("   - Easy to interpret")
        recommendation.append("   - No feature scaling needed")
        
        recommendation.append("\n5. Support Vector Machine (SVM)")
        recommendation.append("   - Good for classification tasks")
        recommendation.append("   - Works well with preprocessed categorical data")
        recommendation.append("   - Best for datasets with clear margins between classes")
    
    return "\n".join(recommendation)