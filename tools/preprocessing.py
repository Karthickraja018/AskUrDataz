import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures, MultiLabelBinarizer
from typing import List, Dict, Any
from utils.cache import get_df_hash, get_cached_result, cache_result
from openai import OpenAI
import streamlit as st
import re

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    NVIDIA_API_KEY = "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"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

FAST_MODEL_NAME = "nvidia/nemotron-2-8b-chat-v1"

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
    datetime_columns: List[str] = None,
    imputation_strategy: str = 'simple',
    knn_neighbors: int = 5,
    multi_label_columns: List[str] = None
) -> pd.DataFrame:
    df_processed = df.copy()
    
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
    
    for col in df_processed.select_dtypes(include='object').columns:
        if not df_processed[col].empty and isinstance(df_processed[col].iloc[0], str) and df_processed[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
            try:
                df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            except Exception:
                pass
    
    if missing_strategy and missing_strategy != "None":
        if imputation_strategy == 'knn':
            num_cols = df_processed.select_dtypes(include=np.number).columns
            if not df_processed[num_cols].empty:
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
    
    if encode_categorical:
        all_cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        le = LabelEncoder()
        if target_columns:
            cols_to_le = [col for col in target_columns if col in all_cat_cols and (not one_hot_encode_columns or col not in one_hot_encode_columns)]
        else:
            cols_to_le = [col for col in all_cat_cols if (not one_hot_encode_columns or col not in one_hot_encode_columns)]
        for col in cols_to_le:
            if col in df_processed.columns:
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    if one_hot_encode_columns:
        valid_ohe_cols = [col for col in one_hot_encode_columns if col in df_processed.columns and df_processed[col].dtype in ['object', 'category']]
        if valid_ohe_cols:
            df_processed = pd.get_dummies(df_processed, columns=valid_ohe_cols, dummy_na=False)
    
    if scale_features:
        all_num_cols = df_processed.select_dtypes(include=np.number).columns
        if target_columns:
            num_cols_to_process = [col for col in target_columns if col in all_num_cols]
        else:
            num_cols_to_process = list(all_num_cols)
        num_cols_to_process = [col for col in num_cols_to_process if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]
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
            print("[PreprocessingTool] No numeric columns found for scaling.")
    
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
    
    if feature_engineering:
        if feature_engineering.get('polynomial_cols'):
            degree = feature_engineering.get('polynomial_degree', 2)
            poly_cols = feature_engineering['polynomial_cols']
            valid_poly_cols = [col for col in poly_cols if col in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[col])]
            if valid_poly_cols:
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_data = poly.fit_transform(df_processed[valid_poly_cols])
                poly_feature_names = poly.get_feature_names_out(valid_poly_cols)
                poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df_processed.index)
                for col_name in poly_df.columns:
                    if col_name not in df_processed.columns:
                        df_processed[col_name] = poly_df[col_name]
            else:
                print(f"[PreprocessingTool] No valid numeric columns found for polynomial features: {poly_cols}")
        
        if feature_engineering.get('date_cols'):
            for col in feature_engineering['date_cols']:
                if col in df_processed.columns:
                    try:
                        temp_series = pd.to_datetime(df_processed[col], errors='coerce')
                        if pd.api.types.is_datetime64_any_dtype(temp_series):
                            df_processed[f'{col}_year'] = temp_series.dt.year
                            df_processed[f'{col}_month'] = temp_series.dt.month
                            df_processed[f'{col}_day'] = temp_series.dt.day
                    except Exception as e:
                        print(f"Error processing date column {col} for feature engineering: {e}")
    
    if multi_label_columns:
        for col in multi_label_columns:
            if col in df_processed.columns:
                split_lists = df_processed[col].astype(str).fillna("").apply(lambda x: [s.strip() for s in re.split(r'\s*[|,|\\s*]\s*', x) if s.strip()])
                mlb = MultiLabelBinarizer()
                mlb_df = pd.DataFrame(mlb.fit_transform(split_lists), columns=[f"{col}_{c.replace(' ', '_')}" for c in mlb.classes_], index=df_processed.index)
                df_processed = pd.concat([df_processed.drop(columns=[col]), mlb_df], axis=1)
    
    return df_processed

def PreprocessingSuggestionTool(df: pd.DataFrame) -> Dict[str, str]:
    dataset_hash_query = f"preprocessing_suggestions_{get_df_hash(df)}"
    _, cached_result = get_cached_result(dataset_hash_query)
    if cached_result and isinstance(cached_result, dict):
        return cached_result

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
    
    if len(num_cols) > 0:
        std_devs = df[num_cols].std()
        if not std_devs.empty and std_devs.max() > 10:
            suggestions["scale_features"] = "Scale numerical features to normalize large value ranges."
    
    newline = '\n'
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
            model=FAST_MODEL_NAME,
            messages=[{"role": "system", "content": "Provide concise explanations for preprocessing steps."},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        explanation_text = response.choices[0].message.content.strip()
        suggestions["explanation"] = explanation_text
        cache_result(dataset_hash_query, prompt, suggestions)
        return suggestions
    except Exception as e:
        print(f"Error in PreprocessingSuggestionTool (LLM part): {e}")
        suggestions["explanation"] = "Could not generate LLM explanation due to an error."
        cache_result(dataset_hash_query, prompt, suggestions)
        return suggestions