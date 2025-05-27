import streamlit as st
import pandas as pd
from typing import Dict, Any, List
import numpy as np
import json

from utils.data_loading import load_data
from agents.data_insights import CombinedAnalysisAgent
from tools.preprocessing import PreprocessingTool

def render_ui(session_state):
    st.header("Data Analysis Agent")
    st.markdown("<medium>Powered by <a href='https://build.nvidia.com/nvidia/llama-3.1-nemotron-ultra-253b-v1'>NVIDIA Llama-3.1-Nemotron-Ultra-253b-v1</a></medium>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose CSV or Excel", type=["csv", "xlsx"], key="file_uploader")
    
    if uploaded_file is not None:
        if session_state.current_file_name != uploaded_file.name:
            session_state.df = load_data(uploaded_file)
            session_state.current_file_name = uploaded_file.name
            session_state.messages = []
            session_state.plots = []
            session_state.df_processed_history = []
            session_state.initial_analysis_done = False
            session_state.insights = None
            session_state.preprocessing_suggestions = {}
            session_state.visualization_suggestions = []
            session_state.model_suggestions = None
            st.rerun()
        
        if session_state.df is not None:
            st.markdown(f"**Dataset Info: {session_state.current_file_name}**")
            st.markdown(f"Rows: {len(session_state.df)}, Columns: {len(session_state.df.columns)}")
            with st.expander("Column Names and Types"):
                col_info_df = pd.DataFrame({
                    'Column Name': session_state.df.columns,
                    'Data Type': [str(dtype) for dtype in session_state.df.dtypes]
                })
                st.dataframe(col_info_df, use_container_width=True)
            
            with st.expander("Dataset Preview (First 5 Rows)", expanded=False):
                st.dataframe(session_state.df.head())
            
            if not session_state.initial_analysis_done:
                with st.spinner("Generating initial dataset analysis..."):
                    analysis_results = CombinedAnalysisAgent(session_state.df)
                    session_state.insights = analysis_results.get("insights", "Insights generation failed.")
                    session_state.preprocessing_suggestions = analysis_results.get("preprocessing_suggestions", {})
                    session_state.visualization_suggestions = analysis_results.get("visualization_suggestions", [])
                    session_state.model_suggestions = analysis_results.get("model_recommendations", "Model suggestions failed.")
                    session_state.initial_analysis_done = True
                    initial_chat_messages = []
                    if session_state.insights and "failed" not in session_state.insights and "Error:" not in session_state.insights:
                        initial_chat_messages.append(f"### Dataset Insights\n{session_state.insights}")
                    else:
                        initial_chat_messages.append("### Dataset Insights\nCould not retrieve insights at this time.")
                    if session_state.model_suggestions and "failed" not in session_state.model_suggestions and "Error:" not in session_state.model_suggestions:
                        initial_chat_messages.append(f"### Model Suggestions\n{session_state.model_suggestions}")
                    else:
                        initial_chat_messages.append("### Model Suggestions\nCould not retrieve model suggestions at this time.")
                    if initial_chat_messages:
                        session_state.messages.insert(0, {
                            "role": "assistant",
                            "content": "\n\n---\n\n".join(initial_chat_messages)
                        })
                    st.rerun()
        else:
            st.info("Upload a CSV or Excel file to begin.")
            if session_state.current_file_name is not None:
                session_state.current_file_name = None
                session_state.df = None
                session_state.messages = []
                session_state.plots = []
                session_state.initial_analysis_done = False
                session_state.insights = None
                session_state.preprocessing_suggestions = {}
                session_state.visualization_suggestions = []
                session_state.model_suggestions = None
                st.rerun()
    
    st.header("Tool Dashboard")
    tab_pre, tab_eda, tab_utils = st.tabs(["Preprocessing", "EDA", "Utilities"])
    with tab_pre:
        with st.expander("AI Suggestions", expanded=True):
            st.subheader("AI Suggested Preprocessing")
            if session_state.initial_analysis_done and session_state.preprocessing_suggestions:
                suggestions_to_display = dict(session_state.preprocessing_suggestions)
                explanation = suggestions_to_display.pop("explanation", None)
                if not suggestions_to_display:
                    st.caption("No specific preprocessing steps suggested by AI.")
                for i, (key, desc) in enumerate(suggestions_to_display.items()):
                    button_key = f"preprocess_btn_{key.replace(' ', '_')}_{i}"
                    if st.button(desc, key=button_key, help=f"Apply action: {key}"):
                        session_state.df_before_preprocess = session_state.df.copy() if session_state.df is not None else None
                        query_for_preprocessing = f"Apply AI suggestion: {desc}"
                        session_state.messages.append({"role": "user", "content": query_for_preprocessing})
                        session_state.last_preprocess_action = key
                        st.rerun()
                if explanation:
                    st.markdown(f"**AI Explanation:** {explanation}")
            elif session_state.df is not None and not session_state.initial_analysis_done:
                st.caption("Suggestions will appear after initial analysis.")
            elif session_state.df is None:
                st.caption("Upload a dataset to see suggestions.")
            else:
                st.caption("No preprocessing suggestions from AI.")
            
            if (
                hasattr(session_state, 'df_before_preprocess') and
                session_state.df_before_preprocess is not None and
                session_state.df is not None and
                hasattr(session_state, 'last_preprocess_action') and
                session_state.last_preprocess_action is not None
            ):
                old_df = session_state.df_before_preprocess
                new_df = session_state.df
                changed_cols = [col for col in new_df.columns if not old_df[col].equals(new_df[col]) if col in old_df.columns]
                added_cols = [col for col in new_df.columns if col not in old_df.columns]
                removed_cols = [col for col in old_df.columns if col not in new_df.columns]
                st.markdown(f"### Preprocessing Applied: {session_state.last_preprocess_action}")
                st.markdown(f"**Changed columns:** {', '.join(changed_cols) if changed_cols else 'None'}")
                st.markdown(f"**Added columns:** {', '.join(added_cols) if added_cols else 'None'}")
                st.markdown(f"**Removed columns:** {', '.join(removed_cols) if removed_cols else 'None'}")
                old_missing = old_df.isnull().sum().sum()
                new_missing = new_df.isnull().sum().sum()
                st.markdown(f"**Missing values before:** {old_missing}, **after:** {new_missing}")
                st.markdown("#### Preview of Modified Dataset:")
                st.dataframe(new_df.head())
                session_state.df_before_preprocess = None
                session_state.last_preprocess_action = None
    
        with st.expander("Preprocessing Tools", expanded=True):
            if session_state.df is None:
                st.caption("Upload a dataset to use preprocessing tools.")
            else:
                st.subheader("Handle Missing Values")
                with st.form("impute_form"):
                    st.selectbox("Column to Impute", options=session_state.df.columns, key="impute_col_select")
                    st.selectbox("Imputation Strategy", options=["mean", "median", "mode", "constant", "forward_fill", "backward_fill"], key="impute_strategy_select")
                    st.text_input("Constant Value (if strategy is 'constant')", key="impute_constant_val")
                    impute_submit = st.form_submit_button("Apply Imputation")
                    if impute_submit:
                        col = st.session_state.impute_col_select
                        strategy = st.session_state.impute_strategy_select
                        const_val = st.session_state.impute_constant_val
                        query = f"Impute column '{col}' with {strategy}"
                        if strategy == "constant":
                            query += f" (value: {const_val})"
                        session_state.messages.append({"role": "user", "content": query})
                        st.rerun()
                
                st.subheader("Encode Categorical Variables")
                with st.form("encode_form"):
                    st.selectbox("Column to Encode", options=session_state.df.select_dtypes(include='object').columns, key="encode_col_select")
                    st.selectbox("Encoding Strategy", options=["label_encoding", "one_hot_encoding"], key="encode_strategy_select")
                    encode_submit = st.form_submit_button("Apply Encoding")
                    if encode_submit:
                        col = st.session_state.encode_col_select
                        strategy = st.session_state.encode_strategy_select
                        query = f"{strategy} for column '{col}'"
                        session_state.messages.append({"role": "user", "content": query})
                        st.rerun()
                
                st.subheader("Scale Numerical Features")
                with st.form("scale_form"):
                    st.multiselect("Columns to Scale", options=session_state.df.select_dtypes(include=np.number).columns, key="scale_cols_select")
                    st.selectbox("Scaling Strategy", options=["standard_scaling", "min_max_scaling", "robust_scaling"], key="scale_strategy_select")
                    scale_submit = st.form_submit_button("Apply Scaling")
                    if scale_submit:
                        cols = st.session_state.scale_cols_select
                        strategy = st.session_state.scale_strategy_select
                        if cols:
                            query = f"{strategy} for columns: {', '.join(cols)}"
                            session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                        else:
                            st.warning("Please select columns to scale.")
                
                st.subheader("Outlier Handling (IQR)")
                with st.form("outlier_form"):
                    outlier_cols = st.multiselect("Select columns for outlier handling", options=session_state.df.select_dtypes(include='number').columns, key="outlier_cols_select")
                    outlier_strategy = st.selectbox("Outlier Strategy", options=["remove", "cap"], key="outlier_strategy_select")
                    outlier_submit = st.form_submit_button("Apply Outlier Handling")
                    if outlier_submit and outlier_cols:
                        query = f"Apply {outlier_strategy} outlier handling to columns: {', '.join(outlier_cols)}"
                        session_state.messages.append({"role": "user", "content": query})
                        st.rerun()
                
                st.subheader("Feature Engineering")
                with st.form("feature_eng_form"):
                    feat_type = st.selectbox("Feature Type", ["Polynomial Features", "Date Component Extraction"], key="feat_type_select")
                    if feat_type == "Polynomial Features":
                        poly_cols = st.multiselect("Columns for Polynomial Features", options=session_state.df.select_dtypes(include='number').columns, key="poly_cols_select")
                        poly_degree = st.number_input("Polynomial Degree", min_value=2, max_value=5, value=2, key="poly_degree_input")
                    else:
                        date_cols = st.multiselect("Date Columns", options=session_state.df.columns, key="date_cols_select")
                    feat_submit = st.form_submit_button("Apply Feature Engineering")
                    if feat_submit:
                        if feat_type == "Polynomial Features" and poly_cols:
                            query = f"Add polynomial features (degree {poly_degree}) for columns: {', '.join(poly_cols)}"
                        elif feat_type == "Date Component Extraction" and date_cols:
                            query = f"Extract date components (year, month, day) from columns: {', '.join(date_cols)}"
                        else:
                            query = None
                        if query:
                            session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
    with tab_eda:
        with st.expander("EDA Tools", expanded=True):
            if session_state.df is None:
                st.caption("Upload a dataset to use EDA tools.")
            else:
                st.subheader("Manual Visualizations")
                with st.form("manual_viz_form"):
                    plot_type = st.selectbox("Plot Type", ["bar", "line", "pie", "histogram", "scatter"], key="plot_type_select")
                    x_col = st.selectbox("X Axis", options=session_state.df.columns, key="x_col_select")
                    y_col = None
                    if plot_type in ["bar", "line", "scatter"]:
                        y_col = st.selectbox("Y Axis", options=session_state.df.columns, key="y_col_select")
                    chart_lib = st.selectbox("Chart Library", ["Matplotlib", "Chart.js"], key="chart_lib_select")
                    viz_submit = st.form_submit_button("Generate Visualization")
                    if viz_submit:
                        if plot_type in ["bar", "line", "scatter"] and y_col:
                            query = f"Show {plot_type} chart of {y_col} vs {x_col} using {chart_lib}"
                        elif plot_type in ["pie", "histogram"]:
                            query = f"Show {plot_type} of {x_col} using {chart_lib}"
                        else:
                            query = None
                        if query:
                            session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
                st.subheader("Statistical Summaries")
                if st.button("Show Statistical Summary", key="stat_summary_btn"):
                    session_state.messages.append({"role": "user", "content": "Show statistical summary (describe)"})
                    st.rerun()
                st.subheader("Correlation Matrix")
                if st.button("Show Correlation Matrix", key="corr_matrix_btn"):
                    session_state.messages.append({"role": "user", "content": "Show correlation matrix heatmap"})
                    st.rerun()
                st.subheader("Data Filtering")
                with st.form("filter_form"):
                    filter_col = st.selectbox("Column to Filter", options=session_state.df.columns, key="filter_col_select")
                    filter_op = st.selectbox("Operator", [">", ">=", "<", "<=", "==", "!="], key="filter_op_select")
                    filter_val = st.text_input("Value", key="filter_val_input")
                    filter_submit = st.form_submit_button("Apply Filter")
                    if filter_submit and filter_col and filter_op and filter_val:
                        query = f"Filter rows where {filter_col} {filter_op} {filter_val}"
                        session_state.messages.append({"role": "user", "content": query})
                        st.rerun()
    with tab_utils:
        if session_state.df is None:
            st.caption("Upload a dataset to use utility tools.")
        else:
            st.subheader("Missing Value Summary")
            if st.button("Show Detailed Missing Values Summary", key="detailed_missing_val_summary_btn"):
                session_state.messages.append({"role": "user", "content": "Show detailed missing value summary and analysis"})
                st.rerun()
            
            st.subheader("Model Recommendations")
            if st.button("Show Model Recommendations", key="model_recom_btn_sidebar", disabled=not session_state.initial_analysis_done):
                if session_state.model_suggestions:
                    session_state.messages.append({
                        "role": "assistant", 
                        "content": f"### Model Suggestions (from initial analysis)\n{session_state.model_suggestions}"
                    })
                    st.rerun()
                else:
                    st.warning("Model suggestions not available from initial analysis. You can ask the chat.")
            
            st.subheader("Download Dataset")
            csv_data = session_state.df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Current Dataset (CSV)",
                data=csv_data,
                file_name=f"processed_{session_state.current_file_name if session_state.current_file_name else 'dataset.csv'}",
                mime="text/csv",
                key="download_csv_sidebar_btn"
            )
            
            st.subheader("Export Chat History")
            if session_state.messages:
                chat_json = json.dumps(session_state.messages, indent=2, default=str)
                st.download_button(
                    label="Download Chat History (JSON)",
                    data=chat_json,
                    file_name="chat_history.json",
                    mime="application/json",
                    key="download_json_btn"
                )
                chat_txt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in session_state.messages])
                st.download_button(
                    label="Download Chat History (Text)",
                    data=chat_txt,
                    file_name="chat_history.txt",
                    mime="text/plain",
                    key="download_txt_btn"
                )
            else:
                st.caption("No chat history to export.")

# Placeholder for sidebar rendering function
def render_sidebar(df: pd.DataFrame = None):
    st.sidebar.title("AskurData Controls")
    if df is not None:
        st.sidebar.metric("Dataset Rows", len(df))
        st.sidebar.metric("Dataset Columns", len(df.columns))
    
    st.sidebar.subheader("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    # Placeholder for preprocessing options if df is loaded
    if df is not None:
        st.sidebar.subheader("Preprocessing Options")
        # Example: Imputation strategy selection
        # missing_strategy = st.sidebar.selectbox("Missing Value Strategy", ["None", "mean", "median", "mode", "constant", "knn"], key="ui_missing_strategy")
        # Further UI elements for other preprocessing steps would go here

    st.sidebar.info("Refactored AskurData App")
    return uploaded_file # Return uploaded file to main app

# Placeholder for chat interface rendering function
def render_chat_interface(chat_history: List[Dict[str, str]]):
    st.subheader("Chat with your Data")
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new user query
    # user_query = st.chat_input("Ask a question about your data...")
    # return user_query 
    # Chat input handling will likely be in main.py to manage session state

# Placeholder for rendering analysis results (tabs, tables, plots)
def render_analysis_results(analysis_output: Any, output_type: str = "text"):
    if output_type == "text":
        st.markdown(analysis_output)
    elif output_type == "dataframe":
        if isinstance(analysis_output, pd.DataFrame):
            st.dataframe(analysis_output)
        elif isinstance(analysis_output, list) and all(isinstance(i, dict) for i in analysis_output):
            st.table(analysis_output) # Good for list of records
        else:
            st.text("Result is a DataFrame, but format is unexpected for st.dataframe or st.table.")
            st.text(str(analysis_output)) # Fallback to text
    elif output_type == "plot": # Matplotlib figure
        st.pyplot(analysis_output)
    elif output_type == "chartjs_config":
        # This would require a custom component or st_echarts like library for Chart.js
        # For now, just display the config as JSON
        st.json(analysis_output)
        st.caption("Chart.js configuration object. Integrate with a Chart.js component to render.")
    elif output_type == "error":
        st.error(analysis_output)
    else:
        st.text(str(analysis_output)) # Default to text display

# Add other UI component functions as needed (e.g., for tabs, forms)