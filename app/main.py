import streamlit as st
import pandas as pd
import os
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to Python path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules after adding to path
from ui_components import render_ui
from utils.data_loading import load_data
from utils.cache import init_cache, get_cached_result, cache_result
from utils.helpers import extract_first_code_block

# Import Agents
import agents.code_generation as code_gen
CodeGenerationAgent = code_gen.CodeGenerationAgent
ExecutionAgent = code_gen.ExecutionAgent
ReasoningAgent = code_gen.ReasoningAgent

# Import Tools
from tools.query_understanding import QueryUnderstandingTool
from tools.preprocessing import PreprocessingTool, PreprocessingSuggestionTool
from tools.visualization import PlotCodeGeneratorTool, ChartJSCodeGeneratorTool, VisualizationSuggestionTool
from tools.model_recommendation import ModelRecommendationTool
from tools.code_writing import CodeWritingTool, PreprocessingCodeGeneratorTool

# === Main Application Logic ===
def main():
    st.set_page_config(layout="wide", page_title="AskurData Education")
    
    # Initialize SQLite cache
    init_cache()

    # Initialize session state variables if they don't exist
    default_session_state = {
        "plots": [],
        "messages": [],
        "df": None,
        "df_processed_history": [],
        "current_file_name": None,
        "insights": None,
        "preprocessing_suggestions": {},
        "visualization_suggestions": [],
        "model_suggestions": None,
        "initial_analysis_done": False,
        "df_before_preprocess": None,
        "last_preprocess_action": None
    }
    
    for key, value in default_session_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Create two column layout: left for tools, right for chat
    left, right = st.columns([3, 7])
    
    with left:
        # Render UI components in sidebar
        render_ui(st.session_state)
    
    with right:
        st.header("💬 Chat with your Data")
        
        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if user_query := st.chat_input("Ask a question about your data..."):
            if st.session_state.df is None:
                st.warning("Please upload a dataset first.")
                st.stop()

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_query)

            # Process the query
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Step 1: Understand the query intent
                    intent_result = QueryUnderstandingTool(user_query)
                    intent = intent_result.get("intent", "unknown")
                    is_chartjs = intent_result.get("is_chartjs", False)
                    
                    assistant_response = ""
                    
                    if intent_result.get("needs_clarification"):
                        assistant_response = intent_result.get("clarification_question", "I'm not sure how to proceed. Could you clarify?")
                    
                    elif intent == "preprocessing":
                        # Handle preprocessing requests
                        try:
                            code, params = PreprocessingCodeGeneratorTool(st.session_state.df.columns.tolist(), user_query)
                            if code:
                                result = ExecutionAgent(code, st.session_state.df, intent, is_chartjs, user_query)
                                if isinstance(result, pd.DataFrame):
                                    st.session_state.df = result
                                    assistant_response = "Preprocessing applied successfully. The DataFrame has been updated."
                                    st.success("DataFrame updated!")
                                    st.dataframe(result.head())
                                else:
                                    assistant_response = str(result)
                            else:
                                assistant_response = "Could not generate preprocessing code. Please try rephrasing your request."
                        except Exception as e:
                            assistant_response = f"Error during preprocessing: {str(e)}"
                    
                    elif intent == "visualization":
                        # Handle visualization requests
                        try:
                            if is_chartjs:
                                # Generate Chart.js config using prompt
                                prompt = ChartJSCodeGeneratorTool(st.session_state.df.columns.tolist(), user_query)
                                code, _, _, result = CodeGenerationAgent(user_query, st.session_state.df)
                                if code:
                                    result = ExecutionAgent(code, st.session_state.df, intent, is_chartjs, user_query)
                                    if isinstance(result, dict):
                                        assistant_response = "Generated Chart.js configuration:"
                                        st.json(result)
                                    else:
                                        assistant_response = str(result)
                            else:
                                # Generate matplotlib plot using prompt
                                prompt = PlotCodeGeneratorTool(st.session_state.df.columns.tolist(), user_query)
                                code, _, _, result = CodeGenerationAgent(user_query, st.session_state.df)
                                if code:
                                    result = ExecutionAgent(code, st.session_state.df, intent, is_chartjs, user_query)
                                    if isinstance(result, (plt.Figure, plt.Axes)):
                                        assistant_response = "Generated visualization:"
                                        st.pyplot(result)
                                    else:
                                        assistant_response = str(result)
                        except Exception as e:
                            assistant_response = f"Error generating visualization: {str(e)}"
                    
                    elif intent == "analytics":
                        # Handle analysis requests
                        try:
                            code, _, _, result = CodeGenerationAgent(user_query, st.session_state.df)
                            if code:
                                result = ExecutionAgent(code, st.session_state.df, intent, is_chartjs, user_query)
                                if isinstance(result, str) and "Error" in result:
                                    assistant_response = result
                                elif isinstance(result, (pd.DataFrame, pd.Series)):
                                    assistant_response = "Analysis complete. Results:"
                                    st.dataframe(result)
                                else:
                                    assistant_response = f"Analysis result: {str(result)}"
                            else:
                                # Try DataInsightAgent for general insights
                                if "insights" in user_query.lower() or "describe" in user_query.lower():
                                    assistant_response = DataInsightAgent(st.session_state.df)
                                else:
                                    assistant_response = "Could not generate analysis code. Please try rephrasing your request."
                        except Exception as e:
                            assistant_response = f"Error during analysis: {str(e)}"
                    
                    else:
                        # Unknown or clarification needed
                        assistant_response = "I'm not sure how to help with that. Could you try rephrasing your question or be more specific?"
                    
                    # Get reasoning about the result if we have one
                    if assistant_response and not intent_result.get("needs_clarification"):
                        try:
                            _, reasoning = ReasoningAgent(user_query, result if 'result' in locals() else assistant_response)
                            if reasoning:
                                assistant_response = f"{assistant_response}\n\n**Analysis**: {reasoning}"
                        except Exception as e:
                            print(f"Error getting reasoning: {e}")  # Log but don't affect response
                    
                    # Display assistant response
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()