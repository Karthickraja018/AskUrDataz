import streamlit as st
import pandas as pd
import os
import json
import sys
from pathlib import Path

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
from agents.code_generation import CodeGenerationAgent, ExecutionAgent, ReasoningAgent
from agents.data_insights import CombinedAnalysisAgent, MissingValueAgent, DataInsightAgent

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
                    intent_result = QueryUnderstandingTool(st.session_state.df, user_query)
                    intent = intent_result.get("intent", "unknown")
                    
                    assistant_response = ""
                    
                    if intent_result.get("needs_clarification"):
                        assistant_response = intent_result.get("clarification_question", "I'm not sure how to proceed. Could you clarify?")
                    
                    elif intent == "preprocess":
                        # Handle preprocessing requests
                        try:
                            preprocess_params = intent_result.get("preprocess_params", {})
                            df_processed = PreprocessingTool(st.session_state.df, **preprocess_params)
                            
                            if not st.session_state.df.equals(df_processed):
                                st.session_state.df = df_processed
                                assistant_response = "Preprocessing applied successfully. The DataFrame has been updated."
                                st.success("DataFrame updated!")
                            else:
                                assistant_response = "Preprocessing was applied, but no changes were detected in the DataFrame."
                        except Exception as e:
                            assistant_response = f"Error during preprocessing: {str(e)}"
                    
                    elif intent == "visualize":
                        # Handle visualization requests
                        chart_type = intent_result.get("chart_type", "")
                        plot_desc = intent_result.get("plot_description", user_query)
                        
                        if "chartjs" in chart_type.lower():
                            # Generate Chart.js config
                            try:
                                config = ChartJSCodeGeneratorTool(st.session_state.df, plot_desc, chart_type)
                                assistant_response = f"Generated Chart.js configuration for: {plot_desc}"
                                st.json(config)
                            except Exception as e:
                                assistant_response = f"Error generating Chart.js visualization: {str(e)}"
                        else:
                            # Generate matplotlib plot
                            try:
                                viz_code = PlotCodeGeneratorTool(st.session_state.df, plot_desc, chart_type)
                                plot_result = ExecutionAgent(viz_code, st.session_state.df, user_query)
                                
                                if isinstance(plot_result, str) and "Error" in plot_result:
                                    assistant_response = plot_result
                                else:
                                    assistant_response = f"Generated plot for: {plot_desc}"
                                    if hasattr(plot_result, 'getvalue'):  # BytesIO object
                                        st.image(plot_result.getvalue())
                                    else:
                                        st.pyplot(plot_result)
                            except Exception as e:
                                assistant_response = f"Error generating visualization: {str(e)}"
                    
                    elif intent == "analyze":
                        # Handle analysis requests
                        analysis_type = intent_result.get("analysis_type", "general")
                        
                        if analysis_type == "missing_values":
                            try:
                                missing_info = MissingValueAgent(st.session_state.df)
                                assistant_response = missing_info.get("summary", "Could not retrieve missing value summary.")
                                
                                if missing_info.get("dataframe"):
                                    st.dataframe(pd.DataFrame(missing_info["dataframe"]))
                            except Exception as e:
                                assistant_response = f"Error analyzing missing values: {str(e)}"
                        else:
                            # Generic analysis
                            try:
                                analysis_code = CodeWritingTool(st.session_state.df.columns.tolist(), user_query, intent="analyze")
                                analysis_result = ExecutionAgent(analysis_code, st.session_state.df, user_query)
                                
                                if isinstance(analysis_result, str) and "Error" in analysis_result:
                                    assistant_response = analysis_result
                                elif isinstance(analysis_result, (pd.DataFrame, pd.Series)):
                                    assistant_response = "Analysis complete. Results:"
                                    st.dataframe(analysis_result)
                                else:
                                    assistant_response = f"Analysis result: {str(analysis_result)}"
                            except Exception as e:
                                assistant_response = f"Error during analysis: {str(e)}"
                    
                    elif intent == "model":
                        # Handle model recommendation requests
                        try:
                            target_var = intent_result.get("model_params", {}).get("target_variable")
                            recommendations = ModelRecommendationTool(st.session_state.df, user_query, target_var)
                            
                            if isinstance(recommendations, list):
                                assistant_response = "**Suggested Models:**\n"
                                for rec in recommendations:
                                    assistant_response += f"- **{rec.get('model_name', 'N/A')}**: {rec.get('description', 'N/A')} (Use case: {rec.get('use_case', 'N/A')})\n"
                            else:
                                assistant_response = recommendations
                        except Exception as e:
                            assistant_response = f"Error generating model recommendations: {str(e)}"
                    
                    else:
                        # Unknown or clarification needed
                        assistant_response = "I'm not sure how to help with that. Could you try rephrasing your question or be more specific?"
                    
                    # Display assistant response
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()