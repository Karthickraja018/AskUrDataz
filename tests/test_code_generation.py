import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from askurdata.agents.code_generation import CodeGenerationAgent, ExecutionAgent, ReasoningAgent
from askurdata.tools.code_writing import CodeWritingTool, PreprocessingCodeGeneratorTool

class TestCodeGenerationAgent:
    
    def test_code_generation_agent_basic(self):
        """Test basic code generation functionality."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        query = "Calculate the mean of column A"
        
        # This should return generated code
        result = CodeGenerationAgent(query, df)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_code_generation_agent_visualization(self):
        """Test code generation for visualization queries."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        query = "Create a scatter plot of A vs B"
        
        result = CodeGenerationAgent(query, df)
        assert isinstance(result, str)
        assert "plt" in result.lower() or "matplotlib" in result.lower()

class TestExecutionAgent:
    
    def test_execution_agent_simple_calculation(self):
        """Test execution of simple calculation code."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        code = "result = df['A'].mean()"
        
        result = ExecutionAgent(code, df, "analytics", False, "test_query")
        assert result == 2.0
    
    def test_execution_agent_dataframe_operation(self):
        """Test execution of DataFrame operations."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        code = "result = df.describe()"
        
        result = ExecutionAgent(code, df, "analytics", False, "test_query")
        assert isinstance(result, pd.DataFrame)
        assert "A" in result.columns
        assert "B" in result.columns
    
    def test_execution_agent_error_handling(self):
        """Test error handling in ExecutionAgent."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        code = "result = df['nonexistent_column'].mean()"  # This should cause an error
        
        result = ExecutionAgent(code, df, "analytics", False, "test_query")
        assert isinstance(result, str)
        assert "Error" in result
    
    def test_execution_agent_missing_result(self):
        """Test handling when code doesn't set result variable."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        code = "x = df['A'].mean()"  # No result variable set
        
        result = ExecutionAgent(code, df, "analytics", False, "test_query")
        assert result is None  # Should return None when no result is set

class TestReasoningAgent:
    
    def test_reasoning_agent_basic(self):
        """Test basic reasoning functionality."""
        query = "What does this analysis tell us?"
        result = 2.5  # Some analysis result
        
        reasoning = ReasoningAgent(query, result)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
    
    def test_reasoning_agent_dataframe_result(self):
        """Test reasoning with DataFrame results."""
        query = "Explain this correlation matrix"
        result = pd.DataFrame({"A": [1.0, 0.5], "B": [0.5, 1.0]})
        
        reasoning = ReasoningAgent(query, result)
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

class TestCodeWritingTool:
    
    def test_code_writing_tool_basic(self):
        """Test basic code writing functionality."""
        cols = ["A", "B", "C"]
        query = "Calculate the sum of column A"
        
        code = CodeWritingTool(cols, query, intent="analyze")
        assert isinstance(code, str)
        assert "result" in code
        assert "A" in code
    
    def test_code_writing_tool_missing_values(self):
        """Test code generation for missing values analysis."""
        cols = ["A", "B", "C"]
        query = "Show missing values in the dataset"
        
        code = CodeWritingTool(cols, query, intent="analyze")
        assert isinstance(code, str)
        assert "isnull" in code.lower() or "isna" in code.lower()
    
    def test_code_writing_tool_statistics(self):
        """Test code generation for statistical analysis."""
        cols = ["A", "B", "C"]
        query = "Show summary statistics"
        
        code = CodeWritingTool(cols, query, intent="analyze")
        assert isinstance(code, str)
        assert "describe" in code.lower() or "stats" in code.lower()

class TestPreprocessingCodeGeneratorTool:
    
    def test_preprocessing_code_generator_basic(self):
        """Test basic preprocessing code generation."""
        cols = ["A", "B", "C"]
        params = {
            "missing_strategy": "mean",
            "target_columns": ["A"],
            "encode_categorical": False,
            "scale_features": False
        }
        
        code = PreprocessingCodeGeneratorTool(cols, params)
        assert isinstance(code, str)
        assert "result" in code
        assert "mean" in code.lower() or "impute" in code.lower()
    
    def test_preprocessing_code_generator_scaling(self):
        """Test preprocessing code generation with scaling."""
        cols = ["A", "B", "C"]
        params = {
            "scale_features": True,
            "scaling_strategy": "standard",
            "target_columns": ["A", "B"]
        }
        
        code = PreprocessingCodeGeneratorTool(cols, params)
        assert isinstance(code, str)
        assert "StandardScaler" in code or "standard" in code.lower()
    
    def test_preprocessing_code_generator_encoding(self):
        """Test preprocessing code generation with encoding."""
        cols = ["A", "B", "C"]
        params = {
            "encode_categorical": True,
            "target_columns": ["C"]
        }
        
        code = PreprocessingCodeGeneratorTool(cols, params)
        assert isinstance(code, str)
        assert "LabelEncoder" in code or "encode" in code.lower()

if __name__ == "__main__":
    pytest.main([__file__]) 