import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from askurdata.tools.preprocessing import PreprocessingTool, PreprocessingSuggestionTool
from askurdata.utils.cache import get_df_hash

class TestPreprocessingTool:
    
    def test_preprocessing_tool_impute_mean(self):
        """Test mean imputation for numerical columns."""
        df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", "z"]})
        result = PreprocessingTool(df, missing_strategy="mean", target_columns=["A"])
        assert result["A"].isnull().sum() == 0
        assert abs(result["A"].iloc[1] - 2.0) < 1e-6
        assert result["B"].equals(df["B"])
    
    def test_preprocessing_tool_impute_median(self):
        """Test median imputation for numerical columns."""
        df = pd.DataFrame({"A": [1, None, 3, 5], "B": ["x", "y", "z", "w"]})
        result = PreprocessingTool(df, missing_strategy="median", target_columns=["A"])
        assert result["A"].isnull().sum() == 0
        assert result["A"].iloc[1] == 3.0  # median of [1, 3, 5]
    
    def test_preprocessing_tool_encode_categorical(self):
        """Test label encoding of categorical columns."""
        df = pd.DataFrame({"A": ["cat", "dog", "cat"], "B": [1, 2, 3]})
        result = PreprocessingTool(df, encode_categorical=True, target_columns=["A"])
        assert result["A"].dtype in [np.int32, np.int64]
        assert len(result["A"].unique()) == 2  # cat and dog
        assert result["B"].equals(df["B"])
    
    def test_preprocessing_tool_scale_features(self):
        """Test standard scaling of numerical features."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
        result = PreprocessingTool(df, scale_features=True, scaling_strategy="standard")
        # After standard scaling, mean should be ~0 and std should be ~1
        assert abs(result["A"].mean()) < 1e-10
        assert abs(result["A"].std() - 1.0) < 1e-10
        assert abs(result["B"].mean()) < 1e-10
        assert abs(result["B"].std() - 1.0) < 1e-10
    
    def test_preprocessing_tool_one_hot_encoding(self):
        """Test one-hot encoding of categorical columns."""
        df = pd.DataFrame({"A": ["cat", "dog", "cat"], "B": [1, 2, 3]})
        result = PreprocessingTool(df, one_hot_encode_columns=["A"])
        # Should have A_cat and A_dog columns
        assert "A_cat" in result.columns
        assert "A_dog" in result.columns
        assert "A" not in result.columns  # Original column should be removed
        assert result["B"].equals(df["B"])
    
    def test_preprocessing_tool_outlier_handling(self):
        """Test outlier handling using IQR method."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 100]})  # 100 is an outlier
        result = PreprocessingTool(df, outlier_strategy="remove", outlier_columns=["A"])
        assert len(result) < len(df)  # Should have fewer rows after removing outliers
        assert 100 not in result["A"].values
    
    def test_preprocessing_tool_constant_imputation(self):
        """Test constant value imputation."""
        df = pd.DataFrame({"A": [1, None, 3]})
        result = PreprocessingTool(df, missing_strategy="constant", 
                                 constant_value_impute=999, target_columns=["A"])
        assert result["A"].isnull().sum() == 0
        assert result["A"].iloc[1] == 999
    
    def test_preprocessing_tool_knn_imputation(self):
        """Test KNN imputation."""
        df = pd.DataFrame({"A": [1, None, 3], "B": [10, 20, 30]})
        result = PreprocessingTool(df, imputation_strategy="knn", knn_neighbors=2)
        assert result["A"].isnull().sum() == 0
        # KNN should impute a reasonable value based on neighbors
        assert result["A"].iloc[1] > 0

class TestPreprocessingSuggestionTool:
    
    def test_preprocessing_suggestions_basic(self):
        """Test basic preprocessing suggestions."""
        df = pd.DataFrame({
            "A": [1, None, 3], 
            "B": ["cat", "dog", "cat"],
            "C": [100, 200, 300]
        })
        suggestions = PreprocessingSuggestionTool(df)
        assert isinstance(suggestions, dict)
        assert "explanation" in suggestions
        # Should suggest imputation for column A
        impute_keys = [k for k in suggestions.keys() if k.startswith("impute_")]
        assert len(impute_keys) > 0
    
    def test_preprocessing_suggestions_no_missing(self):
        """Test suggestions when there are no missing values."""
        df = pd.DataFrame({
            "A": [1, 2, 3], 
            "B": ["cat", "dog", "mouse"]
        })
        suggestions = PreprocessingSuggestionTool(df)
        assert isinstance(suggestions, dict)
        assert "explanation" in suggestions
        # Should suggest encoding categorical variables
        assert any("encode" in key for key in suggestions.keys())

if __name__ == "__main__":
    pytest.main([__file__]) 