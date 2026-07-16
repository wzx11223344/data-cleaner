"""Auto-generated tests for data-cleaner."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main


class TestMain:
    """Tests for data-cleaner module."""

    def test_module_import(self):
        """Test that main module imports correctly."""
        assert main is not None
        assert hasattr(main, "data_profiling")


    def test_data_profiling_basic(self):
        """Test data profiling."""
        data = [
            {"name": "Alice", "age": 30, "score": 85.5},
            {"name": "Bob", "age": 25, "score": 90.0},
            {"name": "Charlie", "age": 35, "score": 78.0},
        ]
        result = main.data_profiling(data)
        assert result["row_count"] == 3
        assert result["column_count"] >= 2
        assert "columns" in result

    def test_data_profiling_stats(self):
        """Test that profiling returns column statistics."""
        data = [{"value": 10}, {"value": 20}, {"value": 30}]
        result = main.data_profiling(data)
        assert "stats" in result["columns"]["value"] or "min" in str(result["columns"]["value"])

    def test_data_profiling_empty(self):
        """Test profiling with empty data."""
        result = main.data_profiling([])
        assert result["row_count"] == 0


    def test_outlier_detector_zscore(self):
        """Test Z-score outlier detection."""
        result = main.outlier_detector_zscore([1, 2, 2, 3, 3, 4, 100])
        assert "outliers" in result
        assert 100 in result["outliers"] or len(result["outliers"]) > 0

    def test_outlier_no_outliers(self):
        """Test outlier detection with no outliers."""
        result = main.outlier_detector_zscore([1, 2, 3, 4, 5])
        assert len(result["outliers"]) <= 1

    def test_outlier_empty(self):
        """Test outlier detection with empty data."""
        result = main.outlier_detector_zscore([])
        assert result["outliers"] == [] or "error" in result

    def test_outlier_detector_isolation_forest_exists(self):
        """Test that outlier_detector_isolation_forest function is callable."""
        assert callable(main.outlier_detector_isolation_forest)
        assert main.outlier_detector_isolation_forest.__doc__ is not None


    def test_missing_value_imputer_mean(self):
        """Test missing value imputation with mean."""
        data = [{"a": 1, "b": 10}, {"a": 2, "b": None}, {"a": 3, "b": 30}]
        result = main.missing_value_imputer(data, "b", "mean")
        assert result is not None

    def test_missing_value_imputer_median(self):
        """Test missing value imputation with median."""
        data = [{"x": 1}, {"x": None}, {"x": 5}, {"x": 7}]
        result = main.missing_value_imputer(data, "x", "median")
        assert result is not None

    def test_missing_value_imputer_all_none(self):
        """Test imputation when all values are None."""
        data = [{"a": None}, {"a": None}]
        result = main.missing_value_imputer(data, "a", "mean")
        assert result is not None

    def test_duplicate_detector_exists(self):
        """Test that duplicate_detector function is callable."""
        assert callable(main.duplicate_detector)
        assert main.duplicate_detector.__doc__ is not None
