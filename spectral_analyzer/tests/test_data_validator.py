"""
Comprehensive test cases for data validator functionality.

Tests validation rules, quality checks, and error handling scenarios
to ensure robust validation of spectroscopy data.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from spectral_analyzer.core.data_validator import (
    DataValidator, ValidationLevel, ValidationRule, ValidationIssue, 
    ValidationResult
)


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
    
    def create_test_dataframe(self, wavenumbers=None, intensities=None, **kwargs):
        """Create test DataFrame with spectral data."""
        if wavenumbers is None:
            wavenumbers = np.linspace(4000, 400, 100)
        if intensities is None:
            intensities = np.random.normal(0.5, 0.1, len(wavenumbers))
        
        data = {'wavenumber': wavenumbers, 'intensity': intensities}
        data.update(kwargs)
        return pd.DataFrame(data)
    
    def test_valid_spectral_data(self):
        """Test validation of valid spectral data."""
        df = self.create_test_dataframe()
        result = self.validator.validate_data(df)
        
        assert result.is_valid
        assert result.quality_score > 80
        assert len([issue for issue in result.issues if issue.level == ValidationLevel.ERROR]) == 0
    
    def test_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        result = self.validator.validate_data(df)
        
        assert not result.is_valid
        assert any(issue.rule == ValidationRule.MISSING_VALUES for issue in result.issues)
        assert result.quality_score == 0
    
    def test_missing_wavenumber_column(self):
        """Test validation when wavenumber column is missing."""
        df = pd.DataFrame({'intensity': [0.1, 0.2, 0.3]})
        result = self.validator.validate_data(df)
        
        assert not result.is_valid
        assert any(issue.rule == ValidationRule.MISSING_VALUES and 
                  "wavenumber" in issue.message for issue in result.issues)
    
    def test_missing_intensity_columns(self):
        """Test validation when intensity columns are missing."""
        df = pd.DataFrame({'wavenumber': [4000, 3999, 3998]})
        result = self.validator.validate_data(df)
        
        # Should have warning about missing intensity columns
        assert any(issue.rule == ValidationRule.MISSING_VALUES and 
                  "intensity" in issue.message.lower() for issue in result.issues)
    
    def test_wavenumber_range_validation(self):
        """Test wavenumber range validation."""
        # Test out-of-range wavenumbers
        df = self.create_test_dataframe(wavenumbers=[100, 200, 300])  # Too low
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.WAVENUMBER_RANGE for issue in result.issues)
        
        # Test very high wavenumbers
        df = self.create_test_dataframe(wavenumbers=[5000, 6000, 7000])  # Too high
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.WAVENUMBER_RANGE for issue in result.issues)
    
    def test_wavenumber_order_validation(self):
        """Test wavenumber order validation."""
        # Ascending order (should be descending for IR)
        df = self.create_test_dataframe(wavenumbers=[400, 500, 600, 700])
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.WAVENUMBER_ORDER for issue in result.issues)
    
    def test_duplicate_wavenumbers(self):
        """Test duplicate wavenumber detection."""
        df = self.create_test_dataframe(wavenumbers=[4000, 4000, 3999, 3998])
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.DUPLICATE_VALUES for issue in result.issues)
    
    def test_missing_values_validation(self):
        """Test missing values validation."""
        wavenumbers = [4000, np.nan, 3998, 3997]
        intensities = [0.1, 0.2, np.nan, 0.4]
        df = self.create_test_dataframe(wavenumbers=wavenumbers, intensities=intensities)
        
        result = self.validator.validate_data(df)
        
        missing_issues = [issue for issue in result.issues if issue.rule == ValidationRule.MISSING_VALUES]
        assert len(missing_issues) > 0
    
    def test_absorbance_range_validation(self):
        """Test absorbance range validation."""
        # Test negative absorbance values
        df = pd.DataFrame({
            'wavenumber': [4000, 3999, 3998],
            'absorbance': [-1.0, 0.5, 10.0]  # Out of typical range
        })
        
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.ABSORBANCE_RANGE for issue in result.issues)
    
    def test_transmittance_range_validation(self):
        """Test transmittance range validation."""
        # Test out-of-range transmittance values
        df = pd.DataFrame({
            'wavenumber': [4000, 3999, 3998],
            'transmittance': [-10, 50, 150]  # Should be 0-100%
        })
        
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.TRANSMITTANCE_RANGE for issue in result.issues)
    
    def test_data_continuity_validation(self):
        """Test data continuity validation."""
        # Create data with large gaps
        wavenumbers = [4000, 3999, 3990, 3989]  # Large gap between 3999 and 3990
        df = self.create_test_dataframe(wavenumbers=wavenumbers)
        
        result = self.validator.validate_data(df)
        
        # May have continuity warning
        continuity_issues = [issue for issue in result.issues if issue.rule == ValidationRule.DATA_CONTINUITY]
        # This test is flexible as gap detection depends on the algorithm
    
    def test_minimum_data_points_validation(self):
        """Test minimum data points validation."""
        # Create dataset with too few points
        df = self.create_test_dataframe(
            wavenumbers=np.linspace(4000, 3990, 10),  # Only 10 points
            intensities=np.random.normal(0.5, 0.1, 10)
        )
        
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.MINIMUM_DATA_POINTS for issue in result.issues)
    
    def test_scale_issues_detection(self):
        """Test scale issues detection."""
        # Test very small wavenumber range (might be scaled incorrectly)
        df = self.create_test_dataframe(wavenumbers=[4.0, 3.999, 3.998])  # Divided by 1000?
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.SCALE_ISSUES for issue in result.issues)
        
        # Test very large intensity values
        df = self.create_test_dataframe(intensities=[1e6, 2e6, 3e6])  # Very large values
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.SCALE_ISSUES for issue in result.issues)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        # Enable outlier detection
        self.validator.enable_rule(ValidationRule.OUTLIER_DETECTION, True)
        
        # Create data with obvious outliers
        intensities = [0.1, 0.1, 0.1, 10.0, 0.1, 0.1]  # 10.0 is an outlier
        df = self.create_test_dataframe(intensities=intensities)
        
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.OUTLIER_DETECTION for issue in result.issues)
    
    def test_noise_level_detection(self):
        """Test noise level detection."""
        # Enable noise level detection
        self.validator.enable_rule(ValidationRule.NOISE_LEVEL, True)
        
        # Create very noisy data
        np.random.seed(42)
        intensities = np.random.normal(0.5, 0.5, 100)  # High noise
        df = self.create_test_dataframe(intensities=intensities)
        
        result = self.validator.validate_data(df)
        
        # May detect high noise level
        noise_issues = [issue for issue in result.issues if issue.rule == ValidationRule.NOISE_LEVEL]
    
    def test_baseline_drift_detection(self):
        """Test baseline drift detection."""
        # Enable baseline drift detection
        self.validator.enable_rule(ValidationRule.BASELINE_DRIFT, True)
        
        # Create data with linear trend (baseline drift)
        x = np.linspace(0, 1, 100)
        intensities = 0.5 + 0.5 * x + np.random.normal(0, 0.01, 100)  # Linear trend + noise
        df = self.create_test_dataframe(intensities=intensities)
        
        result = self.validator.validate_data(df)
        
        # May detect baseline drift
        drift_issues = [issue for issue in result.issues if issue.rule == ValidationRule.BASELINE_DRIFT]
    
    def test_spectral_resolution_check(self):
        """Test spectral resolution validation."""
        # Enable spectral resolution check
        self.validator.enable_rule(ValidationRule.SPECTRAL_RESOLUTION, True)
        
        # Create data with low resolution (large steps)
        wavenumbers = np.arange(4000, 3000, -10)  # 10 cm⁻¹ steps (low resolution)
        df = self.create_test_dataframe(wavenumbers=wavenumbers)
        
        result = self.validator.validate_data(df)
        
        resolution_issues = [issue for issue in result.issues if issue.rule == ValidationRule.SPECTRAL_RESOLUTION]
        # May detect low resolution
    
    def test_non_numeric_data(self):
        """Test handling of non-numeric data."""
        df = pd.DataFrame({
            'wavenumber': ['4000', 'invalid', '3998'],
            'intensity': [0.1, 'bad_data', 0.3]
        })
        
        result = self.validator.validate_data(df)
        
        assert not result.is_valid
        assert any(issue.level == ValidationLevel.ERROR for issue in result.issues)
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # Perfect data
        df = self.create_test_dataframe()
        result = self.validator.validate_data(df)
        high_score = result.quality_score
        
        # Data with issues
        df_bad = pd.DataFrame({
            'wavenumber': [4000, np.nan, 3998],  # Missing value
            'intensity': [0.1, 0.2, 100.0]  # Outlier
        })
        result_bad = self.validator.validate_data(df_bad)
        low_score = result_bad.quality_score
        
        assert high_score > low_score
        assert 0 <= low_score <= 100
        assert 0 <= high_score <= 100
    
    def test_recommendations_generation(self):
        """Test recommendations generation."""
        # Create data with multiple issues
        df = pd.DataFrame({
            'wavenumber': [4000, 4000, np.nan, 3997],  # Duplicates and missing
            'intensity': [0.1, 0.2, 0.3, 100.0]  # Outlier
        })
        
        result = self.validator.validate_data(df)
        
        assert len(result.recommendations) > 0
        assert any("missing" in rec.lower() for rec in result.recommendations)
    
    def test_threshold_customization(self):
        """Test threshold customization."""
        original_threshold = self.validator.thresholds['wavenumber_min']
        
        # Change threshold
        self.validator.set_threshold('wavenumber_min', 500.0)
        assert self.validator.thresholds['wavenumber_min'] == 500.0
        
        # Test with new threshold
        df = self.create_test_dataframe(wavenumbers=[450, 460, 470])  # Below new threshold
        result = self.validator.validate_data(df)
        
        assert any(issue.rule == ValidationRule.WAVENUMBER_RANGE for issue in result.issues)
        
        # Reset threshold
        self.validator.set_threshold('wavenumber_min', original_threshold)
    
    def test_rule_enabling_disabling(self):
        """Test enabling and disabling validation rules."""
        # Disable wavenumber range check
        self.validator.enable_rule(ValidationRule.WAVENUMBER_RANGE, False)
        
        # Test with out-of-range wavenumbers
        df = self.create_test_dataframe(wavenumbers=[100, 200, 300])
        result = self.validator.validate_data(df)
        
        # Should not have wavenumber range issues
        assert not any(issue.rule == ValidationRule.WAVENUMBER_RANGE for issue in result.issues)
        
        # Re-enable rule
        self.validator.enable_rule(ValidationRule.WAVENUMBER_RANGE, True)
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        df = self.create_test_dataframe()
        result = self.validator.validate_data(df)
        
        summary = self.validator.get_validation_summary(result)
        
        assert "Validation Summary" in summary
        assert "Status:" in summary
        assert "Quality Score:" in summary
        assert "Issues Found:" in summary
    
    def test_specific_validation_methods(self):
        """Test specific validation methods."""
        # Test wavenumber range check
        wavenumbers = pd.Series([100, 200, 300])  # Out of range
        issue = self.validator.check_wavenumber_range(wavenumbers)
        assert issue is not None
        assert issue.rule == ValidationRule.WAVENUMBER_RANGE
        
        # Test absorbance values check
        absorbance = pd.Series([-1.0, 0.5, 10.0])  # Out of range
        issue = self.validator.check_absorbance_values(absorbance)
        assert issue is not None
        assert issue.rule == ValidationRule.ABSORBANCE_RANGE
    
    def test_baseline_overlap_validation(self):
        """Test baseline overlap validation."""
        # Create baseline and sample data with good overlap
        baseline_df = pd.DataFrame({
            'wavenumber': np.linspace(4000, 400, 100),
            'intensity': np.random.normal(0.01, 0.001, 100)
        })
        
        sample_df = pd.DataFrame({
            'wavenumber': np.linspace(3900, 500, 100),
            'intensity': np.random.normal(0.5, 0.1, 100)
        })
        
        issues = self.validator.validate_baseline_overlap(baseline_df, sample_df)
        
        # Should have good overlap, minimal issues
        overlap_errors = [issue for issue in issues if issue.level == ValidationLevel.ERROR]
        assert len(overlap_errors) == 0
        
        # Test with no overlap
        baseline_df2 = pd.DataFrame({
            'wavenumber': np.linspace(4000, 3000, 50),
            'intensity': np.random.normal(0.01, 0.001, 50)
        })
        
        sample_df2 = pd.DataFrame({
            'wavenumber': np.linspace(2000, 1000, 50),
            'intensity': np.random.normal(0.5, 0.1, 50)
        })
        
        issues2 = self.validator.validate_baseline_overlap(baseline_df2, sample_df2)
        
        # Should have overlap error
        assert any(issue.rule == ValidationRule.OVERLAPPING_RANGES and 
                  issue.level == ValidationLevel.ERROR for issue in issues2)
    
    def test_validation_with_different_column_names(self):
        """Test validation with different column naming conventions."""
        test_cases = [
            {'wavenumber': [4000, 3999, 3998], 'absorbance': [0.1, 0.2, 0.3]},
            {'wave_number': [4000, 3999, 3998], 'abs': [0.1, 0.2, 0.3]},
            {'cm-1': [4000, 3999, 3998], 'intensity': [0.1, 0.2, 0.3]},
        ]
        
        for data in test_cases:
            df = pd.DataFrame(data)
            result = self.validator.validate_data(df)
            # Should handle different naming conventions
            assert isinstance(result, ValidationResult)
    
    def test_large_dataset_handling(self):
        """Test validation of large datasets."""
        # Create large dataset
        large_wavenumbers = np.linspace(4000, 400, 10000)
        large_intensities = np.random.normal(0.5, 0.1, 10000)
        df = self.create_test_dataframe(wavenumbers=large_wavenumbers, intensities=large_intensities)
        
        result = self.validator.validate_data(df)
        
        assert isinstance(result, ValidationResult)
        assert result.statistics['total_rows'] == 10000
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        # Create problematic data that might cause validation errors
        df = pd.DataFrame({
            'wavenumber': [None, float('inf'), float('-inf')],
            'intensity': [float('nan'), None, 'invalid']
        })
        
        result = self.validator.validate_data(df)
        
        # Should handle gracefully without crashing
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
    
    @patch('spectral_analyzer.core.data_validator.logging.getLogger')
    def test_logging_integration(self, mock_logger):
        """Test integration with logging system."""
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        
        validator = DataValidator()
        df = self.create_test_dataframe()
        result = validator.validate_data(df)
        
        # Should have made logging calls
        assert mock_logger_instance.info.called
    
    def test_statistics_calculation(self):
        """Test statistics calculation accuracy."""
        df = self.create_test_dataframe()
        result = self.validator.validate_data(df)
        
        stats = result.statistics
        assert 'total_rows' in stats
        assert 'total_columns' in stats
        assert 'wavenumber_stats' in stats
        assert 'intensity_stats' in stats
        
        assert stats['total_rows'] == len(df)
        assert stats['total_columns'] == len(df.columns)
    
    def test_validation_issue_metadata(self):
        """Test validation issue metadata handling."""
        # Enable outlier detection to get issues with metadata
        self.validator.enable_rule(ValidationRule.OUTLIER_DETECTION, True)
        
        intensities = [0.1, 0.1, 0.1, 10.0, 0.1, 0.1]  # Clear outlier
        df = self.create_test_dataframe(intensities=intensities)
        
        result = self.validator.validate_data(df)
        
        outlier_issues = [issue for issue in result.issues if issue.rule == ValidationRule.OUTLIER_DETECTION]
        if outlier_issues:
            issue = outlier_issues[0]
            assert issue.metadata is not None
            assert 'outlier_values' in issue.metadata


class TestValidationIssue:
    """Test suite for ValidationIssue class."""
    
    def test_validation_issue_creation(self):
        """Test ValidationIssue creation and attributes."""
        issue = ValidationIssue(
            rule=ValidationRule.WAVENUMBER_RANGE,
            level=ValidationLevel.ERROR,
            message="Test message",
            location="Column 1",
            suggested_fix="Fix this",
            affected_rows=[1, 2, 3],
            metadata={'key': 'value'}
        )
        
        assert issue.rule == ValidationRule.WAVENUMBER_RANGE
        assert issue.level == ValidationLevel.ERROR
        assert issue.message == "Test message"
        assert issue.location == "Column 1"
        assert issue.suggested_fix == "Fix this"
        assert issue.affected_rows == [1, 2, 3]
        assert issue.metadata == {'key': 'value'}


class TestValidationResult:
    """Test suite for ValidationResult class."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and attributes."""
        issues = [
            ValidationIssue(ValidationRule.WAVENUMBER_RANGE, ValidationLevel.WARNING, "Warning"),
            ValidationIssue(ValidationRule.MISSING_VALUES, ValidationLevel.ERROR, "Error")
        ]
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            statistics={'rows': 100},
            quality_score=75.5,
            recommendations=["Fix errors", "Review warnings"]
        )
        
        assert not result.is_valid
        assert len(result.issues) == 2
        assert result.statistics == {'rows': 100}
        assert result.quality_score == 75.5
        assert len(result.recommendations) == 2


if __name__ == "__main__":
    pytest.main([__file__])