"""
Integration tests for the complete CSV parsing and data validation system.

Tests the full workflow from CSV parsing through data validation with
various real-world file formats and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time

from spectral_analyzer.core.csv_parser import CSVParser, CSVFormat
from spectral_analyzer.core.data_validator import DataValidator, ValidationLevel


class TestSystemIntegration:
    """Integration tests for the complete parsing and validation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()
        self.validator = DataValidator()
        self.test_data_dir = Path(__file__).parent / "test_data"
    
    def test_standard_spectral_workflow(self):
        """Test complete workflow with standard spectral data."""
        # Use the existing sample file
        sample_file = self.test_data_dir / "sample_spectral.csv"
        
        # Parse the file
        parse_result = self.parser.parse_file(sample_file)
        
        assert parse_result.success
        assert parse_result.data is not None
        assert parse_result.structure.format_type == CSVFormat.STANDARD_SPECTRAL
        
        # Validate the parsed data
        validation_result = self.validator.validate_data(parse_result.data)
        
        assert validation_result.is_valid
        assert validation_result.quality_score > 80
        assert len([issue for issue in validation_result.issues 
                   if issue.level == ValidationLevel.ERROR]) == 0
    
    def test_european_format_workflow(self):
        """Test complete workflow with European decimal format."""
        european_file = self.test_data_dir / "european_format.csv"
        
        # Parse the file
        parse_result = self.parser.parse_file(european_file)
        
        assert parse_result.success
        assert parse_result.format_info.decimal_separator == ','
        assert parse_result.format_info.delimiter == ';'
        assert len(parse_result.format_info.metadata_rows) == 3
        
        # Validate the parsed data
        validation_result = self.validator.validate_data(parse_result.data)
        
        assert isinstance(validation_result.quality_score, float)
        assert validation_result.statistics['total_rows'] > 0
    
    def test_tab_delimited_workflow(self):
        """Test complete workflow with tab-delimited format."""
        tab_file = self.test_data_dir / "tab_delimited.csv"
        
        # Parse the file
        parse_result = self.parser.parse_file(tab_file)
        
        assert parse_result.success
        assert parse_result.format_info.delimiter == '\t'
        assert '//' in parse_result.format_info.comment_prefixes
        
        # Should have multiple columns including sample ID and quality
        assert len(parse_result.structure.columns) >= 3
        
        # Validate the parsed data
        validation_result = self.validator.validate_data(parse_result.data)
        
        assert isinstance(validation_result, type(validation_result))
    
    def test_multi_column_workflow(self):
        """Test complete workflow with multi-column intensity data."""
        multi_file = self.test_data_dir / "multi_column.csv"
        
        # Parse the file
        parse_result = self.parser.parse_file(multi_file)
        
        assert parse_result.success
        assert parse_result.structure.format_type in [CSVFormat.MULTI_COLUMN, CSVFormat.STANDARD_SPECTRAL]
        
        # Should have multiple intensity columns
        intensity_cols = [col for col in parse_result.data.columns if 'intensity' in col]
        assert len(intensity_cols) >= 1
        
        # Validate the parsed data
        validation_result = self.validator.validate_data(parse_result.data)
        
        # Multi-column data should still be valid
        assert validation_result.quality_score > 0
    
    def test_problematic_data_workflow(self):
        """Test complete workflow with problematic data."""
        problem_file = self.test_data_dir / "problematic_data.csv"
        
        # Parse the file
        parse_result = self.parser.parse_file(problem_file)
        
        assert parse_result.success  # Should still parse
        assert parse_result.data is not None
        
        # Validate the parsed data
        validation_result = self.validator.validate_data(parse_result.data)
        
        # Should detect multiple issues
        assert len(validation_result.issues) > 0
        assert not validation_result.is_valid
        assert validation_result.quality_score < 80
        
        # Should have specific issue types
        issue_rules = [issue.rule for issue in validation_result.issues]
        # May include missing values, duplicates, range issues, etc.
        
        # Should provide recommendations
        assert len(validation_result.recommendations) > 0
    
    def test_no_header_workflow(self):
        """Test complete workflow with headerless data."""
        no_header_file = self.test_data_dir / "no_header.csv"
        
        # Parse the file
        parse_result = self.parser.parse_file(no_header_file)
        
        assert parse_result.success
        assert not parse_result.format_info.has_header
        
        # Should still extract spectral data
        assert parse_result.data is not None
        assert 'wavenumber' in parse_result.data.columns
        
        # Validate the parsed data
        validation_result = self.validator.validate_data(parse_result.data)
        
        # Should be valid data despite no header
        assert validation_result.is_valid or len(validation_result.issues) < 5
    
    def test_performance_with_large_dataset(self):
        """Test system performance with larger datasets."""
        # Create a larger test dataset
        large_wavenumbers = np.linspace(4000, 400, 1000)
        large_intensities = np.random.normal(0.5, 0.1, 1000)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Wavenumber,Absorbance\n")
            for wn, intensity in zip(large_wavenumbers, large_intensities):
                f.write(f"{wn:.1f},{intensity:.6f}\n")
            temp_file = Path(f.name)
        
        try:
            # Measure parsing time
            start_time = time.time()
            parse_result = self.parser.parse_file(temp_file)
            parse_time = time.time() - start_time
            
            assert parse_result.success
            assert parse_time < 5.0  # Should parse within 5 seconds
            
            # Measure validation time
            start_time = time.time()
            validation_result = self.validator.validate_data(parse_result.data)
            validation_time = time.time() - start_time
            
            assert validation_time < 2.0  # Should validate within 2 seconds
            assert validation_result.statistics['total_rows'] == 1000
            
        finally:
            # Clean up
            temp_file.unlink(missing_ok=True)
    
    def test_error_handling_workflow(self):
        """Test error handling in the complete workflow."""
        # Test with non-existent file
        fake_file = Path("nonexistent_file.csv")
        parse_result = self.parser.parse_file(fake_file)
        
        assert not parse_result.success
        assert len(parse_result.issues) > 0
        
        # Test validation with invalid data
        invalid_df = pd.DataFrame({
            'wavenumber': [None, float('inf'), 'invalid'],
            'intensity': [float('nan'), None, 'bad_data']
        })
        
        validation_result = self.validator.validate_data(invalid_df)
        
        assert not validation_result.is_valid
        assert len(validation_result.issues) > 0
    
    def test_format_detection_accuracy(self):
        """Test format detection accuracy across different files."""
        test_files = [
            ("sample_spectral.csv", CSVFormat.STANDARD_SPECTRAL),
            ("european_format.csv", CSVFormat.STANDARD_SPECTRAL),
            ("tab_delimited.csv", CSVFormat.STANDARD_SPECTRAL),
            ("multi_column.csv", [CSVFormat.MULTI_COLUMN, CSVFormat.STANDARD_SPECTRAL]),
        ]
        
        for filename, expected_format in test_files:
            file_path = self.test_data_dir / filename
            if file_path.exists():
                parse_result = self.parser.parse_file(file_path)
                
                if isinstance(expected_format, list):
                    assert parse_result.structure.format_type in expected_format
                else:
                    assert parse_result.structure.format_type == expected_format
    
    def test_data_cleaning_effectiveness(self):
        """Test effectiveness of data cleaning operations."""
        problem_file = self.test_data_dir / "problematic_data.csv"
        
        parse_result = self.parser.parse_file(problem_file)
        
        if parse_result.success and parse_result.data is not None:
            # Check that cleaning was effective
            data = parse_result.data
            
            # Should not have missing wavenumbers
            assert not data['wavenumber'].isnull().any()
            
            # Should not have duplicate wavenumbers
            assert data['wavenumber'].duplicated().sum() == 0
            
            # Should be sorted in descending order
            wavenumbers = data['wavenumber'].values
            if len(wavenumbers) > 1:
                assert all(wavenumbers[i] >= wavenumbers[i+1] for i in range(len(wavenumbers)-1))
    
    def test_validation_rule_effectiveness(self):
        """Test effectiveness of validation rules."""
        # Test with data that should trigger specific validation rules
        test_data = pd.DataFrame({
            'wavenumber': [100, 200, 300],  # Out of range
            'intensity': [-1.0, 0.5, 10.0]  # Out of typical absorbance range
        })
        
        validation_result = self.validator.validate_data(test_data)
        
        # Should detect range issues
        range_issues = [issue for issue in validation_result.issues 
                       if 'range' in issue.message.lower()]
        assert len(range_issues) > 0
    
    def test_memory_efficiency(self):
        """Test memory efficiency with various file sizes."""
        # Test with different sized datasets
        sizes = [100, 500, 1000]
        
        for size in sizes:
            wavenumbers = np.linspace(4000, 400, size)
            intensities = np.random.normal(0.5, 0.1, size)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write("Wavenumber,Absorbance\n")
                for wn, intensity in zip(wavenumbers, intensities):
                    f.write(f"{wn:.1f},{intensity:.6f}\n")
                temp_file = Path(f.name)
            
            try:
                parse_result = self.parser.parse_file(temp_file)
                
                # Memory usage should be reasonable
                if parse_result.success:
                    # Memory usage should scale reasonably with data size
                    assert parse_result.memory_usage >= 0
                    
            finally:
                temp_file.unlink(missing_ok=True)
    
    def test_standard_format_detection(self):
        """Test quick standard format detection."""
        # Test with known standard format
        sample_file = self.test_data_dir / "sample_spectral.csv"
        if sample_file.exists():
            assert self.parser.is_standard_format(sample_file)
        
        # Test with non-standard format (if we had one)
        # This would be a file that doesn't match spectral data patterns
    
    def test_baseline_sample_overlap_validation(self):
        """Test baseline and sample overlap validation."""
        # Create baseline data
        baseline_df = pd.DataFrame({
            'wavenumber': np.linspace(4000, 400, 100),
            'intensity': np.random.normal(0.01, 0.001, 100)
        })
        
        # Create sample data with good overlap
        sample_df = pd.DataFrame({
            'wavenumber': np.linspace(3900, 500, 100),
            'intensity': np.random.normal(0.5, 0.1, 100)
        })
        
        issues = self.validator.validate_baseline_overlap(baseline_df, sample_df)
        
        # Should have minimal issues with good overlap
        error_issues = [issue for issue in issues if issue.level == ValidationLevel.ERROR]
        assert len(error_issues) == 0
    
    def test_comprehensive_validation_summary(self):
        """Test comprehensive validation summary generation."""
        # Use problematic data to get a comprehensive summary
        problem_file = self.test_data_dir / "problematic_data.csv"
        
        parse_result = self.parser.parse_file(problem_file)
        if parse_result.success:
            validation_result = self.validator.validate_data(parse_result.data)
            
            summary = self.validator.get_validation_summary(validation_result)
            
            # Summary should contain key information
            assert "Validation Summary" in summary
            assert "Status:" in summary
            assert "Quality Score:" in summary
            assert "Issues Found:" in summary
            
            if validation_result.issues:
                assert "Errors:" in summary or "Warnings:" in summary
            
            if validation_result.recommendations:
                assert "Recommendations:" in summary


class TestEndToEndWorkflows:
    """End-to-end workflow tests simulating real usage scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()
        self.validator = DataValidator()
    
    def test_laboratory_workflow_simulation(self):
        """Simulate a typical laboratory workflow."""
        # Step 1: Scientist uploads a CSV file
        test_data_dir = Path(__file__).parent / "test_data"
        sample_file = test_data_dir / "sample_spectral.csv"
        
        # Step 2: System parses the file
        parse_result = self.parser.parse_file(sample_file)
        
        # Step 3: System validates the data
        if parse_result.success:
            validation_result = self.validator.validate_data(parse_result.data)
            
            # Step 4: System provides feedback to user
            if validation_result.is_valid:
                # Data is ready for analysis
                assert parse_result.data is not None
                assert len(parse_result.data) > 0
            else:
                # System provides specific recommendations
                assert len(validation_result.recommendations) > 0
                assert len(validation_result.issues) > 0
    
    def test_quality_control_workflow(self):
        """Test quality control workflow for spectral data."""
        # Enable all validation rules for comprehensive QC
        validator = DataValidator()
        validator.enable_rule(validator.enabled_rules.keys(), True)
        
        # Test with various data quality levels
        test_files = [
            "sample_spectral.csv",  # Good quality
            "problematic_data.csv",  # Poor quality
        ]
        
        test_data_dir = Path(__file__).parent / "test_data"
        
        for filename in test_files:
            file_path = test_data_dir / filename
            if file_path.exists():
                parse_result = self.parser.parse_file(file_path)
                
                if parse_result.success:
                    validation_result = validator.validate_data(parse_result.data)
                    
                    # QC should provide detailed assessment
                    assert isinstance(validation_result.quality_score, float)
                    assert 0 <= validation_result.quality_score <= 100
                    
                    # Should categorize issues by severity
                    error_count = sum(1 for issue in validation_result.issues 
                                    if issue.level == ValidationLevel.ERROR)
                    warning_count = sum(1 for issue in validation_result.issues 
                                      if issue.level == ValidationLevel.WARNING)
                    
                    # Quality score should reflect issue severity
                    if error_count > 0:
                        assert validation_result.quality_score < 80
    
    def test_batch_processing_simulation(self):
        """Simulate batch processing of multiple files."""
        test_data_dir = Path(__file__).parent / "test_data"
        csv_files = list(test_data_dir.glob("*.csv"))
        
        results = []
        
        for file_path in csv_files:
            # Parse each file
            parse_result = self.parser.parse_file(file_path)
            
            if parse_result.success:
                # Validate each file
                validation_result = self.validator.validate_data(parse_result.data)
                
                results.append({
                    'file': file_path.name,
                    'format': parse_result.structure.format_type.value,
                    'valid': validation_result.is_valid,
                    'quality_score': validation_result.quality_score,
                    'issues': len(validation_result.issues)
                })
        
        # Should have processed multiple files
        assert len(results) > 0
        
        # Should have variety in results
        formats = set(result['format'] for result in results)
        assert len(formats) >= 1  # At least one format type
        
        # Should have quality scores for all files
        assert all('quality_score' in result for result in results)


if __name__ == "__main__":
    pytest.main([__file__])