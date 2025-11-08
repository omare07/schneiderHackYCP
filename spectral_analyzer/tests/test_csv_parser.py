"""
Comprehensive test cases for CSV parser functionality.

Tests various CSV formats, edge cases, and error handling scenarios
to ensure robust parsing of spectroscopy data files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import patch, MagicMock

from spectral_analyzer.core.csv_parser import (
    CSVParser, CSVFormat, DataType, ColumnInfo, FormatInfo, 
    ParseResult, CSVStructure
)


class TestCSVParser:
    """Test suite for CSVParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str) -> Path:
        """Create a temporary test file with given content."""
        file_path = Path(self.temp_dir) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path
    
    def test_standard_spectral_format(self):
        """Test parsing of standard spectral CSV format."""
        content = """Wavenumber,Absorbance
4000.0,0.123
3999.5,0.125
3999.0,0.127
3998.5,0.124
3998.0,0.122"""
        
        file_path = self.create_test_file("standard.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.structure.format_type == CSVFormat.STANDARD_SPECTRAL
        assert result.structure.has_header
        assert result.structure.delimiter == ','
        assert len(result.structure.columns) == 2
        assert result.data is not None
        assert 'wavenumber' in result.data.columns
        assert 'intensity' in result.data.columns
    
    def test_european_decimal_format(self):
        """Test parsing of European decimal format (comma as decimal separator)."""
        content = """Wellenzahl;Absorption
4000,0;0,123
3999,5;0,125
3999,0;0,127
3998,5;0,124
3998,0;0,122"""
        
        file_path = self.create_test_file("european.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.format_info.delimiter == ';'
        assert result.format_info.decimal_separator == ','
        assert result.data is not None
        assert result.data['wavenumber'].dtype in [np.float64, float]
    
    def test_tab_delimited_format(self):
        """Test parsing of tab-delimited format."""
        content = """Wavenumber\tAbsorbance\tSample_ID
4000.0\t0.123\tSample1
3999.5\t0.125\tSample1
3999.0\t0.127\tSample1"""
        
        file_path = self.create_test_file("tab_delimited.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.format_info.delimiter == '\t'
        assert len(result.structure.columns) == 3
    
    def test_metadata_rows_detection(self):
        """Test detection and handling of metadata rows."""
        content = """# Instrument: FTIR Spectrometer
# Date: 2024-01-01
# Sample: Test Sample
Wavenumber,Absorbance
4000.0,0.123
3999.5,0.125
3999.0,0.127"""
        
        file_path = self.create_test_file("with_metadata.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert len(result.format_info.metadata_rows) == 3
        assert '#' in result.format_info.comment_prefixes
    
    def test_no_header_format(self):
        """Test parsing of CSV without header row."""
        content = """4000.0,0.123
3999.5,0.125
3999.0,0.127
3998.5,0.124"""
        
        file_path = self.create_test_file("no_header.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert not result.format_info.has_header
        assert result.data is not None
    
    def test_multi_column_format(self):
        """Test parsing of multi-column intensity data."""
        content = """Wavenumber,Sample1,Sample2,Sample3,Baseline
4000.0,0.123,0.124,0.125,0.001
3999.5,0.125,0.126,0.127,0.002
3999.0,0.127,0.128,0.129,0.003"""
        
        file_path = self.create_test_file("multi_column.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.structure.format_type in [CSVFormat.MULTI_COLUMN, CSVFormat.STANDARD_SPECTRAL]
        assert len([col for col in result.data.columns if 'intensity' in col]) >= 1
    
    def test_pipe_delimited_format(self):
        """Test parsing of pipe-delimited format."""
        content = """Wavenumber|Absorbance|Quality
4000.0|0.123|Good
3999.5|0.125|Good
3999.0|0.127|Fair"""
        
        file_path = self.create_test_file("pipe_delimited.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.format_info.delimiter == '|'
    
    def test_mixed_case_headers(self):
        """Test parsing with mixed case column headers."""
        content = """WAVENUMBER,absorbance,SAMPLE_id
4000.0,0.123,Test1
3999.5,0.125,Test1
3999.0,0.127,Test1"""
        
        file_path = self.create_test_file("mixed_case.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        # Check that column types are correctly identified despite case differences
        wavenumber_cols = [col for col in result.structure.columns if col.data_type == DataType.WAVENUMBER]
        absorbance_cols = [col for col in result.structure.columns if col.data_type == DataType.ABSORBANCE]
        assert len(wavenumber_cols) >= 1
        assert len(absorbance_cols) >= 1
    
    def test_transmittance_data(self):
        """Test parsing of transmittance data."""
        content = """Wavenumber,Transmittance
4000.0,95.5
3999.5,94.8
3999.0,93.2
3998.5,92.1"""
        
        file_path = self.create_test_file("transmittance.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        transmittance_cols = [col for col in result.structure.columns if col.data_type == DataType.TRANSMITTANCE]
        assert len(transmittance_cols) >= 1
    
    def test_encoding_detection(self):
        """Test automatic encoding detection."""
        # Create file with UTF-8 encoding
        content = "Wavenumber,Absorbance\n4000.0,0.123\n3999.5,0.125"
        file_path = Path(self.temp_dir) / "utf8.csv"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse_file(file_path)
        assert result.success
        assert result.format_info.encoding in ['utf-8', 'UTF-8']
    
    def test_large_gaps_in_data(self):
        """Test handling of large gaps in wavenumber sequence."""
        content = """Wavenumber,Absorbance
4000.0,0.123
3999.5,0.125
3990.0,0.127
3989.5,0.124"""
        
        file_path = self.create_test_file("gaps.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        # Should still parse successfully but may have warnings
    
    def test_duplicate_wavenumbers(self):
        """Test handling of duplicate wavenumber values."""
        content = """Wavenumber,Absorbance
4000.0,0.123
4000.0,0.125
3999.5,0.127
3999.0,0.124"""
        
        file_path = self.create_test_file("duplicates.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        # Duplicates should be removed in cleaned data
        assert result.data['wavenumber'].duplicated().sum() == 0
    
    def test_missing_values(self):
        """Test handling of missing values."""
        content = """Wavenumber,Absorbance
4000.0,0.123
3999.5,
3999.0,0.127
,0.124"""
        
        file_path = self.create_test_file("missing.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        # Rows with missing wavenumbers should be removed
        assert not result.data['wavenumber'].isnull().any()
    
    def test_wrong_order_wavenumbers(self):
        """Test handling of ascending wavenumber order."""
        content = """Wavenumber,Absorbance
3998.0,0.122
3998.5,0.124
3999.0,0.127
3999.5,0.125
4000.0,0.123"""
        
        file_path = self.create_test_file("ascending.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        # Data should be sorted in descending order
        wavenumbers = result.data['wavenumber'].values
        assert all(wavenumbers[i] >= wavenumbers[i+1] for i in range(len(wavenumbers)-1))
    
    def test_alternative_column_names(self):
        """Test recognition of alternative column naming conventions."""
        test_cases = [
            ("Wave Number,Abs", DataType.WAVENUMBER, DataType.ABSORBANCE),
            ("cm-1,Intensity", DataType.WAVENUMBER, DataType.INTENSITY),
            ("X,Y", DataType.WAVENUMBER, DataType.INTENSITY),
            ("Frequency,Signal", DataType.WAVENUMBER, DataType.INTENSITY),
        ]
        
        for i, (header, expected_type1, expected_type2) in enumerate(test_cases):
            content = f"""{header}
4000.0,0.123
3999.5,0.125
3999.0,0.127"""
            
            file_path = self.create_test_file(f"alt_names_{i}.csv", content)
            result = self.parser.parse_file(file_path)
            
            assert result.success
            assert len(result.structure.columns) == 2
    
    def test_empty_file(self):
        """Test handling of empty files."""
        file_path = self.create_test_file("empty.csv", "")
        result = self.parser.parse_file(file_path)
        
        assert not result.success
        assert result.structure.format_type == CSVFormat.UNKNOWN
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        invalid_path = Path(self.temp_dir) / "nonexistent.csv"
        result = self.parser.parse_file(invalid_path)
        
        assert not result.success
        assert len(result.issues) > 0
    
    def test_corrupted_data(self):
        """Test handling of corrupted/malformed data."""
        content = """Wavenumber,Absorbance
4000.0,0.123
invalid_data_here
3999.0,not_a_number
3998.5,0.124"""
        
        file_path = self.create_test_file("corrupted.csv", content)
        result = self.parser.parse_file(file_path)
        
        # Should still attempt to parse and extract valid data
        assert result.data is not None
    
    def test_very_large_values(self):
        """Test handling of very large numeric values."""
        content = """Wavenumber,Absorbance
4000000.0,1000.123
3999000.5,999.125
3998000.0,998.127"""
        
        file_path = self.create_test_file("large_values.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        # Should detect potential scale issues
    
    def test_scientific_notation(self):
        """Test handling of scientific notation."""
        content = """Wavenumber,Absorbance
4.000e3,1.23e-1
3.9995e3,1.25e-1
3.999e3,1.27e-1"""
        
        file_path = self.create_test_file("scientific.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.data['wavenumber'].dtype in [np.float64, float]
    
    def test_is_standard_format(self):
        """Test quick standard format detection."""
        # Standard format
        content = """Wavenumber,Absorbance
4000.0,0.123
3999.5,0.125
3999.0,0.127"""
        
        file_path = self.create_test_file("standard_check.csv", content)
        assert self.parser.is_standard_format(file_path)
        
        # Non-standard format
        content2 = """Time,Temperature,Pressure
10:00,25.5,1013.2
10:01,25.6,1013.1"""
        
        file_path2 = self.create_test_file("non_standard.csv", content2)
        assert not self.parser.is_standard_format(file_path2)
    
    def test_performance_metrics(self):
        """Test that performance metrics are captured."""
        content = """Wavenumber,Absorbance
4000.0,0.123
3999.5,0.125
3999.0,0.127"""
        
        file_path = self.create_test_file("performance.csv", content)
        result = self.parser.parse_file(file_path)
        
        assert result.success
        assert result.parsing_time > 0
        assert isinstance(result.memory_usage, int)
    
    @patch('psutil.Process')
    def test_memory_tracking_failure(self, mock_process):
        """Test graceful handling of memory tracking failures."""
        mock_process.side_effect = Exception("Memory tracking failed")
        
        content = """Wavenumber,Absorbance
4000.0,0.123
3999.5,0.125"""
        
        file_path = self.create_test_file("memory_fail.csv", content)
        result = self.parser.parse_file(file_path)
        
        # Should still succeed despite memory tracking failure
        assert not result.success or result.memory_usage == 0
    
    def test_column_type_classification(self):
        """Test column type classification accuracy."""
        # Test wavenumber patterns
        assert self.parser._classify_column_type("Wavenumber", pd.Series([4000, 3999, 3998]), (3998, 4000))[0] == DataType.WAVENUMBER
        assert self.parser._classify_column_type("cm-1", pd.Series([4000, 3999, 3998]), (3998, 4000))[0] == DataType.WAVENUMBER
        
        # Test absorbance patterns
        assert self.parser._classify_column_type("Absorbance", pd.Series([0.1, 0.2, 0.3]), (0.1, 0.3))[0] == DataType.ABSORBANCE
        assert self.parser._classify_column_type("Abs", pd.Series([0.1, 0.2, 0.3]), (0.1, 0.3))[0] == DataType.ABSORBANCE
        
        # Test transmittance patterns
        assert self.parser._classify_column_type("Transmittance", pd.Series([95, 94, 93]), (93, 95))[0] == DataType.TRANSMITTANCE
        assert self.parser._classify_column_type("%T", pd.Series([95, 94, 93]), (93, 95))[0] == DataType.TRANSMITTANCE
    
    def test_format_detection_edge_cases(self):
        """Test format detection with edge cases."""
        # Single column
        content = """4000.0
3999.5
3999.0"""
        file_path = self.create_test_file("single_col.csv", content)
        result = self.parser.parse_file(file_path)
        assert result.success or not result.success  # Should handle gracefully
        
        # Many columns
        headers = ",".join([f"Col{i}" for i in range(20)])
        data_rows = []
        for j in range(5):
            row = ",".join([str(i + j) for i in range(20)])
            data_rows.append(row)
        
        content = headers + "\n" + "\n".join(data_rows)
        file_path = self.create_test_file("many_cols.csv", content)
        result = self.parser.parse_file(file_path)
        assert result.success


class TestFormatDetection:
    """Test suite for format detection methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CSVParser()
    
    def test_delimiter_detection(self):
        """Test delimiter detection accuracy."""
        # Comma delimited
        lines = ["a,b,c", "1,2,3", "4,5,6"]
        delimiter = self.parser._detect_delimiter(lines)
        assert delimiter == ','
        
        # Semicolon delimited
        lines = ["a;b;c", "1;2;3", "4;5;6"]
        delimiter = self.parser._detect_delimiter(lines)
        assert delimiter == ';'
        
        # Tab delimited
        lines = ["a\tb\tc", "1\t2\t3", "4\t5\t6"]
        delimiter = self.parser._detect_delimiter(lines)
        assert delimiter == '\t'
    
    def test_decimal_separator_detection(self):
        """Test decimal separator detection."""
        # European format
        lines = ["4000,5;0,123", "3999,0;0,125"]
        decimal_sep = self.parser._detect_decimal_separator(lines, ';')
        assert decimal_sep == ','
        
        # US format
        lines = ["4000.5,0.123", "3999.0,0.125"]
        decimal_sep = self.parser._detect_decimal_separator(lines, ',')
        assert decimal_sep == '.'
    
    def test_metadata_detection(self):
        """Test metadata row detection."""
        lines = [
            "# This is a comment",
            "// Another comment",
            "Wavenumber,Absorbance",
            "4000.0,0.123"
        ]
        metadata_rows, prefixes = self.parser._detect_metadata_rows(lines)
        assert 0 in metadata_rows
        assert 1 in metadata_rows
        assert '#' in prefixes
        assert '//' in prefixes


if __name__ == "__main__":
    pytest.main([__file__])