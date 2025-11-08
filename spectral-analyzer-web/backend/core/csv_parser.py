"""
CSV parsing engine for spectroscopy data files.

Handles diverse CSV formats from laboratory instruments with intelligent
format detection and data extraction capabilities.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import chardet
from concurrent.futures import ThreadPoolExecutor
import threading


class CSVFormat(Enum):
    """Detected CSV format types."""
    STANDARD_SPECTRAL = "standard_spectral"
    INSTRUMENT_SPECIFIC = "instrument_specific"
    MULTI_COLUMN = "multi_column"
    TRANSPOSED = "transposed"
    UNKNOWN = "unknown"


class DataType(Enum):
    """Data column types."""
    WAVENUMBER = "wavenumber"
    FREQUENCY = "frequency"
    WAVELENGTH = "wavelength"
    ABSORBANCE = "absorbance"
    TRANSMITTANCE = "transmittance"
    INTENSITY = "intensity"
    SAMPLE_ID = "sample_id"
    METADATA = "metadata"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a CSV column."""
    index: int
    name: str
    data_type: DataType
    sample_values: List[Any]
    numeric_range: Optional[Tuple[float, float]] = None
    has_missing: bool = False
    confidence: float = 0.0


@dataclass
class FormatInfo:
    """Details about detected CSV format."""
    delimiter: str
    encoding: str
    has_header: bool
    decimal_separator: str
    thousands_separator: Optional[str]
    quote_char: str
    escape_char: Optional[str]
    line_terminator: str
    metadata_rows: List[int]
    comment_prefixes: List[str]


@dataclass
class ParseResult:
    """Result of CSV parsing operation."""
    success: bool
    data: Optional[pd.DataFrame]
    format_info: FormatInfo
    structure: 'CSVStructure'
    issues: List[str]
    warnings: List[str]
    parsing_time: float
    memory_usage: int
    error: Optional[str] = None


@dataclass
class CSVStructure:
    """Structure analysis of a CSV file."""
    file_path: Path
    encoding: str
    delimiter: str
    has_header: bool
    row_count: int
    column_count: int
    columns: List[ColumnInfo]
    format_type: CSVFormat
    confidence: float
    issues: List[str]
    metadata_rows: List[int]


class CSVParser:
    """
    Advanced CSV parser for spectroscopy data files.
    
    Features:
    - Automatic format detection
    - Encoding detection
    - Column type identification
    - Data validation
    - Metadata extraction
    """
    
    def __init__(self):
        """Initialize the CSV parser."""
        self.logger = logging.getLogger(__name__)
        
        # Common spectroscopy column patterns
        self.wavenumber_patterns = [
            r'wave\s*number', r'wavenumber', r'cm-1', r'cm\^-1',
            r'frequency', r'x\s*axis', r'x-axis', r'wellenzahl'
        ]
        
        self.absorbance_patterns = [
            r'absorbance', r'abs', r'a\s*value', r'y\s*axis', r'y-axis',
            r'intensity', r'signal', r'absorption', r'extinktion'
        ]
        
        self.transmittance_patterns = [
            r'transmittance', r'trans', r't\s*value', r'%t', r'percent\s*t',
            r'transmission', r'durchlässigkeit'
        ]
        
        # Comment line prefixes
        self.comment_prefixes = ['#', '//', ';', '%', '*', '!']
        
        # European number format patterns
        self.european_decimal_pattern = re.compile(r'^\d+,\d+$')
        self.us_decimal_pattern = re.compile(r'^\d+\.\d+$')
        
        # Compile regex patterns
        self.wavenumber_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.wavenumber_patterns]
        self.absorbance_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.absorbance_patterns]
        self.transmittance_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.transmittance_patterns]
    
    def parse_file(self, file_path: Union[str, Path], preview_rows: int = 100,
                   progress_callback: Optional[Callable[[str, float], None]] = None) -> ParseResult:
        """
        Parse a CSV file and analyze its structure.
        
        Args:
            file_path: Path to the CSV file
            preview_rows: Number of rows to analyze for structure detection
            progress_callback: Optional callback for progress reporting (message, percentage)
            
        Returns:
            ParseResult object containing analysis results and data
        """
        import time
        import psutil
        import os
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        file_path = Path(file_path)
        
        try:
            self.logger.info(f"Parsing CSV file: {file_path}")
            
            if progress_callback:
                progress_callback("Starting file analysis...", 0.0)
            
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            if progress_callback:
                progress_callback("Encoding detected", 10.0)
            
            # Detect format details
            format_info = self._detect_format_details(file_path, encoding)
            if progress_callback:
                progress_callback("Format analysis complete", 25.0)
            
            # Read preview data
            preview_data = self._read_preview(file_path, encoding, format_info.delimiter, preview_rows)
            if progress_callback:
                progress_callback("Preview data loaded", 40.0)
            
            # Analyze structure
            structure = self._analyze_structure(
                file_path, encoding, format_info.delimiter, format_info.has_header, preview_data
            )
            if progress_callback:
                progress_callback("Structure analysis complete", 60.0)
            
            # Extract full data if structure is valid
            data = None
            if structure.format_type != CSVFormat.UNKNOWN:
                if progress_callback:
                    progress_callback("Extracting spectral data...", 70.0)
                data = self.extract_spectral_data(structure, format_info, progress_callback)
            
            # Calculate performance metrics
            parsing_time = time.time() - start_time
            final_memory = process.memory_info().rss
            memory_usage = final_memory - initial_memory
            
            # Determine success - be more lenient
            is_successful = (
                data is not None and
                not data.empty and
                len(data.columns) >= 2  # At least 2 columns for spectral data
            )
            
            result = ParseResult(
                success=is_successful,
                data=data,
                format_info=format_info,
                structure=structure,
                issues=structure.issues,
                warnings=[],
                parsing_time=parsing_time,
                memory_usage=memory_usage,
                error=None if is_successful else "Failed to extract valid spectral data"
            )
            
            self.logger.info(f"Successfully parsed CSV: {structure.format_type.value} in {parsing_time:.2f}s")
            return result
            
        except Exception as e:
            parsing_time = time.time() - start_time
            self.logger.error(f"Failed to parse CSV file {file_path}: {e}")
            
            return ParseResult(
                success=False,
                data=None,
                format_info=FormatInfo(
                    delimiter=',', encoding='utf-8', has_header=False,
                    decimal_separator='.', thousands_separator=None,
                    quote_char='"', escape_char=None, line_terminator='\n',
                    metadata_rows=[], comment_prefixes=[]
                ),
                structure=self._create_error_structure(file_path, str(e)),
                issues=[f"Parsing failed: {e}"],
                warnings=[],
                parsing_time=parsing_time,
                memory_usage=0,
                error=str(e)
            )
    
    def extract_spectral_data(self, structure: CSVStructure, format_info: FormatInfo,
                             progress_callback: Optional[Callable[[str, float], None]] = None) -> pd.DataFrame:
        """
        Extract spectral data from CSV based on structure analysis.
        
        Args:
            structure: CSVStructure from parse_file
            format_info: FormatInfo with format details
            progress_callback: Optional callback for progress reporting
            
        Returns:
            DataFrame with standardized spectral data
        """
        try:
            # Read full data with proper format handling
            if progress_callback:
                progress_callback("Reading full dataset...", 75.0)
            df = self._read_full_data(structure.file_path, format_info)
            
            # Skip metadata rows
            if structure.metadata_rows:
                df = df.drop(structure.metadata_rows)
            
            if progress_callback:
                progress_callback("Extracting spectral columns...", 85.0)
            # Extract spectral columns
            spectral_data = self._extract_spectral_columns(df, structure, format_info)
            
            if progress_callback:
                progress_callback("Cleaning and validating data...", 95.0)
            # Validate and clean data
            spectral_data = self._clean_spectral_data(spectral_data)
            
            if progress_callback:
                progress_callback("Data extraction complete", 100.0)
            
            self.logger.info(f"Extracted spectral data: {spectral_data.shape}")
            return spectral_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract spectral data: {e}")
            raise
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            
            if encoding is None or result['confidence'] < 0.7:
                encoding = 'utf-8'
            
            self.logger.debug(f"Detected encoding: {encoding} (confidence: {result.get('confidence', 0)})")
            return encoding
            
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _detect_format_details(self, file_path: Path, encoding: str) -> FormatInfo:
        """Detect comprehensive CSV format details."""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Read first 20 lines for analysis
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    sample_lines.append(line.strip())
                sample_lines = [line for line in sample_lines if line]
            
            if not sample_lines:
                return self._default_format_info()
            
            # Detect delimiter
            delimiter = self._detect_delimiter(sample_lines)
            
            # Detect header
            has_header = self._detect_header(sample_lines[0], delimiter)
            
            # Detect decimal separator
            decimal_separator = self._detect_decimal_separator(sample_lines, delimiter)
            
            # Detect metadata rows and comment prefixes
            metadata_rows, comment_prefixes = self._detect_metadata_rows(sample_lines)
            
            return FormatInfo(
                delimiter=delimiter,
                encoding=encoding,
                has_header=has_header,
                decimal_separator=decimal_separator,
                thousands_separator=',' if decimal_separator == '.' else '.',
                quote_char='"',
                escape_char=None,
                line_terminator='\n',
                metadata_rows=metadata_rows,
                comment_prefixes=comment_prefixes
            )
            
        except Exception as e:
            self.logger.warning(f"Format detection failed: {e}, using defaults")
            return self._default_format_info()
    
    def _detect_delimiter(self, sample_lines: List[str]) -> str:
        """Detect CSV delimiter from sample lines."""
        delimiters = ['\t', ',', ';', '|']  # Tab first - most specific
        delimiter_scores = {}
        
        # Filter out comment lines
        data_lines = [line for line in sample_lines[:10]
                      if line and not any(line.startswith(prefix) for prefix in self.comment_prefixes)]
        
        if not data_lines:
            return ','
        
        for delimiter in delimiters:
            scores = []
            for line in data_lines:
                parts = line.split(delimiter)
                # Count only if we get multiple parts
                if len(parts) > 1:
                    scores.append(len(parts))
            
            if scores and len(scores) >= len(data_lines) * 0.8:  # At least 80% of lines should split
                # Consistent column count is good
                consistency = 1.0 - (np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                avg_columns = np.mean(scores)
                # Bonus for tabs and semicolons as they're more specific
                specificity_bonus = 1.5 if delimiter in ['\t', ';'] else 1.0
                delimiter_scores[delimiter] = consistency * avg_columns * specificity_bonus
        
        if delimiter_scores:
            best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            delimiter_name = repr(best_delimiter) if best_delimiter == '\t' else best_delimiter
            self.logger.debug(f"Detected delimiter: {delimiter_name} (score: {delimiter_scores[best_delimiter]:.2f})")
            return best_delimiter
        
        return ','
    
    def _detect_decimal_separator(self, sample_lines: List[str], delimiter: str) -> str:
        """Detect decimal separator (. or ,) from numeric data."""
        european_count = 0
        us_count = 0
        
        for line in sample_lines[:10]:  # Check first 10 lines
            if not line or any(line.startswith(prefix) for prefix in self.comment_prefixes):
                continue
            
            parts = line.split(delimiter)
            for part in parts:
                part = part.strip().strip('"\'')
                if self.european_decimal_pattern.match(part):
                    european_count += 1
                elif self.us_decimal_pattern.match(part):
                    us_count += 1
        
        decimal_separator = ',' if european_count > us_count else '.'
        self.logger.debug(f"Detected decimal separator: '{decimal_separator}' (EU: {european_count}, US: {us_count})")
        return decimal_separator
    
    def _detect_metadata_rows(self, sample_lines: List[str]) -> Tuple[List[int], List[str]]:
        """Detect metadata rows and comment prefixes."""
        metadata_rows = []
        found_prefixes = []
        
        for i, line in enumerate(sample_lines):
            for prefix in self.comment_prefixes:
                if line.startswith(prefix):
                    metadata_rows.append(i)
                    if prefix not in found_prefixes:
                        found_prefixes.append(prefix)
                    break
        
        return metadata_rows, found_prefixes
    
    def _default_format_info(self) -> FormatInfo:
        """Return default format info."""
        return FormatInfo(
            delimiter=',',
            encoding='utf-8',
            has_header=True,
            decimal_separator='.',
            thousands_separator=',',
            quote_char='"',
            escape_char=None,
            line_terminator='\n',
            metadata_rows=[],
            comment_prefixes=[]
        )
    
    def _detect_header(self, first_line: str, delimiter: str) -> bool:
        """Detect if first line is a header."""
        try:
            values = first_line.split(delimiter)
            numeric_count = 0
            
            for value in values:
                value = value.strip()
                try:
                    float(value)
                    numeric_count += 1
                except ValueError:
                    pass
            
            # If less than 50% of values are numeric, likely a header
            return numeric_count / len(values) < 0.5
            
        except Exception:
            return True
    
    def _read_preview(self, file_path: Path, encoding: str, delimiter: str, rows: int) -> pd.DataFrame:
        """Read preview data from CSV."""
        try:
            # First, try to detect comment lines
            with open(file_path, 'r', encoding=encoding) as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= rows + 10:  # Read extra to account for comments
                        break
                    lines.append(line.strip())
            
            # Filter out comment lines
            data_lines = [line for line in lines
                         if line and not any(line.startswith(prefix) for prefix in self.comment_prefixes)]
            
            if not data_lines:
                return pd.DataFrame()
            
            # Determine comment character for pandas
            comment_char = None
            if self.comment_prefixes:
                # Use the first occurring comment prefix
                for line in lines:
                    for prefix in self.comment_prefixes:
                        if line.startswith(prefix):
                            comment_char = prefix
                            break
                    if comment_char:
                        break
            
            # Read with pandas, skipping comment lines
            read_params = {
                'filepath_or_buffer': file_path,
                'encoding': encoding,
                'delimiter': delimiter,
                'nrows': rows,
                'header': None,  # Read without header for analysis
                'on_bad_lines': 'skip',
                'engine': 'python'  # Python engine handles irregular data better
            }
            
            if comment_char:
                read_params['comment'] = comment_char
            
            return pd.read_csv(**read_params)
            
        except Exception as e:
            self.logger.error(f"Failed to read preview data: {e}")
            # Try one more time with very basic settings
            try:
                return pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    nrows=rows,
                    header=None,
                    on_bad_lines='skip',
                    engine='python'
                )
            except:
                return pd.DataFrame()
    
    def _analyze_structure(self, file_path: Path, encoding: str, delimiter: str, 
                          has_header: bool, preview_data: pd.DataFrame) -> CSVStructure:
        """Analyze CSV structure and identify column types."""
        try:
            # Get total row count
            with open(file_path, 'r', encoding=encoding) as f:
                row_count = sum(1 for _ in f)
            
            if has_header:
                row_count -= 1
            
            # Analyze columns
            columns = []
            issues = []
            metadata_rows = []
            
            for col_idx in preview_data.columns:
                column_data = preview_data[col_idx].dropna()
                
                if len(column_data) == 0:
                    continue
                
                # Get column name
                if has_header:
                    with open(file_path, 'r', encoding=encoding) as f:
                        header_line = f.readline().strip()
                    column_names = header_line.split(delimiter)
                    col_name = column_names[col_idx] if col_idx < len(column_names) else f"Column_{col_idx}"
                else:
                    col_name = f"Column_{col_idx}"
                
                # Analyze column
                col_info = self._analyze_column(col_idx, col_name, column_data)
                columns.append(col_info)
            
            # Determine format type
            format_type, confidence = self._determine_format_type(columns)
            
            # Identify metadata rows
            metadata_rows = self._identify_metadata_rows(preview_data)
            
            return CSVStructure(
                file_path=file_path,
                encoding=encoding,
                delimiter=delimiter,
                has_header=has_header,
                row_count=row_count,
                column_count=len(columns),
                columns=columns,
                format_type=format_type,
                confidence=confidence,
                issues=issues,
                metadata_rows=metadata_rows
            )
            
        except Exception as e:
            self.logger.error(f"Structure analysis failed: {e}")
            return self._create_error_structure(file_path, str(e))
    
    def _analyze_column(self, index: int, name: str, data: pd.Series) -> ColumnInfo:
        """Analyze a single column to determine its type and characteristics."""
        sample_values = data.head(10).tolist()
        
        # Check if column is numeric
        numeric_data = pd.to_numeric(data, errors='coerce')
        is_numeric = not numeric_data.isna().all()
        
        numeric_range = None
        if is_numeric:
            valid_numeric = numeric_data.dropna()
            if len(valid_numeric) > 0:
                numeric_range = (float(valid_numeric.min()), float(valid_numeric.max()))
        
        # Determine data type
        data_type, confidence = self._classify_column_type(name, data, numeric_range)
        
        return ColumnInfo(
            index=index,
            name=name,
            data_type=data_type,
            sample_values=sample_values,
            numeric_range=numeric_range,
            has_missing=data.isna().any(),
            confidence=confidence
        )
    
    def _classify_column_type(self, name: str, data: pd.Series, 
                             numeric_range: Optional[Tuple[float, float]]) -> Tuple[DataType, float]:
        """Classify column type based on name and data characteristics."""
        name_lower = name.lower()
        
        # Check wavenumber patterns
        for pattern in self.wavenumber_regex:
            if pattern.search(name_lower):
                if numeric_range and 400 <= numeric_range[0] <= 4000:
                    return DataType.WAVENUMBER, 0.9
                return DataType.WAVENUMBER, 0.7
        
        # Check absorbance patterns
        for pattern in self.absorbance_regex:
            if pattern.search(name_lower):
                if numeric_range and 0 <= numeric_range[0] <= 5:
                    return DataType.ABSORBANCE, 0.9
                return DataType.ABSORBANCE, 0.7
        
        # Check transmittance patterns
        for pattern in self.transmittance_regex:
            if pattern.search(name_lower):
                if numeric_range and 0 <= numeric_range[0] <= 100:
                    return DataType.TRANSMITTANCE, 0.9
                return DataType.TRANSMITTANCE, 0.7
        
        # Analyze numeric ranges for type inference
        if numeric_range:
            min_val, max_val = numeric_range
            
            # Wavenumber range (400-4000 cm⁻¹)
            if 400 <= min_val <= 4000 and 400 <= max_val <= 4000:
                return DataType.WAVENUMBER, 0.8
            
            # Absorbance range (0-5)
            if 0 <= min_val <= 5 and 0 <= max_val <= 5:
                return DataType.ABSORBANCE, 0.6
            
            # Transmittance range (0-100%)
            if 0 <= min_val <= 100 and 0 <= max_val <= 100:
                return DataType.TRANSMITTANCE, 0.6
        
        # Check for sample ID patterns
        if any(keyword in name_lower for keyword in ['sample', 'id', 'name']):
            return DataType.SAMPLE_ID, 0.8
        
        # Default classification
        if numeric_range:
            return DataType.INTENSITY, 0.3
        else:
            return DataType.METADATA, 0.5
    
    def _determine_format_type(self, columns: List[ColumnInfo]) -> Tuple[CSVFormat, float]:
        """Determine the overall CSV format type."""
        if not columns:
            return CSVFormat.UNKNOWN, 0.0
        
        # Count column types
        type_counts = {}
        for col in columns:
            type_counts[col.data_type] = type_counts.get(col.data_type, 0) + 1
        
        # Standard spectral format: wavenumber + absorbance/transmittance
        if (type_counts.get(DataType.WAVENUMBER, 0) >= 1 and 
            (type_counts.get(DataType.ABSORBANCE, 0) >= 1 or 
             type_counts.get(DataType.TRANSMITTANCE, 0) >= 1)):
            return CSVFormat.STANDARD_SPECTRAL, 0.9
        
        # Multi-column format: multiple intensity columns
        if type_counts.get(DataType.INTENSITY, 0) > 2:
            return CSVFormat.MULTI_COLUMN, 0.7
        
        # Check for transposed format (more rows than expected columns)
        if len(columns) > 10:
            return CSVFormat.TRANSPOSED, 0.6
        
        # Instrument-specific format
        if any(col.data_type != DataType.UNKNOWN for col in columns):
            return CSVFormat.INSTRUMENT_SPECIFIC, 0.5
        
        return CSVFormat.UNKNOWN, 0.0
    
    def _identify_metadata_rows(self, preview_data: pd.DataFrame) -> List[int]:
        """Identify rows that contain metadata rather than spectral data."""
        metadata_rows = []
        
        for idx, row in preview_data.iterrows():
            # Check if row has mostly non-numeric data
            numeric_count = 0
            total_count = 0
            
            for value in row:
                if pd.notna(value):
                    total_count += 1
                    try:
                        float(value)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
            
            # If less than 50% numeric, consider it metadata
            if total_count > 0 and numeric_count / total_count < 0.5:
                metadata_rows.append(idx)
        
        return metadata_rows
    
    def _read_full_data(self, file_path: Path, format_info: FormatInfo,
                       chunk_size: int = 10000) -> pd.DataFrame:
        """Read full CSV data with proper format handling and memory efficiency."""
        try:
            # Check file size to determine reading strategy
            file_size = file_path.stat().st_size
            use_chunked_reading = file_size > 50 * 1024 * 1024  # 50MB threshold
            
            # Prepare pandas read_csv parameters
            read_params = {
                'filepath_or_buffer': file_path,
                'encoding': format_info.encoding,
                'delimiter': format_info.delimiter,
                'header': 0 if format_info.has_header else None,
                'quotechar': format_info.quote_char,
                'skipinitialspace': True,
                'skip_blank_lines': True,
                'on_bad_lines': 'skip'  # Skip problematic lines instead of failing
            }
            
            # Handle European decimal format
            if format_info.decimal_separator == ',':
                read_params['decimal'] = ','
                # Only set thousands separator if it's not the delimiter
                if format_info.thousands_separator and format_info.thousands_separator != format_info.delimiter:
                    read_params['thousands'] = format_info.thousands_separator
            
            # Skip metadata rows (comment lines)
            if format_info.metadata_rows:
                read_params['skiprows'] = format_info.metadata_rows
            
            # Also skip lines starting with comment prefixes
            if format_info.comment_prefixes:
                read_params['comment'] = format_info.comment_prefixes[0] if format_info.comment_prefixes else None
            
            if use_chunked_reading:
                # Read in chunks for large files
                self.logger.info(f"Using chunked reading for large file ({file_size / 1024 / 1024:.1f} MB)")
                read_params['chunksize'] = chunk_size
                
                chunks = []
                for chunk in pd.read_csv(**read_params):
                    # Convert European numbers if needed
                    if format_info.decimal_separator == ',':
                        chunk = self._convert_european_numbers(chunk)
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                # Read entire file at once for smaller files
                df = pd.read_csv(**read_params)
                
                # Convert European numbers if needed
                if format_info.decimal_separator == ',':
                    df = self._convert_european_numbers(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read full data: {e}")
            raise
    
    def _convert_european_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert European number format to standard format."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert European decimal format
                try:
                    # Replace comma with dot for decimal separator
                    converted = df[col].astype(str).str.replace(',', '.', regex=False)
                    # Also handle potential thousands separator (space or dot)
                    converted = converted.str.replace(' ', '', regex=False)
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(converted, errors='coerce')
                    # If conversion was successful for most values, use it
                    if numeric_series.notna().sum() > len(df) * 0.5:  # Lower threshold to 50%
                        df[col] = numeric_series
                except Exception:
                    continue
        return df
    
    def _extract_spectral_columns(self, df: pd.DataFrame, structure: CSVStructure, format_info: FormatInfo) -> pd.DataFrame:
        """Extract and standardize spectral data columns."""
        spectral_df = pd.DataFrame()
        
        # Find wavenumber column
        wavenumber_col = None
        intensity_cols = []
        
        for col_info in structure.columns:
            if col_info.data_type == DataType.WAVENUMBER:
                wavenumber_col = col_info.index
            elif col_info.data_type in [DataType.ABSORBANCE, DataType.TRANSMITTANCE, DataType.INTENSITY]:
                intensity_cols.append(col_info.index)
        
        # Extract wavenumber data
        if wavenumber_col is not None:
            wavenumber_data = df.iloc[:, wavenumber_col]
            spectral_df['wavenumber'] = pd.to_numeric(wavenumber_data, errors='coerce')
        
        # Extract intensity data
        for i, col_idx in enumerate(intensity_cols):
            col_name = f'intensity_{i}' if i > 0 else 'intensity'
            intensity_data = df.iloc[:, col_idx]
            spectral_df[col_name] = pd.to_numeric(intensity_data, errors='coerce')
        
        return spectral_df
    
    def _clean_spectral_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate spectral data."""
        # Remove rows with missing wavenumber data
        if 'wavenumber' in df.columns:
            df = df.dropna(subset=['wavenumber'])
        
        # Remove duplicate wavenumber values
        if 'wavenumber' in df.columns:
            df = df.drop_duplicates(subset=['wavenumber'])
        
        # Sort by wavenumber (descending for IR spectroscopy)
        if 'wavenumber' in df.columns:
            df = df.sort_values('wavenumber', ascending=False)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _create_error_structure(self, file_path: Path, error_message: str) -> CSVStructure:
        """Create error structure for failed parsing."""
        return CSVStructure(
            file_path=file_path,
            encoding='utf-8',
            delimiter=',',
            has_header=False,
            row_count=0,
            column_count=0,
            columns=[],
            format_type=CSVFormat.UNKNOWN,
            confidence=0.0,
            issues=[f"Parsing failed: {error_message}"],
            metadata_rows=[]
        )
    
    def validate_spectral_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate spectral data quality and completeness.
        
        Args:
            df: DataFrame containing spectral data
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check for required columns
            if 'wavenumber' not in df.columns:
                validation_results['issues'].append("Missing wavenumber column")
                validation_results['is_valid'] = False
            
            intensity_cols = [col for col in df.columns if col.startswith('intensity')]
            if not intensity_cols:
                validation_results['issues'].append("No intensity columns found")
                validation_results['is_valid'] = False
            
            if not validation_results['is_valid']:
                return validation_results
            
            # Validate wavenumber range
            wavenumber_range = (df['wavenumber'].min(), df['wavenumber'].max())
            if wavenumber_range[0] < 400 or wavenumber_range[1] > 4000:
                validation_results['warnings'].append(
                    f"Wavenumber range {wavenumber_range} outside typical IR range (400-4000 cm⁻¹)"
                )
            
            # Check for missing data
            missing_data = df.isnull().sum()
            if missing_data.any():
                validation_results['warnings'].append(f"Missing data points: {missing_data.to_dict()}")
            
            # Calculate statistics
            validation_results['statistics'] = {
                'data_points': len(df),
                'wavenumber_range': wavenumber_range,
                'intensity_columns': len(intensity_cols),
                'missing_values': missing_data.sum(),
                'duplicate_wavenumbers': df['wavenumber'].duplicated().sum()
            }
            
            self.logger.info(f"Data validation completed: {validation_results['statistics']}")
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {e}")
            validation_results['is_valid'] = False
        
        return validation_results
    
    def is_standard_format(self, file_path: Union[str, Path]) -> bool:
        """
        Quick check if file is already in standard format.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            True if file is in standard spectral format
        """
        try:
            result = self.parse_file(file_path, preview_rows=50)
            return (result.success and
                    result.structure.format_type == CSVFormat.STANDARD_SPECTRAL and
                    result.structure.confidence > 0.8)
        except Exception:
            return False