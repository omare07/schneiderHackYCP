#!/usr/bin/env python3
"""
Test Data Validation Script

This script validates the quality and characteristics of the generated test data
to ensure it meets scientific and technical requirements for spectral analysis.
"""

import os
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class TestDataValidator:
    """Validate spectroscopic test data quality and characteristics."""
    
    def __init__(self, test_data_dir: str = "test_data"):
        """Initialize validator with test data directory."""
        self.test_data_dir = Path(test_data_dir)
        self.validation_results = {}
        
        # Scientific validation criteria
        self.wavenumber_ranges = {
            'ftir': (400, 4000),
            'raman': (200, 3500),
            'nir': (4000, 12000),
            'uv_vis': (200, 800)
        }
        
        self.absorbance_range = (0.0, 3.0)
        self.min_data_points = 100
        self.max_data_points = 20000
    
    def detect_file_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                return encoding
            except UnicodeDecodeError:
                continue
        
        return 'unknown'
    
    def analyze_csv_structure(self, file_path: Path) -> Dict:
        """Analyze CSV file structure and format."""
        encoding = self.detect_file_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Detect delimiter
            delimiters = [',', ';', '\t']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                count = sum(line.count(delimiter) for line in non_empty_lines[:10])
                delimiter_counts[delimiter] = count
            
            detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            # Detect decimal separator
            decimal_comma_count = sum(1 for line in non_empty_lines[:20] 
                                    if ',' in line and line.count(',') > line.count(detected_delimiter))
            uses_comma_decimal = decimal_comma_count > 5
            
            # Count comment lines
            comment_prefixes = ['#', '//', '*', '%', '!']
            comment_lines = 0
            for line in non_empty_lines:
                if any(line.strip().startswith(prefix) for prefix in comment_prefixes):
                    comment_lines += 1
            
            # Detect headers
            first_data_line = None
            for i, line in enumerate(non_empty_lines):
                if not any(line.strip().startswith(prefix) for prefix in comment_prefixes):
                    first_data_line = i
                    break
            
            has_headers = False
            if first_data_line is not None:
                header_line = non_empty_lines[first_data_line]
                # Check if first line contains text (headers) vs numbers
                parts = header_line.split(detected_delimiter)
                if len(parts) >= 2:
                    try:
                        float(parts[0].replace(',', '.'))
                        float(parts[1].replace(',', '.'))
                        has_headers = False
                    except ValueError:
                        has_headers = True
            
            return {
                'encoding': encoding,
                'delimiter': detected_delimiter,
                'decimal_separator': ',' if uses_comma_decimal else '.',
                'comment_lines': comment_lines,
                'has_headers': has_headers,
                'total_lines': len(non_empty_lines),
                'file_size': file_path.stat().st_size
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_spectral_data(self, file_path: Path) -> Dict:
        """Validate spectroscopic data quality."""
        try:
            # Try to read with pandas, handling various formats
            encoding = self.detect_file_encoding(file_path)
            
            # Skip comment lines and detect delimiter
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            
            # Remove comment lines
            data_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not any(stripped.startswith(p) for p in ['#', '//', '*', '%', '!']):
                    data_lines.append(line)
            
            if not data_lines:
                return {'error': 'No data lines found'}
            
            # Detect delimiter
            first_line = data_lines[0]
            if '\t' in first_line:
                delimiter = '\t'
            elif ';' in first_line:
                delimiter = ';'
            else:
                delimiter = ','
            
            # Create temporary file with clean data
            temp_content = '\n'.join(data_lines)
            
            # Handle European decimal format
            if delimiter == ';' and ',' in temp_content:
                temp_content = temp_content.replace(',', '.')
                delimiter = ';'
            
            # Read with pandas
            from io import StringIO
            df = pd.read_csv(StringIO(temp_content), delimiter=delimiter)
            
            # Get numeric columns (should be wavenumber and absorbance/intensity)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'error': 'Insufficient numeric columns'}
            
            # Assume first two numeric columns are wavenumber and intensity
            wavenumber_col = numeric_cols[0]
            intensity_col = numeric_cols[1]
            
            wavenumbers = df[wavenumber_col].dropna()
            intensities = df[intensity_col].dropna()
            
            if len(wavenumbers) != len(intensities):
                return {'error': 'Mismatched data lengths'}
            
            # Validate wavenumber range
            wn_min, wn_max = wavenumbers.min(), wavenumbers.max()
            wn_range = wn_max - wn_min
            
            # Determine spectroscopy type
            spectroscopy_type = 'unknown'
            if 300 <= wn_min <= 500 and 3500 <= wn_max <= 4500:
                spectroscopy_type = 'ftir'
            elif 100 <= wn_min <= 300 and 3000 <= wn_max <= 4000:
                spectroscopy_type = 'raman'
            elif 3500 <= wn_min <= 4500 and 10000 <= wn_max <= 15000:
                spectroscopy_type = 'nir'
            
            # Validate intensity range
            int_min, int_max = intensities.min(), intensities.max()
            
            # Check data order
            is_ascending = wavenumbers.is_monotonic_increasing
            is_descending = wavenumbers.is_monotonic_decreasing
            
            # Calculate data density
            data_density = len(wavenumbers) / wn_range if wn_range > 0 else 0
            
            # Check for realistic spectral features
            peak_count = 0
            if len(intensities) > 10:
                # Simple peak detection
                for i in range(1, len(intensities) - 1):
                    if (intensities.iloc[i] > intensities.iloc[i-1] and 
                        intensities.iloc[i] > intensities.iloc[i+1] and
                        intensities.iloc[i] > int_max * 0.1):
                        peak_count += 1
            
            # Calculate noise level (standard deviation of differences)
            if len(intensities) > 2:
                diffs = np.diff(intensities)
                noise_level = np.std(diffs)
            else:
                noise_level = 0
            
            return {
                'data_points': len(wavenumbers),
                'wavenumber_range': (float(wn_min), float(wn_max)),
                'wavenumber_span': float(wn_range),
                'intensity_range': (float(int_min), float(int_max)),
                'spectroscopy_type': spectroscopy_type,
                'is_ascending': bool(is_ascending),
                'is_descending': bool(is_descending),
                'data_density': float(data_density),
                'peak_count': peak_count,
                'noise_level': float(noise_level),
                'columns': list(df.columns),
                'numeric_columns': numeric_cols
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_file(self, file_path: Path) -> Dict:
        """Comprehensive validation of a single test file."""
        print(f"Validating {file_path.name}...")
        
        result = {
            'filename': file_path.name,
            'timestamp': datetime.now().isoformat(),
            'file_exists': file_path.exists(),
            'file_size': file_path.stat().st_size if file_path.exists() else 0
        }
        
        if not file_path.exists():
            result['status'] = 'FAIL'
            result['error'] = 'File does not exist'
            return result
        
        # Analyze structure
        structure = self.analyze_csv_structure(file_path)
        result['structure'] = structure
        
        # Validate spectral data
        spectral = self.validate_spectral_data(file_path)
        result['spectral'] = spectral
        
        # Overall validation
        issues = []
        warnings = []
        
        # Check for errors
        if 'error' in structure:
            issues.append(f"Structure error: {structure['error']}")
        if 'error' in spectral:
            issues.append(f"Spectral error: {spectral['error']}")
        
        # Validate data quality
        if 'error' not in spectral:
            data_points = spectral.get('data_points', 0)
            if data_points < self.min_data_points:
                issues.append(f"Too few data points: {data_points}")
            elif data_points > self.max_data_points:
                warnings.append(f"Very large dataset: {data_points} points")
            
            wn_range = spectral.get('wavenumber_range', (0, 0))
            if wn_range[1] - wn_range[0] < 100:
                warnings.append("Narrow wavenumber range")
            
            int_range = spectral.get('intensity_range', (0, 0))
            if int_range[1] > 10:
                warnings.append("Unusually high intensity values")
            
            if spectral.get('peak_count', 0) == 0:
                warnings.append("No spectral peaks detected")
            
            noise_level = spectral.get('noise_level', 0)
            if noise_level > 0.1:
                warnings.append(f"High noise level: {noise_level:.3f}")
        
        # Determine overall status
        if issues:
            result['status'] = 'FAIL'
            result['issues'] = issues
        elif warnings:
            result['status'] = 'WARN'
            result['warnings'] = warnings
        else:
            result['status'] = 'PASS'
        
        return result
    
    def validate_all_files(self) -> Dict:
        """Validate all test files in the directory."""
        print(f"Validating test data in {self.test_data_dir}")
        print("=" * 60)
        
        results = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_files': 0,
            'passed': 0,
            'warnings': 0,
            'failed': 0,
            'files': {}
        }
        
        # Get all CSV files
        csv_files = list(self.test_data_dir.glob("*.csv"))
        results['total_files'] = len(csv_files)
        
        for file_path in sorted(csv_files):
            file_result = self.validate_file(file_path)
            results['files'][file_path.name] = file_result
            
            # Update counters
            status = file_result.get('status', 'FAIL')
            if status == 'PASS':
                results['passed'] += 1
            elif status == 'WARN':
                results['warnings'] += 1
            else:
                results['failed'] += 1
        
        return results
    
    def print_validation_summary(self, results: Dict):
        """Print a formatted validation summary."""
        print("\nValidation Summary")
        print("=" * 60)
        print(f"Total files: {results['total_files']}")
        print(f"Passed: {results['passed']} ✅")
        print(f"Warnings: {results['warnings']} ⚠️")
        print(f"Failed: {results['failed']} ❌")
        print()
        
        # Print file details
        for filename, file_result in results['files'].items():
            status = file_result.get('status', 'FAIL')
            status_icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌'}[status]
            
            print(f"{status_icon} {filename}")
            
            if 'structure' in file_result and 'error' not in file_result['structure']:
                struct = file_result['structure']
                print(f"   📁 Size: {struct['file_size']:,} bytes, "
                      f"Lines: {struct['total_lines']}, "
                      f"Delimiter: '{struct['delimiter']}'")
            
            if 'spectral' in file_result and 'error' not in file_result['spectral']:
                spec = file_result['spectral']
                print(f"   📊 Points: {spec['data_points']:,}, "
                      f"Range: {spec['wavenumber_range'][0]:.0f}-{spec['wavenumber_range'][1]:.0f} cm⁻¹, "
                      f"Type: {spec['spectroscopy_type']}")
            
            if 'issues' in file_result:
                for issue in file_result['issues']:
                    print(f"   ❌ {issue}")
            
            if 'warnings' in file_result:
                for warning in file_result['warnings']:
                    print(f"   ⚠️  {warning}")
            
            print()
    
    def generate_validation_report(self, results: Dict, output_file: str = "validation_report.json"):
        """Generate detailed validation report."""
        report_path = self.test_data_dir / output_file
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Detailed validation report saved to: {report_path}")
    
    def check_format_diversity(self, results: Dict) -> Dict:
        """Check if test data covers diverse format issues."""
        format_coverage = {
            'delimiters': set(),
            'decimal_separators': set(),
            'encodings': set(),
            'has_comments': False,
            'has_no_headers': False,
            'has_extra_columns': False,
            'spectroscopy_types': set()
        }
        
        for filename, file_result in results['files'].items():
            if 'structure' in file_result and 'error' not in file_result['structure']:
                struct = file_result['structure']
                format_coverage['delimiters'].add(struct.get('delimiter', ','))
                format_coverage['decimal_separators'].add(struct.get('decimal_separator', '.'))
                format_coverage['encodings'].add(struct.get('encoding', 'utf-8'))
                
                if struct.get('comment_lines', 0) > 0:
                    format_coverage['has_comments'] = True
                
                if not struct.get('has_headers', True):
                    format_coverage['has_no_headers'] = True
            
            if 'spectral' in file_result and 'error' not in file_result['spectral']:
                spec = file_result['spectral']
                format_coverage['spectroscopy_types'].add(spec.get('spectroscopy_type', 'unknown'))
                
                # Check for extra columns
                columns = spec.get('columns', [])
                if len(columns) > 2:
                    format_coverage['has_extra_columns'] = True
        
        # Convert sets to lists for JSON serialization
        format_coverage['delimiters'] = list(format_coverage['delimiters'])
        format_coverage['decimal_separators'] = list(format_coverage['decimal_separators'])
        format_coverage['encodings'] = list(format_coverage['encodings'])
        format_coverage['spectroscopy_types'] = list(format_coverage['spectroscopy_types'])
        
        return format_coverage
    
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation suite."""
        print("🔬 Spectral Analysis Test Data Validation")
        print("=" * 60)
        
        # Validate all files
        results = self.validate_all_files()
        
        # Check format diversity
        format_coverage = self.check_format_diversity(results)
        results['format_coverage'] = format_coverage
        
        # Print summary
        self.print_validation_summary(results)
        
        # Print format coverage
        print("Format Coverage Analysis")
        print("=" * 60)
        print(f"Delimiters: {', '.join(format_coverage['delimiters'])}")
        print(f"Decimal separators: {', '.join(format_coverage['decimal_separators'])}")
        print(f"Encodings: {', '.join(format_coverage['encodings'])}")
        print(f"Spectroscopy types: {', '.join(format_coverage['spectroscopy_types'])}")
        print(f"Has comment lines: {'Yes' if format_coverage['has_comments'] else 'No'}")
        print(f"Has files without headers: {'Yes' if format_coverage['has_no_headers'] else 'No'}")
        print(f"Has extra columns: {'Yes' if format_coverage['has_extra_columns'] else 'No'}")
        print()
        
        # Generate report
        self.generate_validation_report(results)
        
        return results


def main():
    """Main validation function."""
    validator = TestDataValidator()
    results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if results['failed'] > 0:
        print("❌ Validation failed - some files have critical issues")
        exit(1)
    elif results['warnings'] > 0:
        print("⚠️  Validation completed with warnings")
        exit(0)
    else:
        print("✅ All files passed validation")
        exit(0)


if __name__ == "__main__":
    main()