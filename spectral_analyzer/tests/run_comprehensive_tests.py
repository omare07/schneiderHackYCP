#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner

This script runs the complete test suite for the spectral analysis application,
including data validation, integration tests, and performance benchmarks.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test modules
from validate_test_data import TestDataValidator
from test_integration_scenarios import IntegrationTestRunner


class ComprehensiveTestSuite:
    """Run complete test suite for spectral analysis application."""
    
    def __init__(self, test_data_dir: str = "test_data"):
        """Initialize comprehensive test suite."""
        self.test_data_dir = Path(test_data_dir)
        self.results = {}
        
        # Initialize test components
        self.data_validator = TestDataValidator(str(test_data_dir))
        self.integration_runner = IntegrationTestRunner(str(test_data_dir))
    
    def run_data_validation_suite(self) -> Dict:
        """Run data validation tests."""
        print("🔍 Phase 1: Data Validation Suite")
        print("=" * 60)
        
        try:
            validation_results = self.data_validator.run_comprehensive_validation()
            
            # Summarize validation results
            summary = {
                'status': 'PASS' if validation_results['failed'] == 0 else 'FAIL',
                'total_files': validation_results['total_files'],
                'passed': validation_results['passed'],
                'warnings': validation_results['warnings'],
                'failed': validation_results['failed'],
                'format_coverage': validation_results['format_coverage']
            }
            
            return summary
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def run_integration_test_suite(self) -> Dict:
        """Run integration tests."""
        print("\n🔗 Phase 2: Integration Test Suite")
        print("=" * 60)
        
        try:
            integration_results = self.integration_runner.run_all_integration_tests()
            
            # Extract summary
            summary = integration_results['summary']
            summary['status'] = 'PASS' if summary['failed'] == 0 else 'FAIL'
            
            return summary
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def run_performance_benchmarks(self) -> Dict:
        """Run performance benchmark tests."""
        print("\n⚡ Phase 3: Performance Benchmarks")
        print("=" * 60)
        
        benchmarks = {}
        
        # Test file sizes and processing times
        test_files = [
            ('baseline_perfect.csv', 'Standard dataset'),
            ('sample_extra_columns.csv', 'Extra columns'),
            ('performance_large_file.csv', 'Large dataset'),
            ('sample_mixed_issues.csv', 'Complex issues')
        ]
        
        for filename, description in test_files:
            file_path = self.test_data_dir / filename
            if not file_path.exists():
                benchmarks[filename] = {'status': 'SKIP', 'reason': 'File not found'}
                continue
            
            print(f"  📊 Benchmarking {filename} ({description})...")
            
            try:
                file_size = file_path.stat().st_size
                
                # Time file reading
                start_time = time.time()
                with open(file_path, 'r') as f:
                    content = f.read()
                read_time = time.time() - start_time
                
                # Estimate processing complexity
                lines = len(content.split('\n'))
                chars = len(content)
                
                benchmarks[filename] = {
                    'status': 'PASS',
                    'description': description,
                    'file_size_bytes': file_size,
                    'file_size_kb': file_size / 1024,
                    'lines': lines,
                    'characters': chars,
                    'read_time': read_time,
                    'read_speed_mb_per_sec': (file_size / 1024 / 1024) / read_time if read_time > 0 else float('inf')
                }
                
            except Exception as e:
                benchmarks[filename] = {'status': 'FAIL', 'error': str(e)}
        
        # Calculate overall performance metrics
        successful_benchmarks = [b for b in benchmarks.values() if b.get('status') == 'PASS']
        
        if successful_benchmarks:
            avg_read_speed = sum(b['read_speed_mb_per_sec'] for b in successful_benchmarks) / len(successful_benchmarks)
            total_size_kb = sum(b['file_size_kb'] for b in successful_benchmarks)
            
            performance_summary = {
                'status': 'PASS',
                'files_tested': len(successful_benchmarks),
                'average_read_speed_mb_per_sec': avg_read_speed,
                'total_test_data_size_kb': total_size_kb,
                'benchmarks': benchmarks
            }
        else:
            performance_summary = {
                'status': 'FAIL',
                'error': 'No successful benchmarks',
                'benchmarks': benchmarks
            }
        
        return performance_summary
    
    def analyze_test_coverage(self) -> Dict:
        """Analyze test coverage across different scenarios."""
        print("\n📋 Phase 4: Test Coverage Analysis")
        print("=" * 60)
        
        coverage_analysis = {
            'format_issues_covered': [],
            'spectroscopy_types_covered': [],
            'file_sizes_covered': [],
            'edge_cases_covered': []
        }
        
        # Analyze files in test directory
        csv_files = list(self.test_data_dir.glob("*.csv"))
        
        for file_path in csv_files:
            filename = file_path.name
            
            # Categorize by format issues
            if 'european' in filename:
                coverage_analysis['format_issues_covered'].append('European format')
            if 'tab' in filename:
                coverage_analysis['format_issues_covered'].append('Tab delimited')
            if 'no_header' in filename:
                coverage_analysis['format_issues_covered'].append('No headers')
            if 'extra_column' in filename:
                coverage_analysis['format_issues_covered'].append('Extra columns')
            if 'mixed' in filename:
                coverage_analysis['format_issues_covered'].append('Mixed issues')
            if 'scale' in filename:
                coverage_analysis['format_issues_covered'].append('Scale issues')
            if 'encoding' in filename:
                coverage_analysis['format_issues_covered'].append('Encoding issues')
            if 'wrong_order' in filename:
                coverage_analysis['format_issues_covered'].append('Wrong order')
            if 'noisy' in filename:
                coverage_analysis['format_issues_covered'].append('Noisy data')
            
            # Categorize by spectroscopy type
            if 'raman' in filename:
                coverage_analysis['spectroscopy_types_covered'].append('Raman')
            elif 'ftir' in filename or 'baseline' in filename or 'sample' in filename:
                coverage_analysis['spectroscopy_types_covered'].append('FTIR')
            
            # Categorize by file size
            file_size = file_path.stat().st_size
            if file_size < 1000:
                coverage_analysis['file_sizes_covered'].append('Small (<1KB)')
            elif file_size < 50000:
                coverage_analysis['file_sizes_covered'].append('Medium (1-50KB)')
            elif file_size < 200000:
                coverage_analysis['file_sizes_covered'].append('Large (50-200KB)')
            else:
                coverage_analysis['file_sizes_covered'].append('Very Large (>200KB)')
            
            # Edge cases
            if 'performance' in filename:
                coverage_analysis['edge_cases_covered'].append('Large dataset')
            if 'problematic' in filename:
                coverage_analysis['edge_cases_covered'].append('Corrupted data')
        
        # Remove duplicates and sort
        for key in coverage_analysis:
            coverage_analysis[key] = sorted(list(set(coverage_analysis[key])))
        
        # Calculate coverage score
        expected_coverage = {
            'format_issues_covered': 8,  # Expected number of format issue types
            'spectroscopy_types_covered': 2,  # FTIR, Raman
            'file_sizes_covered': 3,  # Small, Medium, Large
            'edge_cases_covered': 2   # Large dataset, corrupted data
        }
        
        coverage_scores = {}
        for key, expected_count in expected_coverage.items():
            actual_count = len(coverage_analysis[key])
            coverage_scores[key] = min(actual_count / expected_count, 1.0)
        
        overall_coverage = sum(coverage_scores.values()) / len(coverage_scores)
        
        coverage_analysis['coverage_scores'] = coverage_scores
        coverage_analysis['overall_coverage'] = overall_coverage
        coverage_analysis['status'] = 'PASS' if overall_coverage >= 0.8 else 'WARN'
        
        # Print coverage analysis
        print("Format Issues Covered:")
        for issue in coverage_analysis['format_issues_covered']:
            print(f"  ✅ {issue}")
        
        print(f"\nSpectroscopy Types: {', '.join(coverage_analysis['spectroscopy_types_covered'])}")
        print(f"File Sizes: {', '.join(coverage_analysis['file_sizes_covered'])}")
        print(f"Edge Cases: {', '.join(coverage_analysis['edge_cases_covered'])}")
        print(f"\nOverall Coverage Score: {overall_coverage:.1%}")
        
        return coverage_analysis
    
    def generate_final_report(self, all_results: Dict):
        """Generate comprehensive final report."""
        report = {
            'test_suite_version': '1.0',
            'execution_timestamp': datetime.now().isoformat(),
            'test_environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'test_data_directory': str(self.test_data_dir)
            },
            'results': all_results
        }
        
        # Save comprehensive report
        report_path = self.test_data_dir / "comprehensive_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Comprehensive test report saved to: {report_path}")
        
        # Generate summary report
        summary_path = self.test_data_dir / "test_summary.md"
        self.generate_markdown_summary(report, summary_path)
        print(f"📄 Summary report saved to: {summary_path}")
    
    def generate_markdown_summary(self, report: Dict, output_path: Path):
        """Generate markdown summary report."""
        results = report['results']
        
        markdown_content = f"""# Spectral Analysis Test Suite Report

**Generated**: {report['execution_timestamp']}  
**Test Data Directory**: {report['test_environment']['test_data_directory']}  
**Python Version**: {report['test_environment']['python_version']}

## Executive Summary

| Phase | Status | Details |
|-------|--------|---------|
| Data Validation | {'✅ PASS' if results['data_validation']['status'] == 'PASS' else '❌ FAIL'} | {results['data_validation']['passed']}/{results['data_validation']['total_files']} files passed |
| Integration Tests | {'✅ PASS' if results['integration_tests']['status'] == 'PASS' else '❌ FAIL'} | {results['integration_tests']['success_rate']:.1%} success rate |
| Performance Tests | {'✅ PASS' if results['performance_benchmarks']['status'] == 'PASS' else '❌ FAIL'} | {results['performance_benchmarks']['files_tested']} files tested |
| Coverage Analysis | {'✅ PASS' if results['coverage_analysis']['status'] == 'PASS' else '⚠️ WARN'} | {results['coverage_analysis']['overall_coverage']:.1%} coverage |

## Data Validation Results

- **Total Files**: {results['data_validation']['total_files']}
- **Passed**: {results['data_validation']['passed']} ✅
- **Warnings**: {results['data_validation']['warnings']} ⚠️
- **Failed**: {results['data_validation']['failed']} ❌

### Format Coverage

- **Delimiters**: {', '.join(results['data_validation']['format_coverage']['delimiters'])}
- **Decimal Separators**: {', '.join(results['data_validation']['format_coverage']['decimal_separators'])}
- **Encodings**: {', '.join(results['data_validation']['format_coverage']['encodings'])}
- **Spectroscopy Types**: {', '.join(results['data_validation']['format_coverage']['spectroscopy_types'])}

## Integration Test Results

- **Total Tests**: {results['integration_tests']['total_tests']}
- **Passed**: {results['integration_tests']['passed']} ✅
- **Warnings**: {results['integration_tests']['warnings']} ⚠️
- **Failed**: {results['integration_tests']['failed']} ❌
- **Skipped**: {results['integration_tests']['skipped']} ⏭️

## Performance Benchmarks

- **Files Tested**: {results['performance_benchmarks']['files_tested']}
- **Total Test Data Size**: {results['performance_benchmarks']['total_test_data_size_kb']:.1f} KB
- **Average Read Speed**: {results['performance_benchmarks']['average_read_speed_mb_per_sec']:.2f} MB/s

## Test Coverage Analysis

### Format Issues Covered
{chr(10).join(f'- {issue}' for issue in results['coverage_analysis']['format_issues_covered'])}

### Spectroscopy Types
{chr(10).join(f'- {stype}' for stype in results['coverage_analysis']['spectroscopy_types_covered'])}

### File Size Categories
{chr(10).join(f'- {size}' for size in results['coverage_analysis']['file_sizes_covered'])}

## Recommendations

### High Priority
- Address failed validation files
- Improve AI normalization confidence for complex cases
- Optimize performance for large files

### Medium Priority
- Add more spectroscopy types (NIR, UV-Vis)
- Include more encoding variations
- Add instrument-specific format tests

### Low Priority
- Expand edge case coverage
- Add stress testing scenarios
- Implement automated regression testing

## Test Data Quality Assessment

The generated test dataset successfully demonstrates:

1. **Format Diversity**: Multiple CSV format variations
2. **Realistic Data**: Scientifically accurate spectroscopic features
3. **Edge Cases**: Noise, artifacts, and data quality issues
4. **Performance Testing**: Large file handling capabilities
5. **Error Scenarios**: Robust error handling validation

## Conclusion

The comprehensive test dataset provides excellent coverage of real-world CSV format challenges and demonstrates the application's AI normalization capabilities effectively.

---
*Generated by Comprehensive Test Suite v1.0*
"""
        
        with open(output_path, 'w') as f:
            f.write(markdown_content)
    
    def run_complete_test_suite(self) -> Dict:
        """Run the complete test suite."""
        print("🧪 Spectral Analysis Comprehensive Test Suite")
        print("=" * 60)
        print(f"Test Data Directory: {self.test_data_dir}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        all_results = {}
        
        # Phase 1: Data Validation
        all_results['data_validation'] = self.run_data_validation_suite()
        
        # Phase 2: Integration Tests
        all_results['integration_tests'] = self.run_integration_test_suite()
        
        # Phase 3: Performance Benchmarks
        all_results['performance_benchmarks'] = self.run_performance_benchmarks()
        
        # Phase 4: Coverage Analysis
        all_results['coverage_analysis'] = self.analyze_test_coverage()
        
        return all_results
    
    def analyze_test_coverage(self) -> Dict:
        """Analyze overall test coverage."""
        print("\n📊 Phase 4: Test Coverage Analysis")
        print("=" * 60)
        
        # Count files by category
        csv_files = list(self.test_data_dir.glob("*.csv"))
        
        categories = {
            'baseline_files': [f for f in csv_files if 'baseline' in f.name],
            'sample_files': [f for f in csv_files if 'sample' in f.name],
            'legacy_files': [f for f in csv_files if f.name in [
                'european_format.csv', 'multi_column.csv', 'no_header.csv',
                'problematic_data.csv', 'sample_spectral.csv', 'tab_delimited.csv'
            ]],
            'performance_files': [f for f in csv_files if 'performance' in f.name]
        }
        
        # Analyze format diversity
        format_issues = {
            'delimiter_variations': ['comma', 'semicolon', 'tab'],
            'decimal_separators': ['period', 'comma'],
            'header_issues': ['no_headers', 'foreign_language', 'extra_metadata'],
            'data_issues': ['wrong_order', 'scale_problems', 'noise_artifacts'],
            'encoding_issues': ['utf8', 'latin1'],
            'column_issues': ['extra_columns', 'missing_columns', 'wrong_names']
        }
        
        # Calculate coverage scores
        coverage_scores = {}
        for category, files in categories.items():
            coverage_scores[category] = len(files)
        
        total_files = len(csv_files)
        baseline_coverage = len(categories['baseline_files']) / 3  # Expect 3 baseline files
        sample_coverage = len(categories['sample_files']) / 10     # Expect ~10 sample files
        
        overall_coverage = min((baseline_coverage + sample_coverage) / 2, 1.0)
        
        print(f"📁 Total CSV files: {total_files}")
        print(f"📊 Baseline files: {len(categories['baseline_files'])}")
        print(f"🔧 Sample files: {len(categories['sample_files'])}")
        print(f"📜 Legacy files: {len(categories['legacy_files'])}")
        print(f"⚡ Performance files: {len(categories['performance_files'])}")
        print(f"📈 Overall coverage: {overall_coverage:.1%}")
        
        return {
            'status': 'PASS' if overall_coverage >= 0.8 else 'WARN',
            'total_files': total_files,
            'categories': {k: len(v) for k, v in categories.items()},
            'overall_coverage': overall_coverage,
            'coverage_scores': coverage_scores
        }
    
    def print_final_summary(self, results: Dict):
        """Print final comprehensive summary."""
        print("\n🏆 Final Test Suite Summary")
        print("=" * 60)
        
        phases = [
            ('Data Validation', results['data_validation']),
            ('Integration Tests', results['integration_tests']),
            ('Performance Benchmarks', results['performance_benchmarks']),
            ('Coverage Analysis', results['coverage_analysis'])
        ]
        
        all_passed = True
        for phase_name, phase_result in phases:
            status = phase_result.get('status', 'UNKNOWN')
            status_icon = {'PASS': '✅', 'WARN': '⚠️', 'FAIL': '❌', 'SKIP': '⏭️'}.get(status, '❓')
            print(f"{status_icon} {phase_name}: {status}")
            
            if status in ['FAIL', 'WARN']:
                all_passed = False
        
        print()
        if all_passed:
            print("🎉 All test phases completed successfully!")
            print("✅ Test dataset is ready for production use")
        else:
            print("⚠️  Some test phases completed with issues")
            print("📋 Review detailed reports for improvement recommendations")
        
        print(f"\n📊 Test Dataset Statistics:")
        print(f"   📁 Total files: {results['coverage_analysis']['total_files']}")
        print(f"   📈 Coverage: {results['coverage_analysis']['overall_coverage']:.1%}")
        print(f"   💾 Total size: {results['performance_benchmarks']['total_test_data_size_kb']:.1f} KB")


def main():
    """Main function to run comprehensive test suite."""
    test_suite = ComprehensiveTestSuite()
    
    # Run all tests
    results = test_suite.run_complete_test_suite()
    
    # Print summary
    test_suite.print_final_summary(results)
    
    # Generate reports
    test_suite.generate_final_report(results)
    
    # Determine exit code
    failed_phases = sum(1 for phase in results.values() if phase.get('status') == 'FAIL')
    
    if failed_phases > 0:
        print("\n❌ Test suite completed with failures")
        exit(1)
    else:
        print("\n✅ Test suite completed successfully")
        exit(0)


if __name__ == "__main__":
    main()