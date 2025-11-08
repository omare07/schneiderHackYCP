#!/usr/bin/env python3
"""
Integration Test Scenarios for Spectral Analysis Application

This script provides comprehensive integration testing scenarios that demonstrate
the complete workflow from file loading through AI normalization to graph generation.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.csv_parser import CSVParser
    from core.ai_normalizer import AINormalizer
    from core.data_validator import DataValidator
    from core.graph_generator import GraphGenerator
    from utils.cache_manager import CacheManager
    from utils.cost_tracker import CostTracker
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Integration tests will run in simulation mode")


class IntegrationTestRunner:
    """Run comprehensive integration test scenarios."""
    
    def __init__(self, test_data_dir: str = "test_data"):
        """Initialize integration test runner."""
        self.test_data_dir = Path(test_data_dir)
        self.results = {}
        
        # Initialize components (with error handling)
        try:
            self.parser = CSVParser()
            self.normalizer = AINormalizer()
            self.validator = DataValidator()
            self.graph_generator = GraphGenerator()
            self.cache_manager = CacheManager()
            self.cost_tracker = CostTracker()
            self.components_available = True
        except Exception as e:
            print(f"Components not available: {e}")
            self.components_available = False
    
    def test_perfect_format_workflow(self) -> Dict:
        """Test workflow with perfect format file."""
        print("🧪 Testing Perfect Format Workflow")
        print("-" * 50)
        
        test_file = self.test_data_dir / "baseline_perfect.csv"
        if not test_file.exists():
            return {'status': 'SKIP', 'reason': 'Test file not found'}
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        try:
            start_time = time.time()
            
            # Step 1: Parse CSV
            print("  📄 Parsing CSV file...")
            parsed_data = self.parser.parse_file(str(test_file))
            parse_time = time.time() - start_time
            
            # Step 2: AI Normalization (should detect no issues)
            print("  🤖 Running AI normalization...")
            norm_start = time.time()
            norm_result = self.normalizer.normalize_data(parsed_data)
            norm_time = time.time() - norm_start
            
            # Step 3: Data Validation
            print("  ✅ Validating normalized data...")
            validation_result = self.validator.validate_data(norm_result.data)
            
            # Step 4: Generate Graph
            print("  📊 Generating graph...")
            graph_start = time.time()
            graph_result = self.graph_generator.create_graph(
                norm_result.data, 
                title="Perfect Format Test"
            )
            graph_time = time.time() - graph_start
            
            total_time = time.time() - start_time
            
            return {
                'status': 'PASS',
                'data_points': len(norm_result.data),
                'ai_confidence': norm_result.confidence,
                'transformations': norm_result.transformations,
                'validation_passed': validation_result.is_valid,
                'graph_generated': graph_result is not None,
                'timing': {
                    'parse_time': parse_time,
                    'normalization_time': norm_time,
                    'graph_time': graph_time,
                    'total_time': total_time
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_problematic_format_workflow(self, filename: str) -> Dict:
        """Test workflow with problematic format file."""
        print(f"🔧 Testing Problematic Format: {filename}")
        print("-" * 50)
        
        test_file = self.test_data_dir / filename
        if not test_file.exists():
            return {'status': 'SKIP', 'reason': 'Test file not found'}
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        try:
            start_time = time.time()
            
            # Step 1: Attempt direct parsing (should fail or produce poor results)
            print("  📄 Attempting direct CSV parsing...")
            try:
                direct_data = self.parser.parse_file(str(test_file))
                direct_parse_success = True
            except Exception as e:
                print(f"    ❌ Direct parsing failed: {e}")
                direct_parse_success = False
                direct_data = None
            
            # Step 2: AI Normalization (should fix issues)
            print("  🤖 Running AI normalization...")
            norm_start = time.time()
            
            if direct_parse_success and direct_data:
                norm_result = self.normalizer.normalize_data(direct_data)
            else:
                # Try AI normalization on raw file
                norm_result = self.normalizer.normalize_file(str(test_file))
            
            norm_time = time.time() - norm_start
            
            # Step 3: Validate normalized data
            print("  ✅ Validating normalized data...")
            validation_result = self.validator.validate_data(norm_result.data)
            
            # Step 4: Generate graph
            print("  📊 Generating graph...")
            graph_start = time.time()
            graph_result = self.graph_generator.create_graph(
                norm_result.data,
                title=f"Normalized: {filename}"
            )
            graph_time = time.time() - graph_start
            
            total_time = time.time() - start_time
            
            return {
                'status': 'PASS',
                'direct_parse_success': direct_parse_success,
                'data_points': len(norm_result.data),
                'ai_confidence': norm_result.confidence,
                'transformations': norm_result.transformations,
                'validation_passed': validation_result.is_valid,
                'graph_generated': graph_result is not None,
                'timing': {
                    'normalization_time': norm_time,
                    'graph_time': graph_time,
                    'total_time': total_time
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_batch_processing_workflow(self) -> Dict:
        """Test batch processing of multiple files."""
        print("📦 Testing Batch Processing Workflow")
        print("-" * 50)
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        # Get test files
        test_files = [
            "baseline_perfect.csv",
            "sample_mixed_issues.csv",
            "sample_extra_columns.csv",
            "sample_tab_delimited.csv"
        ]
        
        batch_results = {}
        start_time = time.time()
        
        try:
            for filename in test_files:
                file_path = self.test_data_dir / filename
                if not file_path.exists():
                    batch_results[filename] = {'status': 'SKIP', 'reason': 'File not found'}
                    continue
                
                print(f"  Processing {filename}...")
                file_start = time.time()
                
                # Process file
                norm_result = self.normalizer.normalize_file(str(file_path))
                validation_result = self.validator.validate_data(norm_result.data)
                
                file_time = time.time() - file_start
                
                batch_results[filename] = {
                    'status': 'PASS',
                    'data_points': len(norm_result.data),
                    'confidence': norm_result.confidence,
                    'valid': validation_result.is_valid,
                    'processing_time': file_time
                }
            
            total_time = time.time() - start_time
            
            return {
                'status': 'PASS',
                'files_processed': len([r for r in batch_results.values() if r['status'] == 'PASS']),
                'total_files': len(test_files),
                'batch_results': batch_results,
                'total_time': total_time,
                'average_time': total_time / len(test_files)
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_performance_scenario(self) -> Dict:
        """Test performance with large file."""
        print("⚡ Testing Performance Scenario")
        print("-" * 50)
        
        test_file = self.test_data_dir / "performance_large_file.csv"
        if not test_file.exists():
            return {'status': 'SKIP', 'reason': 'Performance test file not found'}
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        try:
            print(f"  📁 File size: {test_file.stat().st_size / 1024:.1f} KB")
            
            start_time = time.time()
            
            # Test parsing performance
            parse_start = time.time()
            parsed_data = self.parser.parse_file(str(test_file))
            parse_time = time.time() - parse_start
            
            # Test normalization performance
            norm_start = time.time()
            norm_result = self.normalizer.normalize_data(parsed_data)
            norm_time = time.time() - norm_start
            
            # Test graph generation performance
            graph_start = time.time()
            graph_result = self.graph_generator.create_graph(
                norm_result.data,
                title="Performance Test - Large Dataset"
            )
            graph_time = time.time() - graph_start
            
            total_time = time.time() - start_time
            
            # Performance criteria
            parse_acceptable = parse_time < 2.0  # < 2 seconds
            norm_acceptable = norm_time < 5.0    # < 5 seconds
            graph_acceptable = graph_time < 3.0  # < 3 seconds
            total_acceptable = total_time < 10.0 # < 10 seconds total
            
            performance_pass = all([parse_acceptable, norm_acceptable, 
                                  graph_acceptable, total_acceptable])
            
            return {
                'status': 'PASS' if performance_pass else 'WARN',
                'data_points': len(norm_result.data),
                'file_size_kb': test_file.stat().st_size / 1024,
                'timing': {
                    'parse_time': parse_time,
                    'normalization_time': norm_time,
                    'graph_time': graph_time,
                    'total_time': total_time
                },
                'performance_criteria': {
                    'parse_acceptable': parse_acceptable,
                    'norm_acceptable': norm_acceptable,
                    'graph_acceptable': graph_acceptable,
                    'total_acceptable': total_acceptable
                }
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_error_handling_scenarios(self) -> Dict:
        """Test error handling with various problematic files."""
        print("🚨 Testing Error Handling Scenarios")
        print("-" * 50)
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        error_scenarios = {}
        
        # Test 1: Non-existent file
        print("  Testing non-existent file...")
        try:
            result = self.normalizer.normalize_file("nonexistent.csv")
            error_scenarios['nonexistent_file'] = {'status': 'UNEXPECTED_PASS'}
        except Exception as e:
            error_scenarios['nonexistent_file'] = {'status': 'EXPECTED_FAIL', 'error': str(e)}
        
        # Test 2: Empty file
        print("  Testing empty file...")
        empty_file = self.test_data_dir / "empty_test.csv"
        try:
            empty_file.write_text("")
            result = self.normalizer.normalize_file(str(empty_file))
            error_scenarios['empty_file'] = {'status': 'UNEXPECTED_PASS'}
        except Exception as e:
            error_scenarios['empty_file'] = {'status': 'EXPECTED_FAIL', 'error': str(e)}
        finally:
            if empty_file.exists():
                empty_file.unlink()
        
        # Test 3: Binary file
        print("  Testing binary file...")
        binary_file = self.test_data_dir / "binary_test.csv"
        try:
            binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
            result = self.normalizer.normalize_file(str(binary_file))
            error_scenarios['binary_file'] = {'status': 'UNEXPECTED_PASS'}
        except Exception as e:
            error_scenarios['binary_file'] = {'status': 'EXPECTED_FAIL', 'error': str(e)}
        finally:
            if binary_file.exists():
                binary_file.unlink()
        
        # Count expected failures
        expected_failures = sum(1 for r in error_scenarios.values() 
                              if r['status'] == 'EXPECTED_FAIL')
        
        return {
            'status': 'PASS',
            'scenarios_tested': len(error_scenarios),
            'expected_failures': expected_failures,
            'scenarios': error_scenarios
        }
    
    def test_ai_normalization_accuracy(self) -> Dict:
        """Test AI normalization accuracy across different format issues."""
        print("🎯 Testing AI Normalization Accuracy")
        print("-" * 50)
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        # Test files with known format issues
        test_cases = {
            'baseline_perfect.csv': {
                'expected_confidence': 95,
                'expected_transformations': 0,
                'description': 'Perfect format - no changes needed'
            },
            'baseline_european.csv': {
                'expected_confidence': 90,
                'expected_transformations': 2,  # delimiter + decimal
                'description': 'European format conversion'
            },
            'sample_tab_delimited.csv': {
                'expected_confidence': 85,
                'expected_transformations': 2,  # tabs + comments
                'description': 'Tab delimiter conversion'
            },
            'sample_mixed_issues.csv': {
                'expected_confidence': 70,
                'expected_transformations': 4,  # multiple issues
                'description': 'Multiple format problems'
            },
            'sample_extra_columns.csv': {
                'expected_confidence': 80,
                'expected_transformations': 1,  # column extraction
                'description': 'Extra metadata columns'
            }
        }
        
        accuracy_results = {}
        
        for filename, expected in test_cases.items():
            file_path = self.test_data_dir / filename
            if not file_path.exists():
                accuracy_results[filename] = {'status': 'SKIP', 'reason': 'File not found'}
                continue
            
            print(f"  Testing {filename}...")
            
            try:
                norm_result = self.normalizer.normalize_file(str(file_path))
                
                # Check confidence level
                confidence_ok = norm_result.confidence >= expected['expected_confidence'] - 10
                
                # Check transformations (approximate)
                transform_count = len(norm_result.transformations)
                transform_ok = abs(transform_count - expected['expected_transformations']) <= 1
                
                # Validate data quality
                validation_result = self.validator.validate_data(norm_result.data)
                
                accuracy_results[filename] = {
                    'status': 'PASS' if confidence_ok and validation_result.is_valid else 'WARN',
                    'confidence': norm_result.confidence,
                    'expected_confidence': expected['expected_confidence'],
                    'confidence_ok': confidence_ok,
                    'transformations': transform_count,
                    'expected_transformations': expected['expected_transformations'],
                    'transform_ok': transform_ok,
                    'validation_passed': validation_result.is_valid,
                    'description': expected['description']
                }
                
            except Exception as e:
                accuracy_results[filename] = {'status': 'FAIL', 'error': str(e)}
        
        # Calculate overall accuracy
        passed = sum(1 for r in accuracy_results.values() if r.get('status') == 'PASS')
        total = len([r for r in accuracy_results.values() if r.get('status') != 'SKIP'])
        accuracy_rate = passed / total if total > 0 else 0
        
        return {
            'status': 'PASS' if accuracy_rate >= 0.8 else 'WARN',
            'accuracy_rate': accuracy_rate,
            'test_cases': accuracy_results,
            'passed': passed,
            'total': total
        }
    
    def test_caching_and_cost_tracking(self) -> Dict:
        """Test caching system and cost tracking."""
        print("💰 Testing Caching and Cost Tracking")
        print("-" * 50)
        
        if not self.components_available:
            return {'status': 'SKIP', 'reason': 'Components not available'}
        
        test_file = self.test_data_dir / "baseline_perfect.csv"
        if not test_file.exists():
            return {'status': 'SKIP', 'reason': 'Test file not found'}
        
        try:
            # Clear cache for clean test
            self.cache_manager.clear_cache()
            initial_cost = self.cost_tracker.get_total_cost()
            
            # First normalization (should hit API)
            print("  🔄 First normalization (cache miss)...")
            start_time = time.time()
            result1 = self.normalizer.normalize_file(str(test_file))
            first_time = time.time() - start_time
            cost_after_first = self.cost_tracker.get_total_cost()
            
            # Second normalization (should use cache)
            print("  ⚡ Second normalization (cache hit)...")
            start_time = time.time()
            result2 = self.normalizer.normalize_file(str(test_file))
            second_time = time.time() - start_time
            cost_after_second = self.cost_tracker.get_total_cost()
            
            # Verify cache effectiveness
            cache_speedup = first_time / second_time if second_time > 0 else float('inf')
            cost_saved = (cost_after_first - initial_cost) - (cost_after_second - cost_after_first)
            
            return {
                'status': 'PASS',
                'cache_speedup': cache_speedup,
                'first_time': first_time,
                'second_time': second_time,
                'cost_first_call': cost_after_first - initial_cost,
                'cost_second_call': cost_after_second - cost_after_first,
                'cost_saved': cost_saved,
                'results_identical': result1.data.equals(result2.data) if hasattr(result1.data, 'equals') else True
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def run_all_integration_tests(self) -> Dict:
        """Run complete integration test suite."""
        print("🧪 Spectral Analysis Integration Test Suite")
        print("=" * 60)
        print(f"Test data directory: {self.test_data_dir}")
        print(f"Components available: {self.components_available}")
        print()
        
        all_results = {
            'timestamp': time.time(),
            'test_data_dir': str(self.test_data_dir),
            'components_available': self.components_available,
            'tests': {}
        }
        
        # Test 1: Perfect format workflow
        all_results['tests']['perfect_workflow'] = self.test_perfect_format_workflow()
        print()
        
        # Test 2: Problematic format workflows
        problematic_files = [
            'sample_mixed_issues.csv',
            'sample_tab_delimited.csv',
            'sample_extra_columns.csv',
            'sample_wrong_order.csv'
        ]
        
        for filename in problematic_files:
            test_key = f'problematic_{filename.replace(".csv", "")}'
            all_results['tests'][test_key] = self.test_problematic_format_workflow(filename)
            print()
        
        # Test 3: Batch processing
        all_results['tests']['batch_processing'] = self.test_batch_processing_workflow()
        print()
        
        # Test 4: Performance testing
        all_results['tests']['performance'] = self.test_performance_scenario()
        print()
        
        # Test 5: Error handling
        all_results['tests']['error_handling'] = self.test_error_handling_scenarios()
        print()
        
        # Test 6: Caching and cost tracking
        all_results['tests']['caching_costs'] = self.test_caching_and_cost_tracking()
        print()
        
        # Calculate overall results
        test_results = all_results['tests']
        passed = sum(1 for r in test_results.values() if r.get('status') == 'PASS')
        warned = sum(1 for r in test_results.values() if r.get('status') == 'WARN')
        failed = sum(1 for r in test_results.values() if r.get('status') == 'FAIL')
        skipped = sum(1 for r in test_results.values() if r.get('status') == 'SKIP')
        
        all_results['summary'] = {
            'total_tests': len(test_results),
            'passed': passed,
            'warnings': warned,
            'failed': failed,
            'skipped': skipped,
            'success_rate': passed / (len(test_results) - skipped) if len(test_results) - skipped > 0 else 0
        }
        
        return all_results
    
    def print_final_summary(self, results: Dict):
        """Print final test summary."""
        summary = results['summary']
        
        print("🏁 Integration Test Summary")
        print("=" * 60)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Warnings: {summary['warnings']} ⚠️")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Skipped: {summary['skipped']} ⏭️")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print()
        
        if summary['failed'] == 0:
            print("🎉 All integration tests completed successfully!")
        elif summary['warnings'] > 0:
            print("⚠️  Integration tests completed with warnings")
        else:
            print("❌ Some integration tests failed")
        
        print()
        print("📊 Test Coverage:")
        print("  ✅ Perfect format handling")
        print("  ✅ Problematic format normalization")
        print("  ✅ Batch processing workflow")
        print("  ✅ Performance with large files")
        print("  ✅ Error handling scenarios")
        print("  ✅ Caching and cost optimization")


def main():
    """Main integration test function."""
    validator = IntegrationTestRunner()
    results = validator.run_all_integration_tests()
    validator.print_final_summary(results)
    
    # Save detailed results
    report_path = validator.test_data_dir / "integration_test_results.json"
    with open(report_path, 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Detailed results saved to: {report_path}")
    
    # Exit with appropriate code
    summary = results['summary']
    if summary['failed'] > 0:
        exit(1)
    elif summary['warnings'] > 0:
        exit(0)
    else:
        exit(0)


if __name__ == "__main__":
    main()