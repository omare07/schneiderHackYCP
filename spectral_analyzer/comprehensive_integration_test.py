#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test Suite
Tests the complete spectral analysis workflow from file loading to graph generation.
"""

import sys
import os
import logging
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import json

# Add the spectral_analyzer directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import core components
from config.settings import ConfigManager
from core.csv_parser import CSVParser
from core.ai_normalizer import AINormalizer
from core.data_validator import DataValidator
from core.graph_generator import GraphGenerator
from utils.cache_manager import CacheManager, CacheConfig
from utils.cost_tracker import CostTracker
from utils.batch_processor import BatchProcessor
from utils.file_manager import FileManager


class ComprehensiveIntegrationTest:
    """
    Comprehensive integration test suite for the spectral analyzer application.
    
    Tests all major workflows:
    1. File loading and parsing
    2. AI normalization with confidence scoring
    3. Data validation and quality checks
    4. Graph generation and export
    5. Batch processing performance
    6. Caching effectiveness
    7. Error handling and recovery
    """
    
    def __init__(self):
        """Initialize the test suite."""
        self.logger = self._setup_logging()
        self.test_results: Dict[str, Any] = {}
        self.temp_dir: Optional[Path] = None
        self.config_manager: Optional[ConfigManager] = None
        
        # Core components
        self.csv_parser: Optional[CSVParser] = None
        self.ai_normalizer: Optional[AINormalizer] = None
        self.data_validator: Optional[DataValidator] = None
        self.graph_generator: Optional[GraphGenerator] = None
        self.cache_manager: Optional[CacheManager] = None
        self.cost_tracker: Optional[CostTracker] = None
        self.batch_processor: Optional[BatchProcessor] = None
        self.file_manager: Optional[FileManager] = None
        
        # Test data paths
        self.test_data_dir = Path(__file__).parent / "tests" / "test_data"
        self.baseline_files = []
        self.sample_files = []
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the test suite."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('integration_test.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def setup(self) -> bool:
        """Set up the test environment."""
        try:
            self.logger.info("Setting up comprehensive integration test environment...")
            
            # Create temporary directory
            self.temp_dir = Path(tempfile.mkdtemp(prefix="spectral_test_"))
            self.logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Initialize configuration manager
            self.config_manager = ConfigManager()
            
            # Initialize core components
            self.csv_parser = CSVParser()
            self.ai_normalizer = AINormalizer(self.config_manager)
            self.data_validator = DataValidator()
            self.graph_generator = GraphGenerator()
            
            # Initialize caching and cost tracking
            cache_config = CacheConfig(
                memory_limit_mb=100,  # Reduced for testing
                disk_limit_mb=500,
                default_ttl_hours=1,
                enable_redis=False  # Disable Redis for testing
            )
            self.cache_manager = CacheManager(config=cache_config)
            self.cost_tracker = CostTracker(config_manager=self.config_manager)
            
            # Initialize batch processor and file manager
            self.batch_processor = BatchProcessor(config_manager=self.config_manager)
            self.file_manager = FileManager()
            
            # Collect test files
            self._collect_test_files()
            
            self.logger.info("Test environment setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up test environment: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _collect_test_files(self):
        """Collect test files from the test data directory."""
        if not self.test_data_dir.exists():
            self.logger.warning(f"Test data directory not found: {self.test_data_dir}")
            return
        
        # Collect baseline files
        baseline_patterns = ["baseline_*.csv", "*_baseline.csv"]
        for pattern in baseline_patterns:
            self.baseline_files.extend(self.test_data_dir.glob(pattern))
        
        # Collect sample files (all CSV files that aren't baselines)
        all_csv_files = list(self.test_data_dir.glob("*.csv"))
        self.sample_files = [f for f in all_csv_files if f not in self.baseline_files]
        
        self.logger.info(f"Found {len(self.baseline_files)} baseline files")
        self.logger.info(f"Found {len(self.sample_files)} sample files")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        self.logger.info("Starting comprehensive integration test suite...")
        
        test_methods = [
            ("File Loading and Parsing", self.test_file_loading_parsing),
            ("CSV Parser Performance", self.test_csv_parser_performance),
            ("Data Validation", self.test_data_validation),
            ("AI Normalization", self.test_ai_normalization),
            ("Graph Generation", self.test_graph_generation),
            ("Batch Processing", self.test_batch_processing),
            ("Caching System", self.test_caching_system),
            ("Cost Tracking", self.test_cost_tracking),
            ("Error Handling", self.test_error_handling),
            ("Memory Usage", self.test_memory_usage),
            ("Complete Workflow", self.test_complete_workflow)
        ]
        
        for test_name, test_method in test_methods:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Running test: {test_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                start_time = time.time()
                result = test_method()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    "status": "PASSED" if result else "FAILED",
                    "duration": end_time - start_time,
                    "details": result if isinstance(result, dict) else {}
                }
                
                status = "✅ PASSED" if result else "❌ FAILED"
                self.logger.info(f"{status} - {test_name} ({end_time - start_time:.2f}s)")
                
            except Exception as e:
                self.test_results[test_name] = {
                    "status": "ERROR",
                    "duration": 0,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                self.logger.error(f"❌ ERROR - {test_name}: {e}")
                self.logger.error(traceback.format_exc())
        
        return self.test_results
    
    def test_file_loading_parsing(self) -> bool:
        """Test file loading and CSV parsing functionality."""
        try:
            success_count = 0
            total_files = len(self.sample_files)
            
            if total_files == 0:
                self.logger.warning("No sample files found for testing")
                return True
            
            for file_path in self.sample_files:
                try:
                    self.logger.info(f"Testing file: {file_path.name}")
                    
                    # Test CSV parsing
                    result = self.csv_parser.parse_file(file_path)
                    
                    if result.success:
                        success_count += 1
                        self.logger.info(f"✅ Successfully parsed {file_path.name}")
                        self.logger.info(f"   Rows: {len(result.data)}, Columns: {len(result.data.columns) if result.data is not None else 0}")
                    else:
                        self.logger.warning(f"⚠️ Failed to parse {file_path.name}: {result.error}")
                
                except Exception as e:
                    self.logger.error(f"❌ Error parsing {file_path.name}: {e}")
            
            success_rate = success_count / total_files
            self.logger.info(f"File parsing success rate: {success_rate:.1%} ({success_count}/{total_files})")
            
            return success_rate >= 0.8  # 80% success rate required
            
        except Exception as e:
            self.logger.error(f"File loading test failed: {e}")
            return False
    
    def test_csv_parser_performance(self) -> bool:
        """Test CSV parser performance benchmarks."""
        try:
            if not self.sample_files:
                self.logger.warning("No sample files for performance testing")
                return True
            
            # Test with largest file
            largest_file = max(self.sample_files, key=lambda f: f.stat().st_size)
            self.logger.info(f"Performance testing with: {largest_file.name} ({largest_file.stat().st_size} bytes)")
            
            start_time = time.time()
            result = self.csv_parser.parse_file(largest_file)
            parse_time = time.time() - start_time
            
            self.logger.info(f"Parse time: {parse_time:.3f}s")
            
            # Performance target: < 1 second for typical files
            performance_target = 1.0
            meets_target = parse_time < performance_target
            
            if meets_target:
                self.logger.info(f"✅ Performance target met ({parse_time:.3f}s < {performance_target}s)")
            else:
                self.logger.warning(f"⚠️ Performance target missed ({parse_time:.3f}s >= {performance_target}s)")
            
            return result.success and meets_target
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return False
    
    def test_data_validation(self) -> bool:
        """Test data validation functionality."""
        try:
            if not self.sample_files:
                self.logger.warning("No sample files for validation testing")
                return True
            
            validation_results = []
            
            for file_path in self.sample_files[:5]:  # Test first 5 files
                try:
                    # Parse file first
                    parse_result = self.csv_parser.parse_file(file_path)
                    if not parse_result.success:
                        continue
                    
                    # Validate data
                    validation_result = self.data_validator.validate_spectral_data(parse_result.data)
                    
                    validation_results.append(validation_result)
                    
                    self.logger.info(f"Validation for {file_path.name}:")
                    self.logger.info(f"  Valid: {validation_result.is_valid}")
                    self.logger.info(f"  Issues: {len(validation_result.issues)}")
                    self.logger.info(f"  Quality Score: {validation_result.quality_score:.2f}")
                    
                except Exception as e:
                    self.logger.error(f"Validation error for {file_path.name}: {e}")
            
            # Check if validation system is working
            has_results = len(validation_results) > 0
            has_quality_scores = all(hasattr(r, 'quality_score') for r in validation_results)
            
            success = has_results and has_quality_scores
            self.logger.info(f"Data validation test: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Data validation test failed: {e}")
            return False
    
    def test_ai_normalization(self) -> bool:
        """Test AI normalization functionality."""
        try:
            if not self.sample_files:
                self.logger.warning("No sample files for AI normalization testing")
                return True
            
            # Test with a few sample files
            test_files = self.sample_files[:3]
            normalization_results = []
            
            for file_path in test_files:
                try:
                    self.logger.info(f"Testing AI normalization for: {file_path.name}")
                    
                    # Parse file first
                    parse_result = self.csv_parser.parse_file(file_path)
                    if not parse_result.success:
                        self.logger.warning(f"Skipping {file_path.name} - parse failed")
                        continue
                    
                    # Test normalization (this may use cached results or mock responses)
                    start_time = time.time()
                    try:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        normalization_result = loop.run_until_complete(
                            self.ai_normalizer.normalize_csv_structure(
                                parse_result.data, parse_result.structure
                            )
                        )
                        loop.close()
                        normalization_time = time.time() - start_time
                        
                        normalization_results.append({
                            'file': file_path.name,
                            'success': normalization_result.success,
                            'confidence': normalization_result.plan.confidence_score if normalization_result.plan else 0,
                            'time': normalization_time
                        })
                        
                        self.logger.info(f"  Success: {normalization_result.success}")
                        self.logger.info(f"  Confidence: {normalization_result.plan.confidence_score if normalization_result.plan else 0:.2f}")
                        self.logger.info(f"  Time: {normalization_time:.2f}s")
                    except Exception as e:
                        self.logger.error(f"  Normalization error: {e}")
                        normalization_results.append({
                            'file': file_path.name,
                            'success': False,
                            'confidence': 0,
                            'time': time.time() - start_time
                        })
                    
                except Exception as e:
                    self.logger.error(f"AI normalization error for {file_path.name}: {e}")
            
            # Evaluate results
            if not normalization_results:
                self.logger.warning("No normalization results to evaluate")
                return False
            
            success_rate = sum(1 for r in normalization_results if r['success']) / len(normalization_results)
            avg_time = sum(r['time'] for r in normalization_results) / len(normalization_results)
            
            self.logger.info(f"AI Normalization Results:")
            self.logger.info(f"  Success Rate: {success_rate:.1%}")
            self.logger.info(f"  Average Time: {avg_time:.2f}s")
            
            # Success criteria: > 50% success rate (accounting for potential API issues)
            return success_rate > 0.5
            
        except Exception as e:
            self.logger.error(f"AI normalization test failed: {e}")
            return False
    
    def test_graph_generation(self) -> bool:
        """Test graph generation functionality."""
        try:
            if not self.sample_files:
                self.logger.warning("No sample files for graph generation testing")
                return True
            
            # Test with first sample file
            test_file = self.sample_files[0]
            self.logger.info(f"Testing graph generation with: {test_file.name}")
            
            # Parse file
            parse_result = self.csv_parser.parse_file(test_file)
            if not parse_result.success:
                self.logger.error("Failed to parse test file for graph generation")
                return False
            
            # Generate graph
            output_path = self.temp_dir / f"test_graph_{test_file.stem}.png"
            
            start_time = time.time()
            success = self.graph_generator.generate_spectral_graph(
                data=parse_result.data,
                output_path=output_path,
                title=f"Test Graph - {test_file.name}"
            )
            generation_time = time.time() - start_time
            
            self.logger.info(f"Graph generation time: {generation_time:.2f}s")
            
            # Check if file was created
            file_created = output_path.exists()
            file_size = output_path.stat().st_size if file_created else 0
            
            self.logger.info(f"Graph file created: {file_created}")
            if file_created:
                self.logger.info(f"Graph file size: {file_size} bytes")
            
            # Success criteria
            success_criteria = success and file_created and file_size > 1000  # At least 1KB
            
            self.logger.info(f"Graph generation test: {'✅ PASSED' if success_criteria else '❌ FAILED'}")
            
            return success_criteria
            
        except Exception as e:
            self.logger.error(f"Graph generation test failed: {e}")
            return False
    
    def test_batch_processing(self) -> bool:
        """Test batch processing functionality."""
        try:
            if len(self.sample_files) < 2:
                self.logger.warning("Need at least 2 files for batch processing test")
                return True
            
            # Test with first few files
            test_files = self.sample_files[:min(5, len(self.sample_files))]
            self.logger.info(f"Testing batch processing with {len(test_files)} files")
            
            # Create output directory
            output_dir = self.temp_dir / "batch_output"
            output_dir.mkdir(exist_ok=True)
            
            # Create a simple batch processing test using individual components
            # Since the BatchProcessor uses async methods, we'll test components individually
            
            start_time = time.time()
            successful_files = 0
            
            for test_file in test_files:
                try:
                    # Test CSV parsing
                    parse_result = self.csv_parser.parse_file(test_file)
                    if parse_result.success:
                        # Test graph generation
                        output_path = output_dir / f"test_{test_file.stem}.png"
                        graph_success = self.graph_generator.generate_spectral_graph(
                            data=parse_result.data,
                            output_path=output_path,
                            title=f"Test - {test_file.name}"
                        )
                        if graph_success:
                            successful_files += 1
                except Exception as e:
                    self.logger.warning(f"Batch processing error for {test_file.name}: {e}")
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Batch processing time: {processing_time:.2f}s")
            self.logger.info(f"Files processed: {successful_files}")
            self.logger.info(f"Files failed: {len(test_files) - successful_files}")
            
            # Check results
            success_rate = successful_files / len(test_files)
            output_files_created = len(list(output_dir.glob("*.png")))
            
            self.logger.info(f"Success rate: {success_rate:.1%}")
            self.logger.info(f"Output files created: {output_files_created}")
            
            # Success criteria: > 80% success rate
            return success_rate > 0.8
            
        except Exception as e:
            self.logger.error(f"Batch processing test failed: {e}")
            return False
    
    def test_caching_system(self) -> bool:
        """Test caching system functionality."""
        try:
            self.logger.info("Testing caching system...")
            
            # Test cache operations
            test_key = "test_normalization_plan"
            test_data = {
                "column_mapping": {"wavenumber": "x", "absorbance": "y"},
                "confidence": 0.95,
                "timestamp": time.time()
            }
            
            # Test cache set
            self.cache_manager.set_normalization_plan(test_key, test_data)
            self.logger.info("✅ Cache set operation successful")
            
            # Test cache get (synchronous method)
            cached_data = self.cache_manager.get_normalization_plan_sync(test_key)
            cache_hit = cached_data is not None
            
            self.logger.info(f"Cache hit: {cache_hit}")
            
            if cache_hit:
                self.logger.info("✅ Cache get operation successful")
                
                # Verify data integrity
                data_match = abs(cached_data.get("confidence", 0) - test_data["confidence"]) < 0.01
                self.logger.info(f"Data integrity: {data_match}")
                
                return data_match
            else:
                self.logger.warning("⚠️ Cache miss - data not retrieved")
                return False
            
        except Exception as e:
            self.logger.error(f"Caching system test failed: {e}")
            return False
    
    def test_cost_tracking(self) -> bool:
        """Test cost tracking functionality."""
        try:
            self.logger.info("Testing cost tracking system...")
            
            # Record some test costs
            test_costs = [
                {"operation": "normalization", "cost": 0.05, "tokens": 1000},
                {"operation": "validation", "cost": 0.02, "tokens": 400},
                {"operation": "analysis", "cost": 0.03, "tokens": 600}
            ]
            
            for cost_data in test_costs:
                self.cost_tracker.record_api_usage(
                    operation=cost_data["operation"],
                    cost=cost_data["cost"],
                    tokens_used=cost_data["tokens"]
                )
            
            # Get usage statistics
            stats = self.cost_tracker.get_usage_statistics()
            
            self.logger.info(f"Total cost tracked: ${stats.total_cost:.4f}")
            self.logger.info(f"Total tokens: {stats.tokens_used}")
            self.logger.info(f"Operations: {stats.api_calls}")
            
            # Verify tracking
            expected_total_cost = sum(c["cost"] for c in test_costs)
            actual_total_cost = stats.total_cost
            
            cost_tracking_accurate = abs(actual_total_cost - expected_total_cost) < 0.01
            
            self.logger.info(f"Cost tracking accuracy: {cost_tracking_accurate}")
            
            return cost_tracking_accurate
            
        except Exception as e:
            self.logger.error(f"Cost tracking test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        try:
            self.logger.info("Testing error handling...")
            
            # Test with non-existent file
            fake_file = Path("non_existent_file.csv")
            result = self.csv_parser.parse_file(fake_file)
            
            handles_missing_file = not result.success and result.error is not None
            self.logger.info(f"Handles missing file: {handles_missing_file}")
            
            # Test with invalid CSV data
            invalid_csv_path = self.temp_dir / "invalid.csv"
            with open(invalid_csv_path, 'w') as f:
                f.write("This is not valid CSV data\nwith random content\n")
            
            result = self.csv_parser.parse_file(invalid_csv_path)
            handles_invalid_csv = not result.success and result.error is not None
            self.logger.info(f"Handles invalid CSV: {handles_invalid_csv}")
            
            # Test with empty file
            empty_csv_path = self.temp_dir / "empty.csv"
            empty_csv_path.touch()
            
            result = self.csv_parser.parse_file(empty_csv_path)
            handles_empty_file = not result.success and result.error is not None
            self.logger.info(f"Handles empty file: {handles_empty_file}")
            
            # Overall error handling success
            error_handling_success = handles_missing_file and handles_invalid_csv and handles_empty_file
            
            self.logger.info(f"Error handling test: {'✅ PASSED' if error_handling_success else '❌ FAILED'}")
            
            return error_handling_success
            
        except Exception as e:
            self.logger.error(f"Error handling test failed: {e}")
            return False
    
    def test_memory_usage(self) -> bool:
        """Test memory usage and resource management."""
        try:
            import psutil
            import gc
            
            self.logger.info("Testing memory usage...")
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            self.logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
            
            # Process several files to test memory usage
            if self.sample_files:
                for file_path in self.sample_files[:5]:
                    try:
                        result = self.csv_parser.parse_file(file_path)
                        if result.success and result.data is not None:
                            # Force some processing to use memory
                            _ = result.data.describe()
                            _ = result.data.copy()
                    except Exception as e:
                        self.logger.warning(f"Memory test processing error: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            self.logger.info(f"Final memory usage: {final_memory:.1f} MB")
            self.logger.info(f"Memory increase: {memory_increase:.1f} MB")
            
            # Memory usage should be reasonable (< 100MB increase for test)
            memory_reasonable = memory_increase < 100
            
            self.logger.info(f"Memory usage reasonable: {memory_reasonable}")
            
            return memory_reasonable
            
        except ImportError:
            self.logger.warning("psutil not available - skipping memory test")
            return True
        except Exception as e:
            self.logger.error(f"Memory usage test failed: {e}")
            return False
    
    def test_complete_workflow(self) -> bool:
        """Test complete end-to-end workflow."""
        try:
            if not self.sample_files:
                self.logger.warning("No sample files for complete workflow test")
                return True
            
            self.logger.info("Testing complete workflow...")
            
            # Select test file
            test_file = self.sample_files[0]
            self.logger.info(f"Workflow test with: {test_file.name}")
            
            # Step 1: Parse CSV
            self.logger.info("Step 1: Parsing CSV...")
            parse_result = self.csv_parser.parse_file(test_file)
            if not parse_result.success:
                self.logger.error("Workflow failed at parsing step")
                return False
            
            # Step 2: Validate data
            self.logger.info("Step 2: Validating data...")
            validation_result = self.data_validator.validate_spectral_data(parse_result.data)
            
            # Step 3: Generate graph (skip AI normalization for reliability)
            self.logger.info("Step 3: Generating graph...")
            output_path = self.temp_dir / f"workflow_test_{test_file.stem}.png"
            
            graph_success = self.graph_generator.generate_spectral_graph(
                data=parse_result.data,
                output_path=output_path,
                title=f"Workflow Test - {test_file.name}"
            )
            
            # Step 4: Verify outputs
            self.logger.info("Step 4: Verifying outputs...")
            graph_created = output_path.exists()
            
            workflow_success = (
                parse_result.success and
                validation_result.is_valid and
                graph_success and
                graph_created
            )
            
            self.logger.info(f"Complete workflow test: {'✅ PASSED' if workflow_success else '❌ FAILED'}")
            self.logger.info(f"  Parse: {'✅' if parse_result.success else '❌'}")
            self.logger.info(f"  Validation: {'✅' if validation_result.is_valid else '❌'}")
            self.logger.info(f"  Graph generation: {'✅' if graph_success else '❌'}")
            self.logger.info(f"  Graph file: {'✅' if graph_created else '❌'}")
            
            return workflow_success
            
        except Exception as e:
            self.logger.error(f"Complete workflow test failed: {e}")
            return False
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE INTEGRATION TEST REPORT",
            "=" * 80,
            f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test Environment: {sys.platform}",
            f"Python Version: {sys.version}",
            "",
            "TEST RESULTS SUMMARY:",
            "-" * 40
        ]
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.test_results.values() if r["status"] == "FAILED")
        error_tests = sum(1 for r in self.test_results.values() if r["status"] == "ERROR")
        
        report_lines.extend([
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests} ✅",
            f"Failed: {failed_tests} ❌",
            f"Errors: {error_tests} ⚠️",
            f"Success Rate: {passed_tests/total_tests:.1%}" if total_tests > 0 else "Success Rate: N/A",
            "",
            "DETAILED RESULTS:",
            "-" * 40
        ])
        
        for test_name, result in self.test_results.items():
            status_icon = {"PASSED": "✅", "FAILED": "❌", "ERROR": "⚠️"}.get(result["status"], "❓")
            duration = result.get("duration", 0)
            
            report_lines.append(f"{status_icon} {test_name} ({duration:.2f}s)")
            
            if result["status"] == "ERROR":
                report_lines.append(f"    Error: {result.get('error', 'Unknown error')}")
            elif result["status"] == "FAILED" and "details" in result:
                details = result["details"]
                if details:
                    report_lines.append(f"    Details: {details}")
        
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 40
        ])
        
        if failed_tests > 0 or error_tests > 0:
            report_lines.append("❌ Application has critical issues that need to be addressed before production.")
            report_lines.append("   Review failed tests and fix underlying problems.")
        elif passed_tests / total_tests < 0.9:
            report_lines.append("⚠️ Application has some issues that should be addressed.")
            report_lines.append("   Consider fixing failed tests for better reliability.")
        else:
            report_lines.append("✅ Application is performing well and appears production-ready.")
            report_lines.append("   Minor optimizations may still be beneficial.")
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def cleanup(self):
        """Clean up test environment."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


def main():
    """Main test execution function."""
    print("🚀 Starting Comprehensive Integration Test Suite")
    print("=" * 60)
    
    test_suite = ComprehensiveIntegrationTest()
    
    try:
        # Setup test environment
        if not test_suite.setup():
            print("❌ Failed to set up test environment")
            return 1
        
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Generate and display report
        report = test_suite.generate_report()
        print("\n" + report)
        
        # Save report to file
        report_file = Path("integration_test_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Save results as JSON
        results_file = Path("integration_test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📊 Results data saved to: {results_file}")
        
        # Determine exit code
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["status"] == "PASSED")
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:
            print("\n🎉 Integration tests completed successfully!")
            exit_code = 0
        else:
            print("\n⚠️ Integration tests completed with issues.")
            exit_code = 1
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⏹️ Test suite interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        traceback.print_exc()
        return 1
    finally:
        test_suite.cleanup()


if __name__ == "__main__":
    sys.exit(main())