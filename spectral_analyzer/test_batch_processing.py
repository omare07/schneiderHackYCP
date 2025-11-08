"""
Comprehensive test and demonstration of batch processing capabilities.

Tests the complete batch processing workflow including file validation,
AI normalization, graph generation, and error handling.
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from utils.batch_processor import BatchProcessor
from utils.file_manager import OutputSettings, BatchConfig, ProcessingProgress
from core.csv_parser import CSVParser
from core.ai_normalizer import AINormalizer
from core.graph_generator import SpectralGraphGenerator


class BatchProcessingTester:
    """Comprehensive tester for batch processing functionality."""
    
    def __init__(self):
        """Initialize the tester."""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = None
        self.test_files = []
        
        # Initialize components
        self.batch_processor = BatchProcessor()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_test_environment(self):
        """Create temporary directory and test files."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="spectral_batch_test_"))
        self.logger.info(f"Created test directory: {self.temp_dir}")
        
        # Create test data directory
        test_data_dir = self.temp_dir / "test_data"
        test_data_dir.mkdir()
        
        # Create baseline file
        baseline_file = test_data_dir / "baseline.csv"
        self._create_test_spectral_file(baseline_file, "baseline", noise_level=0.01)
        
        # Create sample files with variations
        sample_files = []
        for i in range(5):
            sample_file = test_data_dir / f"sample_{i+1}.csv"
            self._create_test_spectral_file(
                sample_file, f"sample_{i+1}", 
                noise_level=0.02 + i * 0.01,
                shift_factor=i * 0.1
            )
            sample_files.append(str(sample_file))
        
        # Create a problematic file for error testing
        problematic_file = test_data_dir / "problematic.csv"
        self._create_problematic_file(problematic_file)
        sample_files.append(str(problematic_file))
        
        self.test_files = {
            'baseline': str(baseline_file),
            'samples': sample_files,
            'output_dir': str(self.temp_dir / "output")
        }
        
        self.logger.info(f"Created {len(sample_files)} test files")
    
    def _create_test_spectral_file(self, filepath: Path, name: str, 
                                 noise_level: float = 0.01, shift_factor: float = 0.0):
        """Create a synthetic spectral data file."""
        # Generate wavenumber range (4000 to 400 cm⁻¹)
        wavenumbers = np.linspace(4000, 400, 1000)
        
        # Generate synthetic absorbance data with peaks
        absorbance = np.zeros_like(wavenumbers)
        
        # Add some characteristic peaks
        peaks = [3500, 2900, 1650, 1450, 1000]  # Common IR peaks
        for peak in peaks:
            # Gaussian peak
            sigma = 50
            amplitude = 0.8 + shift_factor
            absorbance += amplitude * np.exp(-((wavenumbers - peak) ** 2) / (2 * sigma ** 2))
        
        # Add baseline and noise
        baseline = 0.1 + shift_factor
        noise = np.random.normal(0, noise_level, len(wavenumbers))
        absorbance = absorbance + baseline + noise
        
        # Ensure non-negative values
        absorbance = np.maximum(absorbance, 0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'wavenumber': wavenumbers,
            'absorbance': absorbance
        })
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        self.logger.debug(f"Created test file: {filepath}")
    
    def _create_problematic_file(self, filepath: Path):
        """Create a file with issues to test error handling."""
        # Create file with missing data and inconsistent format
        data = [
            "# This is a comment line",
            "Frequency,Signal,Extra Column",
            "4000,0.5,metadata",
            "3900,,0.6",  # Missing value
            "3800,invalid,0.7",  # Invalid numeric value
            "3700,0.8,metadata",
            "",  # Empty line
            "3600,0.9,metadata"
        ]
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(data))
        
        self.logger.debug(f"Created problematic test file: {filepath}")
    
    async def test_basic_batch_processing(self):
        """Test basic batch processing functionality."""
        self.logger.info("=== Testing Basic Batch Processing ===")
        
        # Setup output settings
        output_settings = OutputSettings(
            output_dir=self.test_files['output_dir'],
            format='png',
            dpi=150,  # Lower DPI for faster testing
            open_folder_after=False,
            save_normalized_csvs=True
        )
        
        # Setup batch config
        config = BatchConfig(
            max_concurrent_files=2,
            memory_limit_mb=500,
            timeout_per_file_minutes=5,
            continue_on_error=True,
            cleanup_temp_files=True
        )
        
        # Progress callback for testing
        progress_updates = []
        
        def progress_callback(progress: ProcessingProgress):
            progress_updates.append({
                'stage': progress.current_stage.value,
                'file': progress.current_file,
                'percent': progress.progress_percent,
                'operation': progress.current_operation
            })
            self.logger.info(
                f"Progress: {progress.progress_percent:.1f}% - "
                f"{progress.current_stage.value} - {progress.current_operation}"
            )
        
        # Execute batch processing
        result = await self.batch_processor.process_spectral_batch(
            baseline_file=self.test_files['baseline'],
            sample_files=self.test_files['samples'],
            output_settings=output_settings,
            progress_callback=progress_callback,
            config=config
        )
        
        # Analyze results
        self.logger.info(f"Batch processing completed:")
        self.logger.info(f"  Total files: {result.total_files}")
        self.logger.info(f"  Successful: {result.successful}")
        self.logger.info(f"  Failed: {result.failed}")
        self.logger.info(f"  Processing time: {result.processing_time:.2f}s")
        self.logger.info(f"  AI normalizations: {result.ai_normalizations}")
        self.logger.info(f"  Cache hits: {result.cache_hits}")
        self.logger.info(f"  Output files: {len(result.output_files)}")
        
        if result.errors:
            self.logger.warning(f"Errors encountered: {result.errors}")
        
        if result.warnings:
            self.logger.warning(f"Warnings: {result.warnings}")
        
        # Verify output files exist
        output_dir = Path(self.test_files['output_dir'])
        if output_dir.exists():
            output_files = list(output_dir.glob("*.png"))
            self.logger.info(f"Generated {len(output_files)} graph files")
            
            csv_files = list(output_dir.glob("*.csv"))
            if csv_files:
                self.logger.info(f"Generated {len(csv_files)} normalized CSV files")
        
        return result, progress_updates
    
    async def test_file_validation(self):
        """Test file validation functionality."""
        self.logger.info("=== Testing File Validation ===")
        
        validation_result = await self.batch_processor.file_manager.validate_batch_files(
            baseline_file=self.test_files['baseline'],
            sample_files=self.test_files['samples']
        )
        
        self.logger.info(f"Validation results:")
        self.logger.info(f"  Valid files: {len(validation_result.valid_files)}")
        self.logger.info(f"  Invalid files: {len(validation_result.invalid_files)}")
        self.logger.info(f"  Total size: {validation_result.total_size_mb:.2f} MB")
        self.logger.info(f"  Estimated time: {validation_result.estimated_time_minutes:.1f} minutes")
        
        if validation_result.invalid_files:
            self.logger.info("Invalid files:")
            for filepath, reason in validation_result.invalid_files:
                self.logger.info(f"  {filepath}: {reason}")
        
        if validation_result.warnings:
            self.logger.info(f"Warnings: {validation_result.warnings}")
        
        return validation_result
    
    async def test_filename_conflict_resolution(self):
        """Test filename conflict resolution."""
        self.logger.info("=== Testing Filename Conflict Resolution ===")
        
        # Create a test file
        test_file = Path(self.temp_dir) / "test_conflict.png"
        test_file.touch()
        
        # Test different conflict resolution strategies
        strategies = ["auto_increment", "overwrite", "skip"]
        
        for strategy in strategies:
            resolved = self.batch_processor.file_manager.resolve_filename_conflicts(
                str(test_file), strategy
            )
            self.logger.info(f"Strategy '{strategy}': {test_file} -> {resolved}")
    
    async def test_memory_monitoring(self):
        """Test memory monitoring during processing."""
        self.logger.info("=== Testing Memory Monitoring ===")
        
        stats = self.batch_processor.get_processing_stats()
        self.logger.info(f"Current memory usage: {stats['memory_usage_mb']:.1f} MB")
        
        # Test processing time estimation
        estimate = await self.batch_processor.estimate_processing_time(
            baseline_file=self.test_files['baseline'],
            sample_files=self.test_files['samples']
        )
        
        self.logger.info(f"Processing estimate:")
        self.logger.info(f"  Valid files: {estimate['valid_files']}")
        self.logger.info(f"  Total size: {estimate['total_size_mb']:.2f} MB")
        self.logger.info(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")
        self.logger.info(f"  Estimated memory: {estimate['estimated_memory_mb']:.1f} MB")
        
        return estimate
    
    async def test_error_handling(self):
        """Test error handling and recovery."""
        self.logger.info("=== Testing Error Handling ===")
        
        # Test with non-existent baseline file
        try:
            result = await self.batch_processor.process_spectral_batch(
                baseline_file="non_existent_file.csv",
                sample_files=self.test_files['samples'][:2],
                output_settings=OutputSettings(output_dir=str(self.temp_dir / "error_test"))
            )
            self.logger.info(f"Error handling test - Errors: {result.errors}")
        except Exception as e:
            self.logger.info(f"Expected error caught: {e}")
        
        # Test with empty file list
        result = await self.batch_processor.process_spectral_batch(
            baseline_file=self.test_files['baseline'],
            sample_files=[],
            output_settings=OutputSettings(output_dir=str(self.temp_dir / "empty_test"))
        )
        self.logger.info(f"Empty file list test - Total files: {result.total_files}")
    
    async def run_all_tests(self):
        """Run all batch processing tests."""
        try:
            self.setup_test_environment()
            
            # Run individual tests
            await self.test_file_validation()
            await self.test_filename_conflict_resolution()
            await self.test_memory_monitoring()
            await self.test_error_handling()
            
            # Run main batch processing test
            result, progress_updates = await self.test_basic_batch_processing()
            
            self.logger.info("=== Test Summary ===")
            self.logger.info(f"All tests completed successfully!")
            self.logger.info(f"Progress updates received: {len(progress_updates)}")
            self.logger.info(f"Final result: {result.successful}/{result.total_files} files processed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up test directory: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up test directory: {e}")


async def main():
    """Main test execution function."""
    print("Starting Batch Processing Tests...")
    print("=" * 50)
    
    tester = BatchProcessingTester()
    success = await tester.run_all_tests()
    
    print("=" * 50)
    if success:
        print("✅ All batch processing tests completed successfully!")
    else:
        print("❌ Some tests failed. Check the logs for details.")
    
    return success


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    exit(0 if success else 1)