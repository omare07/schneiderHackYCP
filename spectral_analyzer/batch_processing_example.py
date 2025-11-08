"""
Batch Processing Usage Example for Spectral Analyzer.

This example demonstrates how to use the comprehensive batch processing
capabilities for spectral analysis with baseline comparison.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

from utils.batch_processor import BatchProcessor
from utils.file_manager import OutputSettings, BatchConfig, ProcessingProgress


class SpectralBatchExample:
    """Example usage of batch processing for spectral analysis."""
    
    def __init__(self):
        """Initialize the example."""
        self.logger = logging.getLogger(__name__)
        self.batch_processor = BatchProcessor()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    async def basic_batch_processing_example(self):
        """Basic batch processing example."""
        print("=== Basic Batch Processing Example ===")
        
        # Define file paths
        baseline_file = "path/to/baseline.csv"
        sample_files = [
            "path/to/sample1.csv",
            "path/to/sample2.csv",
            "path/to/sample3.csv"
        ]
        
        # Configure output settings
        output_settings = OutputSettings(
            output_dir="./batch_output",
            format='png',
            dpi=300,
            open_folder_after=True,
            save_normalized_csvs=True,
            filename_template="{sample}_vs_{baseline}",
            conflict_resolution="auto_increment"
        )
        
        # Configure batch processing
        config = BatchConfig(
            max_concurrent_files=3,
            memory_limit_mb=1000,
            timeout_per_file_minutes=10,
            retry_failed_files=True,
            continue_on_error=True,
            cleanup_temp_files=True
        )
        
        # Define progress callback
        def progress_callback(progress: ProcessingProgress):
            print(f"Progress: {progress.progress_percent:.1f}% - "
                  f"{progress.current_stage.value} - {progress.current_operation}")
            
            if progress.current_file:
                print(f"  Current file: {progress.current_file}")
            
            if progress.errors_count > 0:
                print(f"  Errors so far: {progress.errors_count}")
        
        try:
            # Execute batch processing
            result = await self.batch_processor.process_spectral_batch(
                baseline_file=baseline_file,
                sample_files=sample_files,
                output_settings=output_settings,
                progress_callback=progress_callback,
                config=config
            )
            
            # Display results
            print(f"\nBatch Processing Results:")
            print(f"  Total files: {result.total_files}")
            print(f"  Successful: {result.successful}")
            print(f"  Failed: {result.failed}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  AI normalizations: {result.ai_normalizations}")
            print(f"  Cache hits: {result.cache_hits}")
            print(f"  Output files generated: {len(result.output_files)}")
            
            if result.errors:
                print(f"\nErrors encountered:")
                for error in result.errors:
                    print(f"  - {error}")
            
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            
            return result
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            return None
    
    async def advanced_batch_processing_example(self):
        """Advanced batch processing with custom settings."""
        print("\n=== Advanced Batch Processing Example ===")
        
        # Get processing time estimate first
        baseline_file = "path/to/baseline.csv"
        sample_files = ["path/to/sample1.csv", "path/to/sample2.csv"]
        
        estimate = await self.batch_processor.estimate_processing_time(
            baseline_file, sample_files
        )
        
        print(f"Processing Estimate:")
        print(f"  Valid files: {estimate.get('valid_files', 0)}")
        print(f"  Estimated time: {estimate.get('estimated_time_minutes', 0):.1f} minutes")
        print(f"  Estimated memory: {estimate.get('estimated_memory_mb', 0):.1f} MB")
        
        # Configure for high-quality output
        output_settings = OutputSettings(
            output_dir="./high_quality_output",
            format='png',
            dpi=600,  # High DPI for publication quality
            open_folder_after=False,
            save_normalized_csvs=True
        )
        
        # Configure for memory-efficient processing
        config = BatchConfig(
            max_concurrent_files=1,  # Process one at a time for large files
            memory_limit_mb=500,
            timeout_per_file_minutes=15,
            retry_failed_files=True,
            max_retries=3,
            continue_on_error=True
        )
        
        # Advanced progress tracking
        progress_history = []
        
        def detailed_progress_callback(progress: ProcessingProgress):
            progress_history.append({
                'timestamp': asyncio.get_event_loop().time(),
                'stage': progress.current_stage.value,
                'percent': progress.progress_percent,
                'file': progress.current_file,
                'operation': progress.current_operation
            })
            
            print(f"[{progress.current_stage.value.upper()}] "
                  f"{progress.progress_percent:.1f}% - {progress.current_operation}")
        
        try:
            result = await self.batch_processor.process_spectral_batch(
                baseline_file=baseline_file,
                sample_files=sample_files,
                output_settings=output_settings,
                progress_callback=detailed_progress_callback,
                config=config
            )
            
            print(f"\nAdvanced Processing Complete!")
            print(f"Progress updates received: {len(progress_history)}")
            
            return result, progress_history
            
        except Exception as e:
            print(f"Advanced batch processing failed: {e}")
            return None, progress_history
    
    async def file_validation_example(self):
        """Example of file validation before processing."""
        print("\n=== File Validation Example ===")
        
        baseline_file = "path/to/baseline.csv"
        sample_files = [
            "path/to/sample1.csv",
            "path/to/sample2.csv",
            "path/to/invalid_file.txt",  # This will fail validation
            "path/to/nonexistent.csv"    # This will also fail
        ]
        
        # Validate files before processing
        validation_result = await self.batch_processor.file_manager.validate_batch_files(
            baseline_file=baseline_file,
            sample_files=sample_files
        )
        
        print(f"Validation Results:")
        print(f"  Valid files: {len(validation_result.valid_files)}")
        print(f"  Invalid files: {len(validation_result.invalid_files)}")
        print(f"  Total size: {validation_result.total_size_mb:.2f} MB")
        print(f"  Estimated processing time: {validation_result.estimated_time_minutes:.1f} minutes")
        
        if validation_result.invalid_files:
            print(f"\nInvalid files:")
            for filepath, reason in validation_result.invalid_files:
                print(f"  {filepath}: {reason}")
        
        if validation_result.warnings:
            print(f"\nWarnings:")
            for warning in validation_result.warnings:
                print(f"  {warning}")
        
        return validation_result
    
    async def error_handling_example(self):
        """Example of error handling and recovery."""
        print("\n=== Error Handling Example ===")
        
        # Configure to continue on errors
        config = BatchConfig(
            continue_on_error=True,
            retry_failed_files=True,
            max_retries=2
        )
        
        output_settings = OutputSettings(
            output_dir="./error_test_output",
            conflict_resolution="skip"  # Skip files that would conflict
        )
        
        # Mix of valid and invalid files
        baseline_file = "path/to/baseline.csv"
        sample_files = [
            "path/to/valid_sample.csv",
            "path/to/corrupted_sample.csv",
            "path/to/nonexistent.csv"
        ]
        
        def error_tracking_callback(progress: ProcessingProgress):
            if progress.errors_count > 0:
                print(f"⚠️  Errors encountered: {progress.errors_count}")
            if progress.warnings_count > 0:
                print(f"⚠️  Warnings: {progress.warnings_count}")
        
        try:
            result = await self.batch_processor.process_spectral_batch(
                baseline_file=baseline_file,
                sample_files=sample_files,
                output_settings=output_settings,
                progress_callback=error_tracking_callback,
                config=config
            )
            
            print(f"\nError Handling Results:")
            print(f"  Successful: {result.successful}")
            print(f"  Failed: {result.failed}")
            print(f"  Continued processing despite errors: {result.failed > 0 and result.successful > 0}")
            
            return result
            
        except Exception as e:
            print(f"Critical error in batch processing: {e}")
            return None
    
    def filename_conflict_example(self):
        """Example of filename conflict resolution."""
        print("\n=== Filename Conflict Resolution Example ===")
        
        # Test different conflict resolution strategies
        test_file = "output/sample1_vs_baseline.png"
        
        strategies = ["auto_increment", "overwrite", "skip"]
        
        for strategy in strategies:
            resolved = self.batch_processor.file_manager.resolve_filename_conflicts(
                test_file, strategy
            )
            print(f"Strategy '{strategy}': {test_file} -> {resolved}")
    
    async def memory_monitoring_example(self):
        """Example of memory monitoring during processing."""
        print("\n=== Memory Monitoring Example ===")
        
        # Get current processing statistics
        stats = self.batch_processor.get_processing_stats()
        
        print(f"Current Processing Stats:")
        print(f"  Is processing: {stats['is_processing']}")
        print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"  Peak memory: {stats['memory_peak_mb']:.1f} MB")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  AI normalizations: {stats['ai_normalizations']}")
        print(f"  Cache hits: {stats['cache_hits']}")
        
        return stats
    
    async def run_all_examples(self):
        """Run all batch processing examples."""
        print("Spectral Analyzer - Batch Processing Examples")
        print("=" * 50)
        
        try:
            # Run examples (most will fail with dummy paths, but show the API)
            await self.file_validation_example()
            await self.memory_monitoring_example()
            self.filename_conflict_example()
            
            print("\n" + "=" * 50)
            print("✅ All examples completed!")
            print("\nNote: Most examples use dummy file paths and will show")
            print("validation errors. Replace with actual file paths to test.")
            
        except Exception as e:
            print(f"❌ Example execution failed: {e}")


async def main():
    """Main example execution."""
    example = SpectralBatchExample()
    await example.run_all_examples()


if __name__ == "__main__":
    asyncio.run(main())