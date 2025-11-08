"""
Batch processing coordinator for spectral analysis.

Coordinates CSV parsing, AI normalization, and graph generation
for multiple files with comprehensive progress tracking and error handling.
"""

import asyncio
import logging
import time
import psutil
import os
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from core.csv_parser import CSVParser
from core.ai_normalizer import AINormalizer
from core.graph_generator import SpectralGraphGenerator
from utils.file_manager import (
    FileManager, OutputSettings, BatchProcessResult, ProcessingProgress,
    BatchConfig, ProcessingStage, FileManagerError
)
from utils.cache_manager import CacheManager


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing session."""
    start_time: float = field(default_factory=time.time)
    files_processed: int = 0
    ai_normalizations_used: int = 0
    cache_hits: int = 0
    memory_peak_mb: float = 0.0
    processing_stages: Dict[str, float] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)


class BatchProcessor:
    """
    Comprehensive batch processing coordinator for spectral analysis.
    
    Features:
    - Coordinates CSV parsing, AI normalization, and graph generation
    - Memory-efficient processing with controlled concurrency
    - Comprehensive progress tracking and error handling
    - Automatic retry logic and fallback strategies
    - Integration with caching system for cost optimization
    - Detailed statistics and performance monitoring
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the batch processor.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize components
        self.csv_parser = CSVParser()
        self.ai_normalizer = AINormalizer(config_manager)
        self.graph_generator = SpectralGraphGenerator()
        self.file_manager = FileManager()
        self.cache_manager = CacheManager()
        
        # Processing state
        self.is_processing = False
        self.cancel_requested = False
        self.current_stats = BatchProcessingStats()
        
        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        
        self.logger.info("BatchProcessor initialized")
    
    async def process_spectral_batch(self, baseline_file: str, 
                                   sample_files: List[str],
                                   output_settings: OutputSettings,
                                   progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
                                   config: Optional[BatchConfig] = None) -> BatchProcessResult:
        """
        Process a batch of spectral files with baseline comparison.
        
        Args:
            baseline_file: Path to baseline CSV file
            sample_files: List of sample CSV file paths
            output_settings: Output configuration
            progress_callback: Optional callback for progress updates
            config: Optional batch configuration
            
        Returns:
            BatchProcessResult with comprehensive processing statistics
        """
        if config is None:
            config = BatchConfig()
        
        # Reset processing state
        self.is_processing = True
        self.cancel_requested = False
        self.current_stats = BatchProcessingStats()
        
        start_time = time.time()
        result = BatchProcessResult(
            total_files=len(sample_files),
            successful=0,
            failed=0
        )
        
        progress = ProcessingProgress(
            total_files=len(sample_files),
            current_stage=ProcessingStage.VALIDATION
        )
        
        temp_files = []
        baseline_data = None
        
        try:
            self.logger.info(f"Starting spectral batch processing: {len(sample_files)} files")
            
            # Phase 1: File Validation
            stage_start = time.time()
            if progress_callback:
                progress.current_operation = "Validating files..."
                progress_callback(progress)
            
            validation_result = await self._validate_files_phase(
                baseline_file, sample_files, progress, progress_callback
            )
            
            if not validation_result.valid_files:
                result.errors.append("No valid files to process")
                return result
            
            result.warnings.extend(validation_result.warnings)
            self.current_stats.processing_stages['validation'] = time.time() - stage_start
            
            # Phase 2: Data Loading and Baseline Processing
            stage_start = time.time()
            progress.current_stage = ProcessingStage.LOADING
            if progress_callback:
                progress.current_operation = "Loading baseline file..."
                progress_callback(progress)
            
            baseline_data = await self._load_baseline_phase(
                baseline_file, progress, progress_callback, config
            )
            
            if baseline_data is None:
                result.errors.append("Failed to load baseline file")
                return result
            
            self.current_stats.processing_stages['loading'] = time.time() - stage_start
            
            # Phase 3: Batch Processing
            stage_start = time.time()
            progress.current_stage = ProcessingStage.GRAPH_GENERATION
            progress.total_files = len(validation_result.valid_files)
            
            # Create output directory
            output_dir = Path(output_settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process files with controlled concurrency
            processing_results = await self._process_files_phase(
                baseline_file, baseline_data, validation_result.valid_files,
                output_settings, progress, progress_callback, config, temp_files
            )
            
            # Aggregate results
            for success, output_file, warnings, stats in processing_results:
                if success:
                    result.successful += 1
                    if output_file:
                        result.output_files.append(output_file)
                else:
                    result.failed += 1
                
                result.warnings.extend(warnings)
                
                # Update statistics
                if stats:
                    if stats.get('ai_normalization_used'):
                        result.ai_normalizations += 1
                    if stats.get('cache_hit'):
                        result.cache_hits += 1
            
            self.current_stats.processing_stages['processing'] = time.time() - stage_start
            
            # Phase 4: Cleanup
            stage_start = time.time()
            progress.current_stage = ProcessingStage.CLEANUP
            if progress_callback:
                progress.current_operation = "Cleaning up temporary files..."
                progress_callback(progress)
            
            if config.cleanup_temp_files:
                cleaned_count = await self.file_manager.cleanup_temp_files_batch(temp_files)
                self.logger.debug(f"Cleaned up {cleaned_count} temporary files")
            
            self.current_stats.processing_stages['cleanup'] = time.time() - stage_start
            
            # Final statistics
            result.processing_time = time.time() - start_time
            result.memory_peak_mb = self._get_peak_memory_usage()
            
            if progress_callback:
                progress.progress_percent = 100.0
                progress.current_operation = "Batch processing complete"
                progress_callback(progress)
            
            self.logger.info(
                f"Batch processing complete: {result.successful}/{result.total_files} successful "
                f"in {result.processing_time:.2f}s, {result.ai_normalizations} AI normalizations, "
                f"{result.cache_hits} cache hits"
            )
            
            # Open output folder if requested
            if output_settings.open_folder_after and result.successful > 0:
                self._open_output_folder(output_dir)
            
            return result
            
        except Exception as e:
            result.errors.append(f"Batch processing failed: {str(e)}")
            self.logger.error(f"Batch processing error: {e}")
            
            # Cleanup on error
            if config.cleanup_temp_files:
                await self.file_manager.cleanup_temp_files_batch(temp_files)
            
            result.processing_time = time.time() - start_time
            return result
        
        finally:
            self.is_processing = False
            self.cancel_requested = False
    
    async def _validate_files_phase(self, baseline_file: str, sample_files: List[str],
                                  progress: ProcessingProgress,
                                  progress_callback: Optional[Callable[[ProcessingProgress], None]]):
        """Validate all files for batch processing."""
        try:
            return await self.file_manager.validate_batch_files(
                baseline_file, sample_files, progress_callback
            )
        except Exception as e:
            self.logger.error(f"File validation phase failed: {e}")
            raise
    
    async def _load_baseline_phase(self, baseline_file: str,
                                 progress: ProcessingProgress,
                                 progress_callback: Optional[Callable[[ProcessingProgress], None]],
                                 config: BatchConfig) -> Optional[pd.DataFrame]:
        """Load and normalize baseline file."""
        try:
            # Parse baseline file
            parse_result = self.csv_parser.parse_file(baseline_file)
            
            if not parse_result.success:
                self.logger.error(f"Failed to parse baseline file: {parse_result.issues}")
                return None
            
            baseline_data = parse_result.data
            
            # Check if AI normalization is needed
            if parse_result.structure.confidence < 0.8:
                if progress_callback:
                    progress.current_operation = "Normalizing baseline with AI..."
                    progress_callback(progress)
                
                normalization_result = await self.ai_normalizer.normalize_csv(
                    baseline_data, baseline_file
                )
                
                if normalization_result.success:
                    baseline_data = normalization_result.normalized_data
                    self.current_stats.ai_normalizations_used += 1
                    
                    if normalization_result.cache_hit:
                        self.current_stats.cache_hits += 1
                else:
                    self.logger.warning(f"AI normalization failed for baseline: {normalization_result.error_message}")
            
            return baseline_data
            
        except Exception as e:
            self.logger.error(f"Baseline loading phase failed: {e}")
            return None
    
    async def _process_files_phase(self, baseline_file: str, baseline_data: pd.DataFrame,
                                 sample_files: List[str], output_settings: OutputSettings,
                                 progress: ProcessingProgress,
                                 progress_callback: Optional[Callable[[ProcessingProgress], None]],
                                 config: BatchConfig, temp_files: List[str]) -> List[Tuple[bool, str, List[str], Dict[str, Any]]]:
        """Process all sample files against baseline."""
        try:
            # Control concurrency to manage memory usage
            semaphore = asyncio.Semaphore(config.max_concurrent_files)
            
            async def process_single_sample(sample_file: str, index: int):
                async with semaphore:
                    return await self._process_single_sample(
                        baseline_file, baseline_data, sample_file, output_settings,
                        progress, progress_callback, config, temp_files, index
                    )
            
            # Create tasks for all files
            tasks = [
                process_single_sample(sample_file, i)
                for i, sample_file in enumerate(sample_files)
            ]
            
            # Execute with progress tracking
            results = []
            for completed_task in asyncio.as_completed(tasks):
                if self.cancel_requested:
                    self.logger.info("Batch processing cancelled by user")
                    break
                
                try:
                    result = await completed_task
                    results.append(result)
                    
                    # Update progress
                    progress.files_completed = len(results)
                    progress.progress_percent = (len(results) / len(tasks)) * 100
                    
                    if progress_callback:
                        progress_callback(progress)
                    
                    # Memory check
                    current_memory = self._get_current_memory_usage()
                    if current_memory > config.memory_limit_mb:
                        self.logger.warning(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({config.memory_limit_mb}MB)")
                        # Could implement memory management strategies here
                    
                except Exception as e:
                    self.logger.error(f"Task execution failed: {e}")
                    results.append((False, None, [str(e)], {}))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Files processing phase failed: {e}")
            raise
    
    async def _process_single_sample(self, baseline_file: str, baseline_data: pd.DataFrame,
                                   sample_file: str, output_settings: OutputSettings,
                                   progress: ProcessingProgress,
                                   progress_callback: Optional[Callable[[ProcessingProgress], None]],
                                   config: BatchConfig, temp_files: List[str],
                                   file_index: int) -> Tuple[bool, str, List[str], Dict[str, Any]]:
        """Process a single sample file."""
        warnings = []
        stats = {}
        output_file = None
        
        try:
            sample_name = Path(sample_file).name
            
            # Update progress
            progress.current_file = sample_name
            progress.current_operation = f"Processing {sample_name}..."
            
            if progress_callback:
                progress_callback(progress)
            
            # Parse sample file
            parse_result = self.csv_parser.parse_file(sample_file)
            
            if not parse_result.success:
                warnings.append(f"Failed to parse {sample_name}: {', '.join(parse_result.issues)}")
                return False, None, warnings, stats
            
            sample_data = parse_result.data
            
            # AI normalization if needed
            if parse_result.structure.confidence < 0.8:
                normalization_result = await self.ai_normalizer.normalize_csv(
                    sample_data, sample_file
                )
                
                if normalization_result.success:
                    sample_data = normalization_result.normalized_data
                    stats['ai_normalization_used'] = True
                    stats['cache_hit'] = normalization_result.cache_hit
                else:
                    warnings.append(f"AI normalization failed for {sample_name}: {normalization_result.error_message}")
            
            # Generate comparison graph
            baseline_name = Path(baseline_file).stem
            sample_stem = Path(sample_file).stem
            
            fig = self.graph_generator.generate_comparison_graph(
                baseline_data, sample_data, baseline_name, sample_stem
            )
            
            # Generate output filename
            filename = output_settings.filename_template.format(
                sample=sample_stem,
                baseline=baseline_name
            )
            
            output_path = Path(output_settings.output_dir) / f"{filename}.{output_settings.format}"
            
            # Resolve filename conflicts
            resolved_path = self.file_manager.resolve_filename_conflicts(
                str(output_path), output_settings.conflict_resolution
            )
            
            if resolved_path is None:  # Skip strategy
                warnings.append(f"Skipped {sample_name} due to existing output file")
                return False, None, warnings, stats
            
            # Save graph
            success = self.graph_generator.save_graph(
                fig, resolved_path, output_settings.format, output_settings.dpi
            )
            
            if success:
                output_file = resolved_path
                
                # Save normalized CSV if requested
                if output_settings.save_normalized_csvs:
                    csv_path = Path(resolved_path).with_suffix('.csv')
                    sample_data.to_csv(csv_path, index=False)
                    temp_files.append(str(csv_path))
                
                return True, output_file, warnings, stats
            else:
                warnings.append(f"Failed to save graph for {sample_name}")
                return False, None, warnings, stats
            
        except Exception as e:
            warnings.append(f"Processing failed for {Path(sample_file).name}: {str(e)}")
            return False, output_file, warnings, stats
    
    def cancel_processing(self):
        """Cancel ongoing batch processing."""
        self.cancel_requested = True
        self.logger.info("Batch processing cancellation requested")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'is_processing': self.is_processing,
            'files_processed': self.current_stats.files_processed,
            'ai_normalizations': self.current_stats.ai_normalizations_used,
            'cache_hits': self.current_stats.cache_hits,
            'memory_usage_mb': self._get_current_memory_usage(),
            'memory_peak_mb': self.current_stats.memory_peak_mb,
            'processing_stages': self.current_stats.processing_stages.copy(),
            'error_types': self.current_stats.error_types.copy()
        }
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage during processing."""
        current_memory = self._get_current_memory_usage()
        self.current_stats.memory_peak_mb = max(self.current_stats.memory_peak_mb, current_memory)
        return self.current_stats.memory_peak_mb
    
    def _open_output_folder(self, output_dir: Path):
        """Open output folder in system file manager."""
        try:
            import subprocess
            import sys
            
            if sys.platform == "win32":
                subprocess.run(["explorer", str(output_dir)])
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_dir)])
            else:
                subprocess.run(["xdg-open", str(output_dir)])
                
        except Exception as e:
            self.logger.warning(f"Failed to open output folder: {e}")
    
    async def estimate_processing_time(self, baseline_file: str, sample_files: List[str]) -> Dict[str, Any]:
        """
        Estimate processing time and resource requirements.
        
        Args:
            baseline_file: Path to baseline file
            sample_files: List of sample file paths
            
        Returns:
            Dictionary with time and resource estimates
        """
        try:
            # Validate files to get size information
            validation_result = await self.file_manager.validate_batch_files(
                baseline_file, sample_files
            )
            
            total_size_mb = validation_result.total_size_mb
            valid_files = len(validation_result.valid_files)
            
            # Rough estimates based on file size and complexity
            # These could be refined based on actual performance data
            parsing_time = total_size_mb * 0.5  # seconds per MB
            ai_normalization_time = valid_files * 2.0  # seconds per file (if needed)
            graph_generation_time = valid_files * 1.5  # seconds per graph
            
            total_estimated_seconds = parsing_time + ai_normalization_time + graph_generation_time
            
            # Memory estimate (rough)
            estimated_memory_mb = total_size_mb * 3  # Factor for processing overhead
            
            return {
                'total_files': len(sample_files),
                'valid_files': valid_files,
                'invalid_files': len(validation_result.invalid_files),
                'total_size_mb': total_size_mb,
                'estimated_time_minutes': total_estimated_seconds / 60,
                'estimated_memory_mb': estimated_memory_mb,
                'warnings': validation_result.warnings
            }
            
        except Exception as e:
            self.logger.error(f"Time estimation failed: {e}")
            return {
                'error': str(e),
                'total_files': len(sample_files),
                'estimated_time_minutes': 0,
                'estimated_memory_mb': 0
            }