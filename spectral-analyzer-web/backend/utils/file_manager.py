"""
File management utilities for Spectral Analyzer.

Provides file operations, path handling, file system utilities,
and comprehensive batch processing capabilities for CSV files
and application data management.
"""

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import csv
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.error_handling import SpectralAnalyzerError


class FileManagerError(SpectralAnalyzerError):
    """Exception for file manager errors."""
    pass


class ProcessingStage(Enum):
    """Batch processing stages."""
    VALIDATION = "validation"
    LOADING = "loading"
    NORMALIZATION = "normalization"
    GRAPH_GENERATION = "graph_generation"
    SAVING = "saving"
    CLEANUP = "cleanup"


@dataclass
class OutputSettings:
    """Output configuration for batch processing."""
    output_dir: str
    format: str = 'png'
    dpi: int = 300
    open_folder_after: bool = True
    save_normalized_csvs: bool = False
    filename_template: str = "{sample}_vs_{baseline}"
    conflict_resolution: str = "auto_increment"  # auto_increment, overwrite, skip


@dataclass
class BatchProcessResult:
    """Result of batch processing operation."""
    total_files: int
    successful: int
    failed: int
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    ai_normalizations: int = 0
    cache_hits: int = 0
    skipped_files: int = 0
    memory_peak_mb: float = 0.0


@dataclass
class FileValidationResult:
    """Result of file validation operation."""
    valid_files: List[str] = field(default_factory=list)
    invalid_files: List[Tuple[str, str]] = field(default_factory=list)  # (filename, error_reason)
    total_size_mb: float = 0.0
    estimated_time_minutes: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcessingProgress:
    """Progress tracking for batch operations."""
    current_file: str = ""
    files_completed: int = 0
    total_files: int = 0
    current_stage: ProcessingStage = ProcessingStage.VALIDATION
    progress_percent: float = 0.0
    estimated_remaining_minutes: float = 0.0
    current_operation: str = ""
    errors_count: int = 0
    warnings_count: int = 0


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent_files: int = 3
    memory_limit_mb: int = 1000
    timeout_per_file_minutes: int = 10
    retry_failed_files: bool = True
    max_retries: int = 2
    continue_on_error: bool = True
    validate_outputs: bool = True
    cleanup_temp_files: bool = True


class FileManager:
    """
    Comprehensive file management system.
    
    Features:
    - Safe file operations with validation
    - CSV file handling and validation
    - Temporary file management
    - File metadata extraction
    - Backup and recovery operations
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize file manager.
        
        Args:
            base_dir: Base directory for file operations
        """
        self.logger = logging.getLogger(__name__)
        
        # Set base directory
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.home() / ".spectral_analyzer"
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.temp_dir = self.base_dir / "temp"
        self.backup_dir = self.base_dir / "backups"
        self.export_dir = self.base_dir / "exports"
        
        # Create subdirectories
        for directory in [self.temp_dir, self.backup_dir, self.export_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Supported file types
        self.supported_extensions = {'.csv', '.txt', '.tsv'}
        self.csv_extensions = {'.csv', '.tsv'}
        
        self.logger.debug("File manager initialized")
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate and normalize file path.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            FileManagerError: If path is invalid
        """
        try:
            path = Path(file_path)
            
            # Check if path exists
            if not path.exists():
                raise FileManagerError(f"File does not exist: {path}")
            
            # Check if it's a file (not directory)
            if not path.is_file():
                raise FileManagerError(f"Path is not a file: {path}")
            
            # Check file extension
            if path.suffix.lower() not in self.supported_extensions:
                raise FileManagerError(
                    f"Unsupported file type: {path.suffix}. "
                    f"Supported types: {', '.join(self.supported_extensions)}"
                )
            
            # Check file size (limit to 100MB)
            file_size = path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                raise FileManagerError(
                    f"File too large: {file_size / (1024*1024):.1f}MB. "
                    f"Maximum size: {max_size / (1024*1024)}MB"
                )
            
            return path
            
        except FileManagerError:
            raise
        except Exception as e:
            raise FileManagerError(f"Path validation failed: {e}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            # Basic file info
            file_info = {
                'path': str(path.absolute()),
                'name': path.name,
                'stem': path.stem,
                'suffix': path.suffix,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                'is_csv': path.suffix.lower() in self.csv_extensions,
                'mime_type': mimetypes.guess_type(str(path))[0]
            }
            
            # File hash for integrity checking
            file_info['md5_hash'] = self._calculate_file_hash(path)
            
            # CSV-specific information
            if file_info['is_csv']:
                csv_info = self._analyze_csv_file(path)
                file_info.update(csv_info)
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}: {e}")
            raise FileManagerError(f"Failed to get file info: {e}")
    
    def _calculate_file_hash(self, file_path: Path, algorithm: str = 'md5') -> str:
        """Calculate file hash for integrity checking."""
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate file hash: {e}")
            return ""
    
    def _analyze_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze CSV file structure and content."""
        try:
            csv_info = {
                'row_count': 0,
                'column_count': 0,
                'delimiter': ',',
                'has_header': False,
                'encoding': 'utf-8',
                'sample_data': [],
                'column_names': [],
                'issues': []
            }
            
            # Detect encoding
            csv_info['encoding'] = self._detect_encoding(file_path)
            
            # Analyze CSV structure
            with open(file_path, 'r', encoding=csv_info['encoding']) as f:
                # Detect delimiter
                sample = f.read(1024)
                f.seek(0)
                
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample, delimiters=',;\t|')
                    csv_info['delimiter'] = dialect.delimiter
                except csv.Error:
                    csv_info['delimiter'] = ','
                
                # Read CSV data
                reader = csv.reader(f, delimiter=csv_info['delimiter'])
                
                rows = []
                for i, row in enumerate(reader):
                    if i == 0:
                        csv_info['column_count'] = len(row)
                        csv_info['column_names'] = row
                        
                        # Check if first row is header
                        csv_info['has_header'] = self._detect_header(row)
                    
                    rows.append(row)
                    
                    # Limit sample data
                    if i < 10:
                        csv_info['sample_data'].append(row)
                    
                    if i >= 1000:  # Limit analysis to first 1000 rows
                        break
                
                csv_info['row_count'] = len(rows)
                
                # Validate CSV structure
                if csv_info['row_count'] == 0:
                    csv_info['issues'].append("File is empty")
                elif csv_info['column_count'] == 0:
                    csv_info['issues'].append("No columns detected")
                elif csv_info['column_count'] < 2:
                    csv_info['issues'].append("CSV should have at least 2 columns for spectral data")
            
            return csv_info
            
        except Exception as e:
            self.logger.warning(f"CSV analysis failed: {e}")
            return {
                'row_count': 0,
                'column_count': 0,
                'issues': [f"Analysis failed: {e}"]
            }
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            
            if encoding and result['confidence'] > 0.7:
                return encoding
            
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"Encoding detection failed: {e}")
        
        return 'utf-8'  # Default fallback
    
    def _detect_header(self, first_row: List[str]) -> bool:
        """Detect if first row is a header."""
        try:
            # Check if most values are non-numeric (likely header)
            numeric_count = 0
            for value in first_row:
                try:
                    float(value.strip())
                    numeric_count += 1
                except (ValueError, AttributeError):
                    pass
            
            # If less than 50% are numeric, likely a header
            return numeric_count / len(first_row) < 0.5
            
        except Exception:
            return True  # Default to assuming header exists
    
    def create_backup(self, file_path: Union[str, Path], 
                     backup_name: Optional[str] = None) -> Path:
        """
        Create backup of a file.
        
        Args:
            file_path: Path to file to backup
            backup_name: Optional custom backup name
            
        Returns:
            Path to backup file
        """
        try:
            source_path = self.validate_file_path(file_path)
            
            # Generate backup name
            if backup_name:
                backup_filename = backup_name
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            
            backup_path = self.backup_dir / backup_filename
            
            # Copy file to backup location
            shutil.copy2(source_path, backup_path)
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            raise FileManagerError(f"Backup creation failed: {e}")
    
    def create_temp_file(self, suffix: str = '.csv', prefix: str = 'spectral_') -> Path:
        """
        Create temporary file.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=suffix,
                prefix=prefix,
                dir=self.temp_dir
            )
            
            # Close file descriptor (we just need the path)
            import os
            os.close(temp_fd)
            
            temp_path = Path(temp_path)
            self.logger.debug(f"Created temporary file: {temp_path}")
            
            return temp_path
            
        except Exception as e:
            raise FileManagerError(f"Temporary file creation failed: {e}")
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age in hours for temp files
            
        Returns:
            Number of files cleaned up
        """
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for temp_file in self.temp_dir.glob("*"):
                try:
                    if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                        temp_file.unlink()
                        cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} temporary files")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Temp file cleanup failed: {e}")
            return 0
    
    def safe_write_csv(self, data: List[List[str]], output_path: Union[str, Path],
                      headers: Optional[List[str]] = None, 
                      delimiter: str = ',') -> bool:
        """
        Safely write CSV data to file.
        
        Args:
            data: CSV data as list of rows
            output_path: Output file path
            headers: Optional column headers
            delimiter: CSV delimiter
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            
            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_path = self.create_temp_file(suffix=output_path.suffix)
            
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=delimiter)
                
                # Write headers if provided
                if headers:
                    writer.writerow(headers)
                
                # Write data
                writer.writerows(data)
            
            # Move temporary file to final location
            shutil.move(temp_path, output_path)
            
            self.logger.info(f"CSV file written successfully: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write CSV file: {e}")
            raise FileManagerError(f"CSV write failed: {e}")
    
    def safe_copy_file(self, source: Union[str, Path], 
                      destination: Union[str, Path],
                      create_backup: bool = True) -> bool:
        """
        Safely copy file with optional backup.
        
        Args:
            source: Source file path
            destination: Destination file path
            create_backup: Create backup if destination exists
            
        Returns:
            True if successful
        """
        try:
            source_path = self.validate_file_path(source)
            dest_path = Path(destination)
            
            # Create destination directory if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if destination exists
            if dest_path.exists() and create_backup:
                self.create_backup(dest_path)
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            self.logger.info(f"File copied: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"File copy failed: {e}")
            raise FileManagerError(f"File copy failed: {e}")
    
    def batch_validate_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Validate multiple files in batch.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid_files': [],
            'invalid_files': [],
            'total_size_mb': 0,
            'csv_files': 0,
            'issues': []
        }
        
        for file_path in file_paths:
            try:
                validated_path = self.validate_file_path(file_path)
                file_info = self.get_file_info(validated_path)
                
                results['valid_files'].append({
                    'path': str(validated_path),
                    'info': file_info
                })
                
                results['total_size_mb'] += file_info['size_mb']
                
                if file_info['is_csv']:
                    results['csv_files'] += 1
                
            except FileManagerError as e:
                results['invalid_files'].append({
                    'path': str(file_path),
                    'error': str(e)
                })
                results['issues'].append(f"{file_path}: {e}")
            
            except Exception as e:
                results['invalid_files'].append({
                    'path': str(file_path),
                    'error': f"Unexpected error: {e}"
                })
                results['issues'].append(f"{file_path}: Unexpected error: {e}")
        
        return results
    
    def get_directory_info(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a directory and its contents.
        
        Args:
            directory: Directory path
            
        Returns:
            Dictionary with directory information
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                raise FileManagerError(f"Directory does not exist: {dir_path}")
            
            if not dir_path.is_dir():
                raise FileManagerError(f"Path is not a directory: {dir_path}")
            
            # Analyze directory contents
            total_files = 0
            total_size = 0
            csv_files = 0
            supported_files = []
            
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    if file_path.suffix.lower() in self.supported_extensions:
                        supported_files.append(str(file_path))
                        
                        if file_path.suffix.lower() in self.csv_extensions:
                            csv_files += 1
            
            return {
                'path': str(dir_path.absolute()),
                'total_files': total_files,
                'total_size_mb': total_size / (1024 * 1024),
                'csv_files': csv_files,
                'supported_files': len(supported_files),
                'supported_file_paths': supported_files[:100]  # Limit to first 100
            }
            
        except Exception as e:
            raise FileManagerError(f"Directory analysis failed: {e}")
    
    def export_file(self, source_data: Any, filename: str, 
                   format_type: str = 'csv') -> Path:
        """
        Export data to file in specified format.
        
        Args:
            source_data: Data to export
            filename: Output filename
            format_type: Export format (csv, json, txt)
            
        Returns:
            Path to exported file
        """
        try:
            export_path = self.export_dir / filename
            
            if format_type.lower() == 'csv':
                # Handle CSV export
                if hasattr(source_data, 'to_csv'):  # pandas DataFrame
                    source_data.to_csv(export_path, index=False)
                else:
                    raise FileManagerError("Unsupported data type for CSV export")
            
            elif format_type.lower() == 'json':
                # Handle JSON export
                with open(export_path, 'w', encoding='utf-8') as f:
                    if hasattr(source_data, 'to_json'):  # pandas DataFrame
                        f.write(source_data.to_json(indent=2))
                    else:
                        json.dump(source_data, f, indent=2, ensure_ascii=False)
            
            elif format_type.lower() == 'txt':
                # Handle text export
                with open(export_path, 'w', encoding='utf-8') as f:
                    if hasattr(source_data, 'to_string'):  # pandas DataFrame
                        f.write(source_data.to_string())
                    else:
                        f.write(str(source_data))
            
            else:
                raise FileManagerError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Data exported to: {export_path}")
            return export_path
            
        except Exception as e:
            raise FileManagerError(f"Export failed: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            stats = {}
            
            for name, directory in [
                ('base', self.base_dir),
                ('temp', self.temp_dir),
                ('backup', self.backup_dir),
                ('export', self.export_dir)
            ]:
                if directory.exists():
                    total_size = sum(
                        f.stat().st_size for f in directory.rglob("*") if f.is_file()
                    )
                    file_count = sum(1 for f in directory.rglob("*") if f.is_file())
                    
                    stats[f'{name}_dir'] = {
                        'path': str(directory),
                        'size_mb': total_size / (1024 * 1024),
                        'file_count': file_count
                    }
                else:
                    stats[f'{name}_dir'] = {
                        'path': str(directory),
                        'size_mb': 0,
                        'file_count': 0
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    async def process_batch_files(self, baseline_file: str, sample_files: List[str],
                                output_settings: OutputSettings,
                                progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
                                config: Optional[BatchConfig] = None) -> BatchProcessResult:
        """
        Process multiple files in batch with comprehensive progress tracking.
        
        Args:
            baseline_file: Path to baseline CSV file
            sample_files: List of sample CSV file paths
            output_settings: Output configuration
            progress_callback: Optional callback for progress updates
            config: Optional batch configuration
            
        Returns:
            BatchProcessResult with detailed processing statistics
        """
        if config is None:
            config = BatchConfig()
        
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
        
        try:
            self.logger.info(f"Starting batch processing: {len(sample_files)} files")
            
            # Phase 1: Validation
            if progress_callback:
                progress.current_operation = "Validating files..."
                progress_callback(progress)
            
            validation_result = await self.validate_batch_files(
                baseline_file, sample_files, progress_callback
            )
            
            if not validation_result.valid_files:
                result.errors.append("No valid files to process")
                return result
            
            result.warnings.extend(validation_result.warnings)
            
            # Phase 2: Processing
            progress.current_stage = ProcessingStage.LOADING
            progress.total_files = len(validation_result.valid_files)
            
            # Create output directory
            output_dir = Path(output_settings.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process files with controlled concurrency
            semaphore = asyncio.Semaphore(config.max_concurrent_files)
            
            async def process_single_file(sample_file: str, index: int) -> Tuple[bool, str, List[str]]:
                async with semaphore:
                    return await self._process_single_file_batch(
                        baseline_file, sample_file, output_settings,
                        progress, progress_callback, config, temp_files, index
                    )
            
            # Execute batch processing
            tasks = [
                process_single_file(sample_file, i)
                for i, sample_file in enumerate(validation_result.valid_files)
            ]
            
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, task_result in enumerate(completed_tasks):
                if isinstance(task_result, Exception):
                    result.failed += 1
                    result.errors.append(f"File {validation_result.valid_files[i]}: {str(task_result)}")
                else:
                    success, output_file, warnings = task_result
                    if success:
                        result.successful += 1
                        if output_file:
                            result.output_files.append(output_file)
                    else:
                        result.failed += 1
                    
                    result.warnings.extend(warnings)
            
            # Phase 3: Cleanup
            progress.current_stage = ProcessingStage.CLEANUP
            if progress_callback:
                progress.current_operation = "Cleaning up temporary files..."
                progress_callback(progress)
            
            if config.cleanup_temp_files:
                await self._cleanup_temp_files_batch(temp_files)
            
            # Final statistics
            result.processing_time = time.time() - start_time
            
            if progress_callback:
                progress.progress_percent = 100.0
                progress.current_operation = "Batch processing complete"
                progress_callback(progress)
            
            self.logger.info(
                f"Batch processing complete: {result.successful}/{result.total_files} successful "
                f"in {result.processing_time:.2f}s"
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
                await self._cleanup_temp_files_batch(temp_files)
            
            result.processing_time = time.time() - start_time
            return result
    
    async def validate_batch_files(self, baseline_file: str, sample_files: List[str],
                                 progress_callback: Optional[Callable[[ProcessingProgress], None]] = None) -> FileValidationResult:
        """
        Validate multiple files for batch processing.
        
        Args:
            baseline_file: Path to baseline file
            sample_files: List of sample file paths
            progress_callback: Optional progress callback
            
        Returns:
            FileValidationResult with validation details
        """
        result = FileValidationResult()
        
        try:
            self.logger.info(f"Validating {len(sample_files)} files for batch processing")
            
            # Validate baseline file
            try:
                baseline_path = self.validate_file_path(baseline_file)
                baseline_info = self.get_file_info(baseline_path)
                
                if not baseline_info['is_csv']:
                    result.invalid_files.append((baseline_file, "Baseline file is not a CSV"))
                    return result
                
            except FileManagerError as e:
                result.invalid_files.append((baseline_file, f"Baseline validation failed: {str(e)}"))
                return result
            
            # Validate sample files
            total_size = 0
            for i, sample_file in enumerate(sample_files):
                try:
                    if progress_callback:
                        progress = ProcessingProgress(
                            current_file=sample_file,
                            files_completed=i,
                            total_files=len(sample_files),
                            current_stage=ProcessingStage.VALIDATION,
                            progress_percent=(i / len(sample_files)) * 100,
                            current_operation=f"Validating {Path(sample_file).name}..."
                        )
                        progress_callback(progress)
                    
                    sample_path = self.validate_file_path(sample_file)
                    sample_info = self.get_file_info(sample_path)
                    
                    if not sample_info['is_csv']:
                        result.invalid_files.append((sample_file, "Not a CSV file"))
                        continue
                    
                    if sample_info.get('issues'):
                        result.warnings.append(f"{sample_file}: {', '.join(sample_info['issues'])}")
                    
                    result.valid_files.append(sample_file)
                    total_size += sample_info['size_mb']
                    
                except FileManagerError as e:
                    result.invalid_files.append((sample_file, str(e)))
            
            result.total_size_mb = total_size
            
            # Estimate processing time (rough estimate: 2-5 seconds per MB)
            result.estimated_time_minutes = (total_size * 3.5) / 60
            
            self.logger.info(
                f"Validation complete: {len(result.valid_files)} valid, "
                f"{len(result.invalid_files)} invalid files"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            result.invalid_files.append(("batch_validation", str(e)))
            return result
    
    def resolve_filename_conflicts(self, filepath: str, strategy: str = "auto_increment") -> str:
        """
        Resolve filename conflicts automatically.
        
        Args:
            filepath: Original file path
            strategy: Conflict resolution strategy
            
        Returns:
            Resolved file path
        """
        try:
            path = Path(filepath)
            
            if not path.exists():
                return filepath
            
            if strategy == "overwrite":
                return filepath
            elif strategy == "skip":
                return None
            elif strategy == "auto_increment":
                counter = 1
                stem = path.stem
                suffix = path.suffix
                parent = path.parent
                
                while path.exists():
                    new_name = f"{stem}_{counter:03d}{suffix}"
                    path = parent / new_name
                    counter += 1
                    
                    # Prevent infinite loop
                    if counter > 999:
                        timestamp = int(time.time())
                        new_name = f"{stem}_{timestamp}{suffix}"
                        path = parent / new_name
                        break
                
                return str(path)
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Filename conflict resolution failed: {e}")
            return filepath
    
    async def cleanup_temp_files_batch(self, temp_files: List[str]) -> int:
        """
        Clean up temporary files from batch processing.
        
        Args:
            temp_files: List of temporary file paths
            
        Returns:
            Number of files cleaned up
        """
        return await self._cleanup_temp_files_batch(temp_files)
    
    async def _process_single_file_batch(self, baseline_file: str, sample_file: str,
                                       output_settings: OutputSettings,
                                       progress: ProcessingProgress,
                                       progress_callback: Optional[Callable[[ProcessingProgress], None]],
                                       config: BatchConfig,
                                       temp_files: List[str],
                                       file_index: int) -> Tuple[bool, str, List[str]]:
        """Process a single file in batch mode."""
        warnings = []
        output_file = None
        
        try:
            # Update progress
            progress.current_file = Path(sample_file).name
            progress.files_completed = file_index
            progress.current_operation = f"Processing {Path(sample_file).name}..."
            
            if progress_callback:
                progress_callback(progress)
            
            # This would integrate with CSV parser, AI normalizer, and graph generator
            # For now, simulate processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Generate output filename
            baseline_name = Path(baseline_file).stem
            sample_name = Path(sample_file).stem
            
            filename = output_settings.filename_template.format(
                sample=sample_name,
                baseline=baseline_name
            )
            
            output_path = Path(output_settings.output_dir) / f"{filename}.{output_settings.format}"
            
            # Resolve conflicts
            resolved_path = self.resolve_filename_conflicts(
                str(output_path),
                output_settings.conflict_resolution
            )
            
            if resolved_path is None:  # Skip strategy
                warnings.append(f"Skipped {sample_file} due to existing output file")
                return False, None, warnings
            
            output_file = resolved_path
            
            # Simulate successful processing
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).touch()  # Create placeholder file
            
            return True, output_file, warnings
            
        except Exception as e:
            warnings.append(f"Processing failed for {sample_file}: {str(e)}")
            return False, output_file, warnings
    
    async def _cleanup_temp_files_batch(self, temp_files: List[str]) -> int:
        """Clean up temporary files."""
        cleaned_count = 0
        
        for temp_file in temp_files:
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    temp_path.unlink()
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove temp file {temp_file}: {e}")
        
        return cleaned_count
    
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