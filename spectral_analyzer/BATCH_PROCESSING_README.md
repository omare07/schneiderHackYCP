# Batch Processing System for Spectral Analyzer

## Overview

The Spectral Analyzer now includes a comprehensive batch processing system that enables efficient processing of multiple spectral files with baseline comparison. This system integrates seamlessly with the existing CSV parser, AI normalizer, and graph generator components.

## Key Features

### 🚀 **Core Capabilities**
- **Batch File Processing**: Process multiple sample files against a single baseline
- **AI Integration**: Automatic normalization using AI when files don't match standard format
- **Progress Tracking**: Real-time progress updates with detailed status information
- **Memory Management**: Efficient processing with controlled memory usage
- **Error Handling**: Robust error handling with recovery mechanisms
- **Caching**: Smart caching to reduce AI API costs and improve performance

### 📊 **File Management**
- **Validation**: Comprehensive file validation before processing
- **Conflict Resolution**: Automatic filename conflict resolution
- **Format Support**: Support for multiple output formats (PNG, JPG, PDF)
- **Cleanup**: Automatic cleanup of temporary files
- **Statistics**: Detailed processing statistics and performance metrics

### 🔧 **Advanced Features**
- **Concurrent Processing**: Configurable concurrent file processing
- **Retry Logic**: Automatic retry for failed operations
- **Cancellation**: Support for cancelling long-running operations
- **Time Estimation**: Processing time and resource estimation
- **Quality Control**: Output validation and quality checks

## Architecture

### Core Components

1. **BatchProcessor** (`utils/batch_processor.py`)
   - Main coordinator for batch processing operations
   - Integrates CSV parser, AI normalizer, and graph generator
   - Manages processing workflow and error handling

2. **FileManager** (`utils/file_manager.py`)
   - Enhanced with batch processing capabilities
   - File validation and conflict resolution
   - Temporary file management and cleanup

3. **Data Classes**
   - `OutputSettings`: Configuration for output generation
   - `BatchProcessResult`: Comprehensive processing results
   - `ProcessingProgress`: Real-time progress tracking
   - `BatchConfig`: Processing configuration options

### Processing Workflow

```
1. File Validation Phase
   ├── Validate baseline file
   ├── Validate all sample files
   ├── Check file permissions and sizes
   └── Estimate processing time and resources

2. Data Loading Phase
   ├── Parse baseline file
   ├── Apply AI normalization if needed
   └── Cache normalized baseline data

3. Batch Processing Phase
   ├── Process sample files concurrently
   ├── Apply AI normalization per file
   ├── Generate comparison graphs
   ├── Save outputs with conflict resolution
   └── Track progress and handle errors

4. Cleanup Phase
   ├── Remove temporary files
   ├── Generate processing statistics
   └── Open output folder (optional)
```

## Usage Examples

### Basic Batch Processing

```python
import asyncio
from utils.batch_processor import BatchProcessor
from utils.file_manager import OutputSettings, BatchConfig

async def basic_example():
    # Initialize batch processor
    batch_processor = BatchProcessor()
    
    # Configure output settings
    output_settings = OutputSettings(
        output_dir="./batch_output",
        format='png',
        dpi=300,
        open_folder_after=True,
        save_normalized_csvs=True
    )
    
    # Configure batch processing
    config = BatchConfig(
        max_concurrent_files=3,
        memory_limit_mb=1000,
        continue_on_error=True
    )
    
    # Define progress callback
    def progress_callback(progress):
        print(f"Progress: {progress.progress_percent:.1f}% - {progress.current_operation}")
    
    # Execute batch processing
    result = await batch_processor.process_spectral_batch(
        baseline_file="baseline.csv",
        sample_files=["sample1.csv", "sample2.csv", "sample3.csv"],
        output_settings=output_settings,
        progress_callback=progress_callback,
        config=config
    )
    
    print(f"Processed {result.successful}/{result.total_files} files successfully")

# Run the example
asyncio.run(basic_example())
```

### Advanced Configuration

```python
# High-quality output for publications
output_settings = OutputSettings(
    output_dir="./publication_graphs",
    format='png',
    dpi=600,  # High DPI
    filename_template="{sample}_vs_{baseline}_publication",
    conflict_resolution="auto_increment"
)

# Memory-efficient processing for large files
config = BatchConfig(
    max_concurrent_files=1,  # Process one at a time
    memory_limit_mb=500,
    timeout_per_file_minutes=15,
    retry_failed_files=True,
    max_retries=3
)
```

### File Validation

```python
# Validate files before processing
validation_result = await batch_processor.file_manager.validate_batch_files(
    baseline_file="baseline.csv",
    sample_files=sample_files
)

print(f"Valid files: {len(validation_result.valid_files)}")
print(f"Invalid files: {len(validation_result.invalid_files)}")
print(f"Estimated processing time: {validation_result.estimated_time_minutes:.1f} minutes")
```

## Configuration Options

### OutputSettings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | Required | Directory for output files |
| `format` | str | 'png' | Output format (png, jpg, pdf) |
| `dpi` | int | 300 | Resolution for raster formats |
| `open_folder_after` | bool | True | Open output folder when complete |
| `save_normalized_csvs` | bool | False | Save normalized CSV files |
| `filename_template` | str | "{sample}_vs_{baseline}" | Filename template |
| `conflict_resolution` | str | "auto_increment" | Conflict resolution strategy |

### BatchConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_files` | int | 3 | Maximum concurrent file processing |
| `memory_limit_mb` | int | 1000 | Memory usage limit in MB |
| `timeout_per_file_minutes` | int | 10 | Timeout per file in minutes |
| `retry_failed_files` | bool | True | Retry failed files |
| `max_retries` | int | 2 | Maximum retry attempts |
| `continue_on_error` | bool | True | Continue processing on errors |
| `cleanup_temp_files` | bool | True | Clean up temporary files |

## Progress Tracking

The system provides detailed progress tracking through callbacks:

```python
def progress_callback(progress: ProcessingProgress):
    print(f"Stage: {progress.current_stage.value}")
    print(f"Progress: {progress.progress_percent:.1f}%")
    print(f"Current file: {progress.current_file}")
    print(f"Operation: {progress.current_operation}")
    print(f"Files completed: {progress.files_completed}/{progress.total_files}")
    print(f"Errors: {progress.errors_count}")
```

### Processing Stages

1. **VALIDATION** - File validation and preparation
2. **LOADING** - Data loading and baseline processing
3. **NORMALIZATION** - AI normalization (if needed)
4. **GRAPH_GENERATION** - Graph creation and comparison
5. **SAVING** - Output file saving
6. **CLEANUP** - Temporary file cleanup

## Error Handling

### Error Recovery Strategies

1. **Continue on Error**: Process remaining files even if some fail
2. **Retry Logic**: Automatically retry failed operations
3. **Fallback Processing**: Use heuristic normalization if AI fails
4. **Graceful Degradation**: Partial results when possible

### Common Error Scenarios

- **File Not Found**: Invalid file paths
- **Format Issues**: Unsupported or corrupted file formats
- **Memory Limits**: Insufficient memory for large files
- **AI Service Errors**: Network or API issues
- **Disk Space**: Insufficient storage for outputs

## Performance Optimization

### Memory Management

- **Controlled Concurrency**: Limit simultaneous file processing
- **Memory Monitoring**: Track and limit memory usage
- **Efficient Data Structures**: Optimize data handling
- **Garbage Collection**: Proper cleanup of resources

### Processing Optimization

- **Caching**: Cache AI normalization results
- **Parallel Processing**: Concurrent file operations
- **Lazy Loading**: Load data only when needed
- **Batch Operations**: Group similar operations

## Integration with Existing Components

### CSV Parser Integration

```python
# The batch processor automatically uses the CSV parser
parse_result = self.csv_parser.parse_file(sample_file)
if parse_result.success:
    sample_data = parse_result.data
```

### AI Normalizer Integration

```python
# AI normalization is applied automatically when needed
if parse_result.structure.confidence < 0.8:
    normalization_result = await self.ai_normalizer.normalize_csv(
        sample_data, sample_file
    )
```

### Graph Generator Integration

```python
# Comparison graphs are generated automatically
fig = self.graph_generator.generate_comparison_graph(
    baseline_data, sample_data, baseline_name, sample_name
)
```

## Testing

### Running Tests

```bash
# Run comprehensive batch processing tests
python test_batch_processing.py

# Run usage examples
python batch_processing_example.py
```

### Test Coverage

- File validation and error handling
- Batch processing workflow
- Progress tracking and callbacks
- Memory monitoring and limits
- Filename conflict resolution
- Error recovery mechanisms

## Best Practices

### File Organization

1. **Consistent Naming**: Use consistent file naming conventions
2. **Directory Structure**: Organize files in logical directories
3. **Backup Strategy**: Keep backups of original files
4. **Output Management**: Use descriptive output directory names

### Performance Tips

1. **Batch Size**: Process 10-50 files per batch for optimal performance
2. **Memory Limits**: Set appropriate memory limits based on file sizes
3. **Concurrent Processing**: Use 2-4 concurrent files for most systems
4. **Caching**: Enable caching to reduce AI API costs

### Error Prevention

1. **File Validation**: Always validate files before processing
2. **Disk Space**: Ensure sufficient disk space for outputs
3. **Network Stability**: Ensure stable network for AI operations
4. **Resource Monitoring**: Monitor system resources during processing

## Troubleshooting

### Common Issues

**Issue**: "Memory limit exceeded"
**Solution**: Reduce `max_concurrent_files` or increase `memory_limit_mb`

**Issue**: "AI normalization failed"
**Solution**: Check network connection and API key configuration

**Issue**: "File validation failed"
**Solution**: Verify file formats and permissions

**Issue**: "Output directory not writable"
**Solution**: Check directory permissions and disk space

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### BatchProcessor Methods

- `process_spectral_batch()` - Main batch processing method
- `estimate_processing_time()` - Estimate processing requirements
- `cancel_processing()` - Cancel ongoing processing
- `get_processing_stats()` - Get current processing statistics

### FileManager Methods

- `validate_batch_files()` - Validate multiple files
- `resolve_filename_conflicts()` - Resolve filename conflicts
- `cleanup_temp_files_batch()` - Clean up temporary files

## Future Enhancements

### Planned Features

- **Distributed Processing**: Support for processing across multiple machines
- **Cloud Integration**: Direct integration with cloud storage services
- **Advanced Analytics**: Statistical analysis of batch results
- **Custom Workflows**: User-defined processing workflows
- **Real-time Monitoring**: Web-based monitoring dashboard

### Performance Improvements

- **GPU Acceleration**: GPU-based processing for large datasets
- **Streaming Processing**: Process files as they arrive
- **Incremental Processing**: Process only changed files
- **Compression**: Automatic compression of output files

## Support

For issues, questions, or feature requests related to batch processing:

1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Enable debug logging for detailed error information
4. Consult the main project documentation

---

*This batch processing system significantly enhances the Spectral Analyzer's capabilities, enabling efficient processing of large datasets while maintaining the quality and accuracy of individual file analysis.*