"""
Logging configuration and utilities for Spectral Analyzer.

Provides comprehensive logging setup with file rotation, 
different log levels, and structured logging for debugging.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format message
        message = record.getMessage()
        
        # Create colored log entry
        log_line = f"{color}[{timestamp}] {record.levelname:8} {record.name:20} | {message}{reset}"
        
        # Add exception information if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"
        
        return log_line


def setup_logging(log_level: str = "INFO", 
                 log_dir: Optional[Path] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_structured: bool = False,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> Dict[str, Any]:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ~/.spectral_analyzer/logs)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Enable structured JSON logging
        max_file_size: Maximum size per log file in bytes
        backup_count: Number of backup log files to keep
        
    Returns:
        Dictionary with logging configuration info
    """
    # Set up log directory
    if log_dir is None:
        log_dir = Path.home() / ".spectral_analyzer" / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    handlers_info = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if enable_structured:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = ColoredConsoleFormatter()
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        handlers_info.append({
            'type': 'console',
            'level': log_level,
            'formatter': 'structured' if enable_structured else 'colored'
        })
    
    # File handlers
    if enable_file:
        # Main application log
        app_log_file = log_dir / "spectral_analyzer.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(numeric_level)
        
        if enable_structured:
            app_formatter = StructuredFormatter()
        else:
            app_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        app_handler.setFormatter(app_formatter)
        root_logger.addHandler(app_handler)
        
        handlers_info.append({
            'type': 'file',
            'file': str(app_log_file),
            'level': log_level,
            'max_size_mb': max_file_size / (1024 * 1024),
            'backup_count': backup_count,
            'formatter': 'structured' if enable_structured else 'standard'
        })
        
        # Error log (ERROR and CRITICAL only)
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s\n'
            '%(pathname)s:%(lineno)d in %(funcName)s\n'
            '%(exc_info)s\n' + '-' * 80,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
        
        handlers_info.append({
            'type': 'error_file',
            'file': str(error_log_file),
            'level': 'ERROR',
            'max_size_mb': max_file_size / (1024 * 1024),
            'backup_count': backup_count
        })
        
        # Debug log (DEBUG level only, if enabled)
        if numeric_level <= logging.DEBUG:
            debug_log_file = log_dir / "debug.log"
            debug_handler = logging.handlers.RotatingFileHandler(
                debug_log_file,
                maxBytes=max_file_size * 2,  # Larger for debug logs
                backupCount=backup_count,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
            
            debug_formatter = logging.Formatter(
                '%(asctime)s | %(name)-30s | %(funcName)-20s:%(lineno)-4d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S.%f'
            )
            
            debug_handler.setFormatter(debug_formatter)
            root_logger.addHandler(debug_handler)
            
            handlers_info.append({
                'type': 'debug_file',
                'file': str(debug_log_file),
                'level': 'DEBUG',
                'max_size_mb': (max_file_size * 2) / (1024 * 1024),
                'backup_count': backup_count
            })
    
    # Set specific logger levels for third-party libraries
    _configure_third_party_loggers()
    
    # Log the logging setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, handlers={len(handlers_info)}")
    
    return {
        'log_level': log_level,
        'log_dir': str(log_dir),
        'handlers': handlers_info,
        'structured_logging': enable_structured
    }


def _configure_third_party_loggers():
    """Configure logging levels for third-party libraries."""
    # Reduce verbosity of third-party libraries
    third_party_loggers = {
        'httpx': logging.WARNING,
        'httpcore': logging.WARNING,
        'urllib3': logging.WARNING,
        'requests': logging.WARNING,
        'matplotlib': logging.WARNING,
        'PIL': logging.WARNING,
        'asyncio': logging.WARNING,
        'redis': logging.WARNING
    }
    
    for logger_name, level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context information."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        """
        Initialize logger adapter.
        
        Args:
            logger: Base logger
            extra: Extra context information
        """
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message with extra context."""
        # Add extra fields to the log record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        kwargs['extra']['extra_fields'] = kwargs['extra']
        
        return msg, kwargs


def get_logger(name: str, **context) -> logging.Logger:
    """
    Get logger with optional context information.
    
    Args:
        name: Logger name
        **context: Additional context information
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        return LoggerAdapter(logger, context)
    
    return logger


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed {func.__name__} in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log exception
            logger.error(f"Exception in {func.__name__} after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls with parameters and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function entry
        logger.debug(f"Entering async {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.debug(f"Completed async {func.__name__} in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log exception
            logger.error(f"Exception in async {func.__name__} after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def setup_performance_logging(enable: bool = True):
    """
    Set up performance logging for monitoring application performance.
    
    Args:
        enable: Enable performance logging
    """
    if not enable:
        return
    
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    
    # Create performance log file
    log_dir = Path.home() / ".spectral_analyzer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    perf_log_file = log_dir / "performance.log"
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    
    perf_formatter = StructuredFormatter()
    perf_handler.setFormatter(perf_formatter)
    perf_logger.addHandler(perf_handler)
    
    perf_logger.info("Performance logging enabled")


def log_memory_usage(operation: str = ""):
    """
    Log current memory usage.
    
    Args:
        operation: Description of current operation
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        logger = logging.getLogger('performance')
        logger.info("Memory usage", extra={
            'operation': operation,
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        })
        
    except ImportError:
        pass  # psutil not available
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to log memory usage: {e}")


def get_log_files_info(log_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about log files.
    
    Args:
        log_dir: Log directory path
        
    Returns:
        Dictionary with log files information
    """
    if log_dir is None:
        log_dir = Path.home() / ".spectral_analyzer" / "logs"
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return {'log_dir': str(log_dir), 'files': []}
    
    log_files = []
    total_size = 0
    
    for log_file in log_dir.glob("*.log*"):
        try:
            stat = log_file.stat()
            log_files.append({
                'name': log_file.name,
                'path': str(log_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
            total_size += stat.st_size
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to get info for {log_file}: {e}")
    
    return {
        'log_dir': str(log_dir),
        'total_files': len(log_files),
        'total_size_mb': total_size / (1024 * 1024),
        'files': sorted(log_files, key=lambda x: x['modified'], reverse=True)
    }


def cleanup_old_logs(log_dir: Optional[Path] = None, days_to_keep: int = 30) -> int:
    """
    Clean up old log files.
    
    Args:
        log_dir: Log directory path
        days_to_keep: Number of days to keep logs
        
    Returns:
        Number of files cleaned up
    """
    if log_dir is None:
        log_dir = Path.home() / ".spectral_analyzer" / "logs"
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return 0
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    cleaned_count = 0
    
    for log_file in log_dir.glob("*.log.*"):  # Rotated log files
        try:
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                cleaned_count += 1
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to clean up {log_file}: {e}")
    
    if cleaned_count > 0:
        logging.getLogger(__name__).info(f"Cleaned up {cleaned_count} old log files")
    
    return cleaned_count