"""
Global error handling and exception management utilities.

Provides comprehensive error handling, exception logging,
and user-friendly error reporting for the application.
"""

import logging
import sys
import traceback
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path
import json

from PyQt6.QtWidgets import QMessageBox, QWidget
from PyQt6.QtCore import QObject, pyqtSignal


class SpectralAnalyzerError(Exception):
    """Base exception class for Spectral Analyzer."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize custom exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional additional details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()


class CSVParsingError(SpectralAnalyzerError):
    """Exception for CSV parsing errors."""
    pass


class AIServiceError(SpectralAnalyzerError):
    """Exception for AI service errors."""
    pass


class DataValidationError(SpectralAnalyzerError):
    """Exception for data validation errors."""
    pass


class ConfigurationError(SpectralAnalyzerError):
    """Exception for configuration errors."""
    pass


class CacheError(SpectralAnalyzerError):
    """Exception for cache-related errors."""
    pass


class SecurityError(SpectralAnalyzerError):
    """Exception for security-related errors."""
    pass


class ErrorReporter(QObject):
    """Error reporting system with user notifications."""
    
    # Signals
    error_reported = pyqtSignal(str, str, dict)  # error_type, message, details
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize error reporter.
        
        Args:
            parent: Parent widget for dialogs
        """
        super().__init__()
        self.parent = parent
        self.logger = logging.getLogger(__name__)
        
        # Error statistics
        self.error_counts = {}
        self.last_errors = []
        self.max_recent_errors = 50
    
    def report_error(self, error: Exception, context: Optional[str] = None,
                    show_dialog: bool = True, critical: bool = False) -> str:
        """
        Report an error with logging and optional user notification.
        
        Args:
            error: Exception to report
            context: Optional context information
            show_dialog: Show error dialog to user
            critical: Whether this is a critical error
            
        Returns:
            Error ID for tracking
        """
        try:
            # Generate error ID
            error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
            
            # Extract error information
            error_type = type(error).__name__
            error_message = str(error)
            
            # Get traceback
            tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
            traceback_str = ''.join(tb_lines)
            
            # Prepare error details
            error_details = {
                'error_id': error_id,
                'error_type': error_type,
                'message': error_message,
                'context': context,
                'traceback': traceback_str,
                'timestamp': datetime.now().isoformat(),
                'critical': critical
            }
            
            # Add custom error details if available
            if isinstance(error, SpectralAnalyzerError):
                error_details.update({
                    'error_code': error.error_code,
                    'custom_details': error.details
                })
            
            # Log the error
            log_level = logging.CRITICAL if critical else logging.ERROR
            self.logger.log(log_level, f"Error {error_id}: {error_message}", 
                          extra={'error_details': error_details})
            
            # Update statistics
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Store recent error
            self.last_errors.append(error_details)
            if len(self.last_errors) > self.max_recent_errors:
                self.last_errors.pop(0)
            
            # Emit signal
            self.error_reported.emit(error_type, error_message, error_details)
            
            # Show user dialog if requested
            if show_dialog:
                self._show_error_dialog(error_details)
            
            return error_id
            
        except Exception as e:
            # Fallback error handling
            self.logger.critical(f"Error in error reporting system: {e}")
            return "ERR_UNKNOWN"
    
    def _show_error_dialog(self, error_details: Dict[str, Any]):
        """Show error dialog to user."""
        try:
            error_type = error_details['error_type']
            message = error_details['message']
            context = error_details.get('context', '')
            critical = error_details.get('critical', False)
            
            # Prepare dialog message
            dialog_message = f"An error occurred: {message}"
            if context:
                dialog_message += f"\n\nContext: {context}"
            
            dialog_message += f"\n\nError ID: {error_details['error_id']}"
            
            # Show appropriate dialog
            if critical:
                QMessageBox.critical(self.parent, f"Critical Error - {error_type}", dialog_message)
            else:
                QMessageBox.warning(self.parent, f"Error - {error_type}", dialog_message)
                
        except Exception as e:
            self.logger.error(f"Failed to show error dialog: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors_count': len(self.last_errors),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        }
    
    def get_recent_errors(self, limit: int = 10) -> list:
        """Get recent errors."""
        return self.last_errors[-limit:] if self.last_errors else []
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_counts.clear()
        self.last_errors.clear()
        self.logger.info("Error history cleared")


class GlobalExceptionHandler:
    """Global exception handler for unhandled exceptions."""
    
    def __init__(self, error_reporter: Optional[ErrorReporter] = None):
        """
        Initialize global exception handler.
        
        Args:
            error_reporter: Optional error reporter instance
        """
        self.logger = logging.getLogger(__name__)
        self.error_reporter = error_reporter
        self.original_excepthook = sys.excepthook
        
        # Error log file
        log_dir = Path.home() / ".spectral_analyzer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.crash_log_file = log_dir / "crashes.log"
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """
        Handle unhandled exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        try:
            # Don't handle KeyboardInterrupt
            if issubclass(exc_type, KeyboardInterrupt):
                self.original_excepthook(exc_type, exc_value, exc_traceback)
                return
            
            # Create crash report
            crash_report = {
                'timestamp': datetime.now().isoformat(),
                'exception_type': exc_type.__name__,
                'exception_message': str(exc_value),
                'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
                'python_version': sys.version,
                'platform': sys.platform
            }
            
            # Log to crash file
            with open(self.crash_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(crash_report, indent=2) + '\n' + '='*80 + '\n')
            
            # Log to main logger
            self.logger.critical(
                f"Unhandled exception: {exc_type.__name__}: {exc_value}",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            
            # Report through error reporter if available
            if self.error_reporter:
                try:
                    exception_obj = exc_value if exc_value else exc_type()
                    self.error_reporter.report_error(
                        exception_obj,
                        context="Unhandled exception",
                        show_dialog=True,
                        critical=True
                    )
                except Exception as e:
                    self.logger.error(f"Error reporter failed: {e}")
            
            # Call original exception hook for default behavior
            self.original_excepthook(exc_type, exc_value, exc_traceback)
            
        except Exception as e:
            # Last resort error handling
            print(f"Critical error in exception handler: {e}", file=sys.stderr)
            self.original_excepthook(exc_type, exc_value, exc_traceback)


def safe_execute(func: Callable, *args, default_return=None, 
                error_reporter: Optional[ErrorReporter] = None, **kwargs):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Default return value on error
        error_reporter: Optional error reporter
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
        
    except Exception as e:
        logger = logging.getLogger(func.__module__ if hasattr(func, '__module__') else __name__)
        logger.error(f"Error in {func.__name__}: {e}")
        
        if error_reporter:
            error_reporter.report_error(e, context=f"Function: {func.__name__}", show_dialog=False)
        
        return default_return


def error_boundary(default_return=None, show_dialog: bool = False):
    """
    Decorator to create an error boundary around a function.
    
    Args:
        default_return: Default return value on error
        show_dialog: Show error dialog on exception
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Error in {func.__name__}: {e}")
                
                if show_dialog:
                    try:
                        QMessageBox.warning(
                            None,
                            "Error",
                            f"An error occurred in {func.__name__}:\n{str(e)}"
                        )
                    except Exception:
                        pass  # Ignore dialog errors
                
                return default_return
        
        return wrapper
    return decorator


def validate_input(validation_func: Callable, error_message: str = "Invalid input"):
    """
    Decorator to validate function input parameters.
    
    Args:
        validation_func: Function to validate inputs
        error_message: Error message for validation failure
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    raise DataValidationError(error_message)
                
                return func(*args, **kwargs)
                
            except DataValidationError:
                raise
            except Exception as e:
                raise DataValidationError(f"Input validation failed: {e}")
        
        return wrapper
    return decorator


def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                  exceptions: tuple = (Exception,)):
    """
    Decorator to retry function execution on specific exceptions.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to retry on
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger = logging.getLogger(func.__module__)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        break
                        
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    raise
            
            # All retries failed
            raise last_exception
        
        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None):
    """
    Decorator to automatically log function errors.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {e}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'args': str(args)[:200],  # Truncate long args
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
        
        return wrapper
    return decorator


def create_error_context(operation: str, **context_data):
    """
    Create error context manager for better error reporting.
    
    Args:
        operation: Description of the operation
        **context_data: Additional context data
        
    Returns:
        Context manager
    """
    class ErrorContext:
        def __init__(self, operation: str, **context_data):
            self.operation = operation
            self.context_data = context_data
            self.logger = logging.getLogger(__name__)
        
        def __enter__(self):
            self.logger.debug(f"Starting operation: {self.operation}")
            return self
        
        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is not None:
                # Add context to exception
                if isinstance(exc_value, SpectralAnalyzerError):
                    exc_value.details.update({
                        'operation': self.operation,
                        **self.context_data
                    })
                
                self.logger.error(
                    f"Operation failed: {self.operation}",
                    exc_info=(exc_type, exc_value, traceback),
                    extra={'context_data': self.context_data}
                )
            else:
                self.logger.debug(f"Operation completed: {self.operation}")
            
            return False  # Don't suppress exceptions
    
    return ErrorContext(operation, **context_data)