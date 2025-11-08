"""
Modern status bar widget for application status and progress monitoring.

Provides real-time status updates, progress indication, system information,
and professional styling with animations and visual feedback.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

from PyQt6.QtWidgets import (
    QStatusBar, QLabel, QProgressBar, QPushButton, QWidget, QHBoxLayout,
    QGraphicsOpacityEffect
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QIcon, QColor

from ui.components.toast_notification import show_success_toast, show_error_toast, show_warning_toast


class StatusType(Enum):
    """Status message types."""
    NORMAL = "normal"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PROCESSING = "processing"


class ModernStatusBar(QStatusBar):
    """
    Modern status bar widget with comprehensive monitoring and professional styling.
    
    Features:
    - Animated status message display with type indicators
    - Progress bar for long operations with smooth animations
    - Real-time system resource monitoring
    - API usage tracking with cost monitoring
    - AI status indicator with connection state
    - Professional Material Design styling
    - Toast notification integration
    """
    
    # Signals
    cancel_requested = pyqtSignal()
    ai_settings_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the modern status bar widget."""
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        
        # Status components
        self.status_label: Optional[QLabel] = None
        self.status_icon: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.cancel_button: Optional[QPushButton] = None
        self.ai_status_label: Optional[QLabel] = None
        self.resource_label: Optional[QLabel] = None
        self.api_usage_label: Optional[QLabel] = None
        self.version_label: Optional[QLabel] = None
        
        # Timers
        self.resource_timer: Optional[QTimer] = None
        self.message_timer: Optional[QTimer] = None
        self.status_animation: Optional[QPropertyAnimation] = None
        
        # Current status
        self.current_status_type = StatusType.NORMAL
        self.ai_connected = False
        self.current_operation = ""
        
        self._setup_ui()
        self._setup_timers()
        self._setup_animations()
        
        self.logger.debug("Modern status bar widget initialized")
    
    def _setup_ui(self):
        """Set up the modern status bar UI components."""
        from ui.themes import theme_manager
        
        # Apply status bar styling
        self.setStyleSheet(theme_manager.create_status_bar_style())
        
        # Status icon and message section
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(8, 0, 8, 0)
        status_layout.setSpacing(8)
        
        # Status icon
        self.status_icon = QLabel("ℹ️")
        self.status_icon.setFixedSize(16, 16)
        status_layout.addWidget(self.status_icon)
        
        # Main status message
        self.status_label = QLabel("Ready")
        self.status_label.setFont(theme_manager.get_font("body_small"))
        status_layout.addWidget(self.status_label)
        
        self.addWidget(status_widget)
        
        # Progress section (initially hidden)
        self.progress_widget = QWidget()
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(8, 2, 8, 2)
        progress_layout.setSpacing(8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(200)
        self.progress_bar.setMaximumHeight(16)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.cancel_button = QPushButton("✕")
        self.cancel_button.setMaximumHeight(20)
        self.cancel_button.setMaximumWidth(20)
        self.cancel_button.setVisible(False)
        self.cancel_button.setStyleSheet(theme_manager.create_button_style("text"))
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        progress_layout.addWidget(self.cancel_button)
        
        self.addWidget(self.progress_widget)
        
        # Permanent widgets (right side)
        self.addPermanentWidget(self._create_separator())
        
        # AI status indicator
        self.ai_status_label = QLabel("🤖 AI: Disconnected")
        self.ai_status_label.setToolTip("AI service connection status")
        self.ai_status_label.setMinimumWidth(120)
        self.ai_status_label.setFont(theme_manager.get_font("label_small"))
        self.ai_status_label.mousePressEvent = lambda e: self.ai_settings_requested.emit()
        self.ai_status_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.addPermanentWidget(self.ai_status_label)
        
        self.addPermanentWidget(self._create_separator())
        
        # API usage indicator
        self.api_usage_label = QLabel("💰 $0.00")
        self.api_usage_label.setToolTip("Current API usage cost")
        self.api_usage_label.setMinimumWidth(80)
        self.api_usage_label.setFont(theme_manager.get_font("label_small"))
        self.addPermanentWidget(self.api_usage_label)
        
        self.addPermanentWidget(self._create_separator())
        
        # Resource monitor
        self.resource_label = QLabel("💻 CPU: 0% | RAM: 0MB")
        self.resource_label.setToolTip("System resource usage")
        self.resource_label.setMinimumWidth(140)
        self.resource_label.setFont(theme_manager.get_font("label_small"))
        self.addPermanentWidget(self.resource_label)
        
        self.addPermanentWidget(self._create_separator())
        
        # Version info
        self.version_label = QLabel("v1.0.0")
        self.version_label.setToolTip("Spectral Analyzer version")
        self.version_label.setFont(theme_manager.get_font("label_small"))
        self.addPermanentWidget(self.version_label)
    
    def _create_separator(self) -> QLabel:
        """Create a visual separator."""
        from ui.themes import theme_manager
        
        separator = QLabel("|")
        separator.setStyleSheet(f"color: {theme_manager.get_color('text_disabled')};")
        separator.setFont(theme_manager.get_font("label_small"))
        return separator
    
    def _setup_animations(self):
        """Set up status animations."""
        # Status message fade animation
        self.status_animation = QPropertyAnimation(self.status_label, b"geometry")
        self.status_animation.setDuration(250)
        self.status_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def _setup_timers(self):
        """Set up update timers."""
        # Resource monitoring timer
        self.resource_timer = QTimer()
        self.resource_timer.timeout.connect(self._update_resource_info)
        self.resource_timer.start(2000)  # Update every 2 seconds
        
        # Message auto-clear timer
        self.message_timer = QTimer()
        self.message_timer.setSingleShot(True)
        self.message_timer.timeout.connect(self._clear_temporary_message)
    
    def show_message(self, message: str, status_type: StatusType = StatusType.NORMAL, timeout: int = 0):
        """
        Show status message with type indicator.
        
        Args:
            message: Message to display
            status_type: Type of status message
            timeout: Auto-clear timeout in milliseconds (0 = permanent)
        """
        from ui.themes import theme_manager
        
        self.current_status_type = status_type
        self.status_label.setText(message)
        
        # Update icon and styling based on type
        if status_type == StatusType.SUCCESS:
            self.status_icon.setText("✅")
            self.status_label.setStyleSheet(f"color: {theme_manager.get_color('success')};")
        elif status_type == StatusType.WARNING:
            self.status_icon.setText("⚠️")
            self.status_label.setStyleSheet(f"color: {theme_manager.get_color('warning')};")
        elif status_type == StatusType.ERROR:
            self.status_icon.setText("❌")
            self.status_label.setStyleSheet(f"color: {theme_manager.get_color('error')};")
        elif status_type == StatusType.PROCESSING:
            self.status_icon.setText("⚙️")
            self.status_label.setStyleSheet(f"color: {theme_manager.get_color('info')};")
        else:
            self.status_icon.setText("ℹ️")
            self.status_label.setStyleSheet(f"color: {theme_manager.get_color('text_primary')};")
        
        if timeout > 0:
            self.message_timer.start(timeout)
        else:
            self.message_timer.stop()
        
        self.logger.debug(f"Status message ({status_type.value}): {message}")
    
    def show_temporary_message(self, message: str, status_type: StatusType = StatusType.NORMAL, timeout: int = 3000):
        """
        Show temporary status message that auto-clears.
        
        Args:
            message: Message to display
            status_type: Type of status message
            timeout: Auto-clear timeout in milliseconds
        """
        self.show_message(message, status_type, timeout)
    
    def show_success(self, message: str, timeout: int = 3000):
        """Show success status message."""
        self.show_temporary_message(message, StatusType.SUCCESS, timeout)
    
    def show_warning(self, message: str, timeout: int = 4000):
        """Show warning status message."""
        self.show_temporary_message(message, StatusType.WARNING, timeout)
    
    def show_error(self, message: str, timeout: int = 5000):
        """Show error status message."""
        self.show_temporary_message(message, StatusType.ERROR, timeout)
    
    def show_processing(self, message: str):
        """Show processing status message."""
        self.show_message(message, StatusType.PROCESSING)
    
    def _clear_temporary_message(self):
        """Clear temporary message and restore default."""
        self.show_message("Ready", StatusType.NORMAL)
    
    def show_progress(self, minimum: int = 0, maximum: int = 0, value: int = 0):
        """
        Show progress bar.
        
        Args:
            minimum: Minimum progress value
            maximum: Maximum progress value (0 = indeterminate)
            value: Current progress value
        """
        self.progress_bar.setMinimum(minimum)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        
        if maximum == 0:
            # Indeterminate progress
            self.progress_bar.setRange(0, 0)
    
    def update_progress(self, value: int, message: Optional[str] = None):
        """
        Update progress value and optional message.
        
        Args:
            value: New progress value
            message: Optional status message
        """
        self.progress_bar.setValue(value)
        
        if message:
            self.show_message(message)
    
    def hide_progress(self):
        """Hide progress bar and cancel button."""
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
    
    def set_progress_text(self, text: str):
        """Set custom progress text."""
        self.progress_bar.setFormat(text)
    
    def reset_progress_text(self):
        """Reset progress text to default percentage."""
        self.progress_bar.setFormat("%p%")
    
    def update_api_usage(self, cost: float, requests: int = 0):
        """
        Update API usage information.
        
        Args:
            cost: Current API cost in USD
            requests: Number of API requests made
        """
        if requests > 0:
            tooltip = f"API Cost: ${cost:.4f}\nRequests: {requests:,}"
            text = f"API: ${cost:.2f} ({requests})"
        else:
            tooltip = f"API Cost: ${cost:.4f}"
            text = f"API: ${cost:.2f}"
        
        self.api_usage_label.setText(text)
        self.api_usage_label.setToolTip(tooltip)
    
    def _update_resource_info(self):
        """Update system resource information with enhanced monitoring."""
        try:
            import psutil
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used // (1024 * 1024)
            memory_percent = memory.percent
            
            # Update display with color coding
            resource_text = f"💻 CPU: {cpu_percent:.0f}% | RAM: {memory_mb:,}MB"
            self.resource_label.setText(resource_text)
            
            # Color code based on usage
            from ui.themes import theme_manager
            if cpu_percent > 80 or memory_percent > 85:
                color = theme_manager.get_color("error")
            elif cpu_percent > 60 or memory_percent > 70:
                color = theme_manager.get_color("warning")
            else:
                color = theme_manager.get_color("text_secondary")
            
            self.resource_label.setStyleSheet(f"color: {color};")
            
            # Enhanced tooltip with performance metrics
            tooltip_lines = [
                f"🖥️ System Performance:",
                f"  CPU Usage: {cpu_percent:.1f}%",
                f"  Memory Used: {memory_mb:,} MB ({memory_percent:.1f}%)",
                f"  Memory Available: {memory.available // (1024 * 1024):,} MB",
                f"  Total Memory: {memory.total // (1024 * 1024):,} MB",
                ""
            ]
            
            # Add disk usage if available
            try:
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                disk_free_gb = disk.free // (1024 * 1024 * 1024)
                tooltip_lines.extend([
                    f"💾 Storage:",
                    f"  Disk Usage: {disk_percent:.1f}%",
                    f"  Free Space: {disk_free_gb:.1f} GB"
                ])
            except:
                pass
            
            # Add process-specific info
            try:
                process = psutil.Process()
                process_memory = process.memory_info().rss // (1024 * 1024)
                tooltip_lines.extend([
                    "",
                    f"📊 Application:",
                    f"  Memory Usage: {process_memory} MB",
                    f"  CPU Usage: {process.cpu_percent():.1f}%"
                ])
            except:
                pass
            
            self.resource_label.setToolTip("\n".join(tooltip_lines))
            
        except ImportError:
            # psutil not available
            self.resource_label.setText("💻 Resource monitoring unavailable")
            self.resource_label.setToolTip("Install psutil for resource monitoring")
            self.resource_label.setStyleSheet(f"color: {theme_manager.get_color('text_disabled')};")
        except Exception as e:
            self.logger.warning(f"Failed to update resource info: {e}")
    
    def set_ai_status(self, connected: bool, model_name: str = "", cost: float = 0.0):
        """
        Set AI service status with enhanced information.
        
        Args:
            connected: Whether AI service is connected
            model_name: Name of the current AI model
            cost: Current session cost
        """
        from ui.themes import theme_manager
        
        self.ai_connected = connected
        
        if connected:
            status_text = f"🤖 AI: Connected"
            if model_name:
                status_text += f" ({model_name})"
            color = theme_manager.get_color("success")
            tooltip = f"AI Service: Connected\nModel: {model_name}\nSession Cost: ${cost:.4f}\n\nClick to configure AI settings"
        else:
            status_text = "🤖 AI: Disconnected"
            color = theme_manager.get_color("error")
            tooltip = "AI Service: Disconnected\n\nClick to configure AI settings"
        
        self.ai_status_label.setText(status_text)
        self.ai_status_label.setStyleSheet(f"color: {color};")
        self.ai_status_label.setToolTip(tooltip)
    
    def update_api_usage(self, cost: float, requests: int = 0, tokens: int = 0):
        """
        Update API usage information with enhanced tracking.
        
        Args:
            cost: Current API cost in USD
            requests: Number of API requests made
            tokens: Total tokens used
        """
        from ui.themes import theme_manager
        
        # Format cost display
        if cost < 0.01:
            cost_text = f"💰 ${cost:.4f}"
        else:
            cost_text = f"💰 ${cost:.2f}"
        
        self.api_usage_label.setText(cost_text)
        
        # Enhanced tooltip
        tooltip_lines = [
            f"💰 API Usage:",
            f"  Total Cost: ${cost:.4f}",
            f"  Requests: {requests:,}",
            f"  Tokens: {tokens:,}"
        ]
        
        if requests > 0:
            avg_cost_per_request = cost / requests
            tooltip_lines.append(f"  Avg Cost/Request: ${avg_cost_per_request:.4f}")
        
        self.api_usage_label.setToolTip("\n".join(tooltip_lines))
        
        # Color code based on cost
        if cost > 10.0:
            color = theme_manager.get_color("error")
        elif cost > 5.0:
            color = theme_manager.get_color("warning")
        else:
            color = theme_manager.get_color("text_secondary")
        
        self.api_usage_label.setStyleSheet(f"color: {color};")
    
    def set_connection_status(self, connected: bool, service: str = ""):
        """
        Set connection status indicator.
        
        Args:
            connected: Whether service is connected
            service: Name of the service
        """
        if connected:
            status_text = f"Connected to {service}" if service else "Connected"
            self.show_message(status_text)
            self.show_success_indicator()
        else:
            status_text = f"Disconnected from {service}" if service else "Disconnected"
            self.show_message(status_text)
            self.show_error_indicator()
    
    def show_processing_status(self, operation: str, progress: Optional[int] = None):
        """
        Show processing status with optional progress.
        
        Args:
            operation: Description of current operation
            progress: Progress percentage (0-100)
        """
        if progress is not None:
            message = f"{operation}... {progress}%"
            if not self.progress_bar.isVisible():
                self.show_progress(0, 100, progress)
            else:
                self.update_progress(progress)
        else:
            message = f"{operation}..."
            if not self.progress_bar.isVisible():
                self.show_progress()  # Indeterminate
        
        self.show_message(message)
    
    def show_file_status(self, filename: str, status: str):
        """
        Show file processing status.
        
        Args:
            filename: Name of file being processed
            status: Processing status
        """
        message = f"{status}: {filename}"
        self.show_message(message)
    
    def show_batch_status(self, current: int, total: int, operation: str = "Processing"):
        """
        Show batch processing status.
        
        Args:
            current: Current item number
            total: Total number of items
            operation: Operation description
        """
        progress = int((current / total) * 100) if total > 0 else 0
        message = f"{operation} {current}/{total} files"
        
        if not self.progress_bar.isVisible():
            self.show_progress(0, 100, progress)
        else:
            self.update_progress(progress, message)
    
    def set_operation_status(self, operation: str, progress: Optional[int] = None):
        """
        Set current operation status.
        
        Args:
            operation: Description of current operation
            progress: Progress percentage (0-100)
        """
        self.current_operation = operation
        
        if progress is not None:
            message = f"{operation}... {progress}%"
            if not self.progress_bar.isVisible():
                self.show_progress(0, 100, progress)
            else:
                self.update_progress(progress)
        else:
            message = f"{operation}..."
            if not self.progress_bar.isVisible():
                self.show_progress()  # Indeterminate
        
        self.show_processing(message)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.resource_timer:
                self.resource_timer.stop()
            if self.message_timer:
                self.message_timer.stop()
            if self.status_animation:
                self.status_animation.stop()
            
            self.logger.debug("Modern status bar cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during status bar cleanup: {e}")


# Maintain backward compatibility
StatusBarWidget = ModernStatusBar