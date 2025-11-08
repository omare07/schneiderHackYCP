"""
Toast notification system for user feedback.

Provides non-intrusive notifications with smooth animations,
different severity levels, and automatic dismissal.
"""

import logging
from typing import Optional, List
from enum import Enum
from PyQt6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton,
    QGraphicsOpacityEffect, QApplication
)
from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect,
    pyqtSignal, QPoint
)
from PyQt6.QtGui import QFont, QIcon, QPainter, QPainterPath, QColor


class ToastType(Enum):
    """Toast notification types."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ToastNotification(QWidget):
    """
    Individual toast notification widget.
    
    Features:
    - Smooth slide-in/fade-out animations
    - Auto-dismiss with configurable timeout
    - Click to dismiss
    - Professional styling with icons
    """
    
    dismissed = pyqtSignal()
    
    def __init__(self, message: str, toast_type: ToastType = ToastType.INFO,
                 duration: int = 4000, parent=None):
        """
        Initialize toast notification.
        
        Args:
            message: Notification message
            toast_type: Type of notification
            duration: Auto-dismiss duration in milliseconds
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.message = message
        self.toast_type = toast_type
        self.duration = duration
        
        # Animation properties
        self.slide_animation = None
        self.fade_animation = None
        self.opacity_effect = None
        
        # Auto-dismiss timer
        self.dismiss_timer = QTimer()
        self.dismiss_timer.setSingleShot(True)
        self.dismiss_timer.timeout.connect(self.dismiss)
        
        self._setup_ui()
        self._setup_animations()
        
        # Start auto-dismiss timer
        if duration > 0:
            self.dismiss_timer.start(duration)
    
    def _setup_ui(self):
        """Set up the notification UI."""
        self.setFixedHeight(60)
        self.setMinimumWidth(300)
        self.setMaximumWidth(500)
        
        # Main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        
        # Icon label
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(24, 24)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        
        # Message label
        self.message_label = QLabel(self.message)
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.message_label, 1)
        
        # Close button
        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(20, 20)
        self.close_button.clicked.connect(self.dismiss)
        layout.addWidget(self.close_button)
        
        # Apply styling
        self._apply_styling()
        
        # Set window flags for overlay
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
    def _apply_styling(self):
        """Apply styling based on toast type."""
        # Import theme manager
        from ui.themes import theme_manager
        
        # Get colors based on type
        if self.toast_type == ToastType.SUCCESS:
            bg_color = theme_manager.get_color("success")
            text_color = "#FFFFFF"
            icon_text = "✓"
        elif self.toast_type == ToastType.ERROR:
            bg_color = theme_manager.get_color("error")
            text_color = "#FFFFFF"
            icon_text = "✕"
        elif self.toast_type == ToastType.WARNING:
            bg_color = theme_manager.get_color("warning")
            text_color = "#FFFFFF"
            icon_text = "⚠"
        else:  # INFO
            bg_color = theme_manager.get_color("info")
            text_color = "#FFFFFF"
            icon_text = "ℹ"
        
        # Apply styles
        self.setStyleSheet(f"""
        QWidget {{
            background-color: {bg_color};
            border-radius: 8px;
            color: {text_color};
        }}
        QLabel {{
            color: {text_color};
            font-size: 13px;
        }}
        QPushButton {{
            background-color: transparent;
            border: none;
            color: {text_color};
            font-size: 16px;
            font-weight: bold;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: rgba(255, 255, 255, 0.2);
        }}
        """)
        
        # Set icon
        self.icon_label.setText(icon_text)
        self.icon_label.setStyleSheet(f"""
        QLabel {{
            font-size: 16px;
            font-weight: bold;
            color: {text_color};
        }}
        """)
        
        # Set message font
        font = theme_manager.get_font("body_medium")
        self.message_label.setFont(font)
    
    def _setup_animations(self):
        """Set up slide and fade animations."""
        # Opacity effect for fade animation
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        # Slide animation
        self.slide_animation = QPropertyAnimation(self, b"geometry")
        self.slide_animation.setDuration(300)
        self.slide_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Fade animation
        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(250)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def show_animated(self, target_position: QPoint):
        """Show toast with slide-in animation."""
        # Set initial position (off-screen to the right)
        start_rect = QRect(target_position.x() + 350, target_position.y(),
                          self.width(), self.height())
        end_rect = QRect(target_position.x(), target_position.y(),
                        self.width(), self.height())
        
        self.setGeometry(start_rect)
        self.show()
        
        # Start slide animation
        self.slide_animation.setStartValue(start_rect)
        self.slide_animation.setEndValue(end_rect)
        self.slide_animation.start()
        
        # Start fade in
        self.opacity_effect.setOpacity(0.0)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()
    
    def dismiss(self):
        """Dismiss toast with fade-out animation."""
        self.dismiss_timer.stop()
        
        # Fade out animation
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.finished.connect(self._on_fade_finished)
        self.fade_animation.start()
    
    def _on_fade_finished(self):
        """Handle fade animation completion."""
        self.dismissed.emit()
        self.hide()
        self.deleteLater()
    
    def mousePressEvent(self, event):
        """Handle mouse click to dismiss."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismiss()
        super().mousePressEvent(event)


class ToastManager(QWidget):
    """
    Toast notification manager.
    
    Manages multiple toast notifications with proper positioning,
    stacking, and lifecycle management.
    """
    
    def __init__(self, parent=None):
        """Initialize toast manager."""
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.active_toasts: List[ToastNotification] = []
        self.toast_spacing = 10
        self.margin_right = 20
        self.margin_bottom = 20
        
        # Set up as overlay widget
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Position manager window
        self._position_manager()
    
    def _position_manager(self):
        """Position the toast manager window."""
        if self.parent():
            parent_rect = self.parent().geometry()
            self.setGeometry(parent_rect)
        else:
            # Use screen geometry
            screen = QApplication.primaryScreen()
            screen_rect = screen.geometry()
            self.setGeometry(screen_rect)
    
    def show_toast(self, message: str, toast_type: ToastType = ToastType.INFO,
                   duration: int = 4000) -> ToastNotification:
        """
        Show a new toast notification.
        
        Args:
            message: Notification message
            toast_type: Type of notification
            duration: Auto-dismiss duration in milliseconds
            
        Returns:
            ToastNotification instance
        """
        # Create new toast
        toast = ToastNotification(message, toast_type, duration, self)
        toast.dismissed.connect(lambda: self._remove_toast(toast))
        
        # Calculate position
        position = self._calculate_toast_position(toast)
        
        # Add to active toasts
        self.active_toasts.append(toast)
        
        # Show with animation
        toast.show_animated(position)
        
        # Reposition existing toasts
        self._reposition_toasts()
        
        self.logger.debug(f"Showed {toast_type.value} toast: {message}")
        return toast
    
    def show_success(self, message: str, duration: int = 3000) -> ToastNotification:
        """Show success toast."""
        return self.show_toast(message, ToastType.SUCCESS, duration)
    
    def show_error(self, message: str, duration: int = 5000) -> ToastNotification:
        """Show error toast."""
        return self.show_toast(message, ToastType.ERROR, duration)
    
    def show_warning(self, message: str, duration: int = 4000) -> ToastNotification:
        """Show warning toast."""
        return self.show_toast(message, ToastType.WARNING, duration)
    
    def show_info(self, message: str, duration: int = 3000) -> ToastNotification:
        """Show info toast."""
        return self.show_toast(message, ToastType.INFO, duration)
    
    def _calculate_toast_position(self, toast: ToastNotification) -> QPoint:
        """Calculate position for new toast."""
        # Start from bottom-right corner
        x = self.width() - toast.width() - self.margin_right
        y = self.height() - toast.height() - self.margin_bottom
        
        # Move up for each existing toast
        for existing_toast in self.active_toasts:
            if existing_toast.isVisible():
                y -= existing_toast.height() + self.toast_spacing
        
        return QPoint(x, y)
    
    def _reposition_toasts(self):
        """Reposition all active toasts."""
        y_offset = self.height() - self.margin_bottom
        
        for toast in reversed(self.active_toasts):
            if toast.isVisible():
                y_offset -= toast.height()
                x = self.width() - toast.width() - self.margin_right
                
                # Animate to new position
                current_rect = toast.geometry()
                new_rect = QRect(x, y_offset, current_rect.width(), current_rect.height())
                
                if current_rect != new_rect:
                    animation = QPropertyAnimation(toast, b"geometry")
                    animation.setDuration(200)
                    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
                    animation.setStartValue(current_rect)
                    animation.setEndValue(new_rect)
                    animation.start()
                
                y_offset -= self.toast_spacing
    
    def _remove_toast(self, toast: ToastNotification):
        """Remove toast from active list."""
        if toast in self.active_toasts:
            self.active_toasts.remove(toast)
            self._reposition_toasts()
    
    def clear_all_toasts(self):
        """Clear all active toasts."""
        for toast in self.active_toasts.copy():
            toast.dismiss()
    
    def resizeEvent(self, event):
        """Handle resize event to reposition toasts."""
        super().resizeEvent(event)
        self._reposition_toasts()


# Convenience functions for global toast notifications
_global_toast_manager = None


def get_toast_manager(parent=None) -> ToastManager:
    """Get or create global toast manager."""
    global _global_toast_manager
    if _global_toast_manager is None:
        _global_toast_manager = ToastManager(parent)
    return _global_toast_manager


def show_success_toast(message: str, duration: int = 3000):
    """Show global success toast."""
    manager = get_toast_manager()
    return manager.show_success(message, duration)


def show_error_toast(message: str, duration: int = 5000):
    """Show global error toast."""
    manager = get_toast_manager()
    return manager.show_error(message, duration)


def show_warning_toast(message: str, duration: int = 4000):
    """Show global warning toast."""
    manager = get_toast_manager()
    return manager.show_warning(message, duration)


def show_info_toast(message: str, duration: int = 3000):
    """Show global info toast."""
    manager = get_toast_manager()
    return manager.show_info(message, duration)