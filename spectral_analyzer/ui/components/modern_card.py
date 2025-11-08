"""
Modern card component system with Material Design styling.

Provides reusable card components with shadows, hover effects,
and professional styling for the spectral analyzer interface.
"""

import logging
from typing import Optional, List, Any
from enum import Enum
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsDropShadowEffect, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal,
    QTimer, QSize
)
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QFont


class CardElevation(Enum):
    """Card elevation levels."""
    FLAT = 0
    LOW = 2
    MEDIUM = 4
    HIGH = 8
    VERY_HIGH = 16


class CardType(Enum):
    """Card types for different use cases."""
    DEFAULT = "default"
    OUTLINED = "outlined"
    FILLED = "filled"
    ELEVATED = "elevated"


class ModernCard(QFrame):
    """
    Modern card component with Material Design styling.
    
    Features:
    - Configurable elevation and shadows
    - Hover effects and animations
    - Rounded corners
    - Professional styling
    - Click interactions
    """
    
    clicked = pyqtSignal()
    hover_entered = pyqtSignal()
    hover_left = pyqtSignal()
    
    def __init__(self, title: str = "", subtitle: str = "", 
                 card_type: CardType = CardType.DEFAULT,
                 elevation: CardElevation = CardElevation.LOW,
                 clickable: bool = False, parent=None):
        """
        Initialize modern card.
        
        Args:
            title: Card title
            subtitle: Card subtitle
            card_type: Type of card styling
            elevation: Shadow elevation level
            clickable: Whether card responds to clicks
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.title_text = title
        self.subtitle_text = subtitle
        self.card_type = card_type
        self.elevation = elevation
        self.clickable = clickable
        
        # Animation properties
        self.hover_animation = None
        self.shadow_effect = None
        self.is_hovered = False
        
        # Content widget
        self.content_widget = None
        self.title_label = None
        self.subtitle_label = None
        
        self._setup_ui()
        self._setup_animations()
        self._apply_styling()
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
    
    def _setup_ui(self):
        """Set up the card UI structure."""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(8)
        
        # Header section
        if self.title_text or self.subtitle_text:
            self._create_header()
        
        # Content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.content_widget)
        
        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    
    def _create_header(self):
        """Create card header with title and subtitle."""
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        # Title
        if self.title_text:
            from ui.themes import theme_manager
            
            self.title_label = QLabel(self.title_text)
            self.title_label.setFont(theme_manager.get_font("headline_small"))
            self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {theme_manager.get_color("text_primary")};
                font-weight: 600;
            }}
            """)
            header_layout.addWidget(self.title_label)
        
        # Subtitle
        if self.subtitle_text:
            from ui.themes import theme_manager
            
            self.subtitle_label = QLabel(self.subtitle_text)
            self.subtitle_label.setFont(theme_manager.get_font("body_medium"))
            self.subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {theme_manager.get_color("text_secondary")};
            }}
            """)
            self.subtitle_label.setWordWrap(True)
            header_layout.addWidget(self.subtitle_label)
        
        self.main_layout.addWidget(header_widget)
    
    def _setup_animations(self):
        """Set up hover animations."""
        # Hover animation for elevation change
        self.hover_animation = QPropertyAnimation(self, b"geometry")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def _apply_styling(self):
        """Apply card styling based on type and theme."""
        from ui.themes import theme_manager
        
        # Base styling
        self.setFrameStyle(QFrame.Shape.NoFrame)
        
        if self.card_type == CardType.OUTLINED:
            self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.get_color("surface")};
                border: 1px solid {theme_manager.get_color("border")};
                border-radius: 12px;
            }}
            """)
        elif self.card_type == CardType.FILLED:
            self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.get_color("surface_variant")};
                border: none;
                border-radius: 12px;
            }}
            """)
        else:  # DEFAULT or ELEVATED
            self.setStyleSheet(f"""
            QFrame {{
                background-color: {theme_manager.get_color("card_background")};
                border: 1px solid {theme_manager.get_color("card_border")};
                border-radius: 12px;
            }}
            """)
        
        # Apply shadow effect
        self._update_shadow()
        
        # Set cursor for clickable cards
        if self.clickable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
    
    def _update_shadow(self, elevated: bool = False):
        """Update shadow effect based on elevation."""
        if self.shadow_effect:
            self.shadow_effect.deleteLater()
        
        if self.elevation == CardElevation.FLAT:
            return
        
        from ui.themes import theme_manager
        
        # Calculate shadow parameters
        base_blur = self.elevation.value
        base_offset = max(1, self.elevation.value // 2)
        
        if elevated:
            blur_radius = base_blur * 2
            offset = base_offset * 2
        else:
            blur_radius = base_blur
            offset = base_offset
        
        # Create shadow effect
        self.shadow_effect = theme_manager.create_shadow_effect(
            blur_radius=blur_radius,
            offset=(0, offset)
        )
        self.setGraphicsEffect(self.shadow_effect)
    
    def add_content(self, widget: QWidget):
        """Add content widget to the card."""
        self.content_layout.addWidget(widget)
    
    def add_action_button(self, text: str, callback=None, button_type: str = "primary") -> QPushButton:
        """Add action button to the card."""
        from ui.themes import theme_manager
        
        button = QPushButton(text)
        button.setStyleSheet(theme_manager.create_button_style(button_type))
        
        if callback:
            button.clicked.connect(callback)
        
        # Add button to a horizontal layout at the bottom
        if not hasattr(self, 'action_layout'):
            action_widget = QWidget()
            self.action_layout = QHBoxLayout(action_widget)
            self.action_layout.setContentsMargins(0, 8, 0, 0)
            self.action_layout.addStretch()
            self.main_layout.addWidget(action_widget)
        
        self.action_layout.addWidget(button)
        return button
    
    def set_title(self, title: str):
        """Update card title."""
        self.title_text = title
        if self.title_label:
            self.title_label.setText(title)
    
    def set_subtitle(self, subtitle: str):
        """Update card subtitle."""
        self.subtitle_text = subtitle
        if self.subtitle_label:
            self.subtitle_label.setText(subtitle)
    
    def enterEvent(self, event):
        """Handle mouse enter event."""
        if self.clickable or self.elevation != CardElevation.FLAT:
            self.is_hovered = True
            self._update_shadow(elevated=True)
            self.hover_entered.emit()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave event."""
        if self.is_hovered:
            self.is_hovered = False
            self._update_shadow(elevated=False)
            self.hover_left.emit()
        super().leaveEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press event."""
        if self.clickable and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DataCard(ModernCard):
    """
    Specialized card for displaying data with metrics.
    
    Features:
    - Large metric display
    - Trend indicators
    - Status colors
    """
    
    def __init__(self, title: str, value: str, unit: str = "",
                 trend: Optional[float] = None, status: str = "normal",
                 parent=None):
        """
        Initialize data card.
        
        Args:
            title: Metric title
            value: Metric value
            unit: Value unit
            trend: Trend percentage (positive/negative)
            status: Status color (normal, success, warning, error)
            parent: Parent widget
        """
        super().__init__(title=title, parent=parent)
        
        self.value = value
        self.unit = unit
        self.trend = trend
        self.status = status
        
        self._create_data_display()
    
    def _create_data_display(self):
        """Create data display layout."""
        from ui.themes import theme_manager
        
        # Main value display
        value_layout = QHBoxLayout()
        
        # Value label
        value_label = QLabel(self.value)
        value_label.setFont(theme_manager.get_font("display_small"))
        
        # Status color
        if self.status == "success":
            color = theme_manager.get_color("success")
        elif self.status == "warning":
            color = theme_manager.get_color("warning")
        elif self.status == "error":
            color = theme_manager.get_color("error")
        else:
            color = theme_manager.get_color("text_primary")
        
        value_label.setStyleSheet(f"""
        QLabel {{
            color: {color};
            font-weight: 700;
        }}
        """)
        value_layout.addWidget(value_label)
        
        # Unit label
        if self.unit:
            unit_label = QLabel(self.unit)
            unit_label.setFont(theme_manager.get_font("body_large"))
            unit_label.setStyleSheet(f"""
            QLabel {{
                color: {theme_manager.get_color("text_secondary")};
                margin-left: 4px;
            }}
            """)
            value_layout.addWidget(unit_label)
        
        value_layout.addStretch()
        
        # Trend indicator
        if self.trend is not None:
            trend_label = QLabel()
            if self.trend > 0:
                trend_label.setText(f"↗ +{self.trend:.1f}%")
                trend_color = theme_manager.get_color("success")
            elif self.trend < 0:
                trend_label.setText(f"↘ {self.trend:.1f}%")
                trend_color = theme_manager.get_color("error")
            else:
                trend_label.setText("→ 0.0%")
                trend_color = theme_manager.get_color("text_secondary")
            
            trend_label.setStyleSheet(f"""
            QLabel {{
                color: {trend_color};
                font-weight: 600;
            }}
            """)
            value_layout.addWidget(trend_label)
        
        # Add to content
        value_widget = QWidget()
        value_widget.setLayout(value_layout)
        self.add_content(value_widget)


class ActionCard(ModernCard):
    """
    Specialized card for actions with icon and description.
    
    Features:
    - Large action icon
    - Description text
    - Primary action button
    """
    
    def __init__(self, title: str, description: str, icon: str = "",
                 action_text: str = "Action", action_callback=None,
                 parent=None):
        """
        Initialize action card.
        
        Args:
            title: Action title
            description: Action description
            icon: Icon character or emoji
            action_text: Action button text
            action_callback: Action button callback
            parent: Parent widget
        """
        super().__init__(title=title, clickable=True, parent=parent)
        
        self.description = description
        self.icon = icon
        self.action_text = action_text
        self.action_callback = action_callback
        
        self._create_action_display()
    
    def _create_action_display(self):
        """Create action display layout."""
        from ui.themes import theme_manager
        
        # Icon and description layout
        content_layout = QHBoxLayout()
        
        # Icon
        if self.icon:
            icon_label = QLabel(self.icon)
            icon_label.setFont(theme_manager.get_font("display_medium"))
            icon_label.setStyleSheet(f"""
            QLabel {{
                color: {theme_manager.get_color("primary")};
            }}
            """)
            icon_label.setFixedSize(48, 48)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content_layout.addWidget(icon_label)
        
        # Description
        desc_label = QLabel(self.description)
        desc_label.setFont(theme_manager.get_font("body_medium"))
        desc_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_secondary")};
        }}
        """)
        desc_label.setWordWrap(True)
        content_layout.addWidget(desc_label, 1)
        
        # Add to content
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        self.add_content(content_widget)
        
        # Add action button
        self.add_action_button(self.action_text, self.action_callback)


class ProgressCard(ModernCard):
    """
    Specialized card for displaying progress information.
    
    Features:
    - Progress bar
    - Status text
    - Percentage display
    """
    
    def __init__(self, title: str, progress: float = 0.0,
                 status_text: str = "", parent=None):
        """
        Initialize progress card.
        
        Args:
            title: Progress title
            progress: Progress value (0.0 to 1.0)
            status_text: Current status text
            parent: Parent widget
        """
        super().__init__(title=title, parent=parent)
        
        self.progress = progress
        self.status_text = status_text
        
        self._create_progress_display()
    
    def _create_progress_display(self):
        """Create progress display layout."""
        from ui.themes import theme_manager
        from PyQt6.QtWidgets import QProgressBar
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(self.progress * 100))
        self.progress_bar.setStyleSheet(f"""
        QProgressBar {{
            border: 1px solid {theme_manager.get_color("border")};
            border-radius: 4px;
            background-color: {theme_manager.get_color("surface_variant")};
            text-align: center;
            height: 20px;
        }}
        QProgressBar::chunk {{
            background-color: {theme_manager.get_color("primary")};
            border-radius: 3px;
        }}
        """)
        
        # Status text
        if self.status_text:
            status_label = QLabel(self.status_text)
            status_label.setFont(theme_manager.get_font("body_small"))
            status_label.setStyleSheet(f"""
            QLabel {{
                color: {theme_manager.get_color("text_secondary")};
                margin-top: 4px;
            }}
            """)
            
            # Add both to content
            progress_widget = QWidget()
            progress_layout = QVBoxLayout(progress_widget)
            progress_layout.setContentsMargins(0, 0, 0, 0)
            progress_layout.setSpacing(4)
            progress_layout.addWidget(self.progress_bar)
            progress_layout.addWidget(status_label)
            
            self.add_content(progress_widget)
        else:
            self.add_content(self.progress_bar)
    
    def update_progress(self, progress: float, status_text: str = None):
        """Update progress value and status."""
        self.progress = progress
        self.progress_bar.setValue(int(progress * 100))
        
        if status_text is not None:
            self.status_text = status_text
            # Update status label if it exists
            # This would need additional tracking of the status label widget