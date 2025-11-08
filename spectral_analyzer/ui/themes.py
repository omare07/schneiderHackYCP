"""
Modern theme system for Spectral Analyzer.

Provides professional Material Design-inspired themes with dark/light mode support,
smooth animations, and comprehensive styling for all UI components.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QRect, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtWidgets import QApplication, QWidget, QGraphicsDropShadowEffect


class ThemeMode(Enum):
    """Available theme modes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class AnimationType(Enum):
    """Animation types for UI transitions."""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    BOUNCE = "bounce"


class ThemeManager(QObject):
    """
    Comprehensive theme management system.
    
    Features:
    - Material Design color schemes
    - Dark/light mode support
    - Smooth animations and transitions
    - Professional styling with shadows and effects
    - System theme detection
    """
    
    theme_changed = pyqtSignal(str)  # Theme name
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Current theme
        self.current_mode = ThemeMode.LIGHT
        self.current_theme = "light"
        
        # Animation settings
        self.animation_duration = 250  # milliseconds
        self.animation_easing = QEasingCurve.Type.OutCubic
        
        # Color schemes
        self.themes = {
            "light": self._create_light_theme(),
            "dark": self._create_dark_theme()
        }
        
        # Typography
        self.fonts = self._create_font_system()
        
        self.logger.info("Theme manager initialized")
    
    def _create_light_theme(self) -> Dict[str, Any]:
        """Create light theme color scheme."""
        return {
            # Primary colors
            "primary": "#1976D2",
            "primary_light": "#42A5F5",
            "primary_dark": "#1565C0",
            "primary_variant": "#0D47A1",
            
            # Secondary colors
            "secondary": "#03DAC6",
            "secondary_light": "#4AEDC4",
            "secondary_dark": "#00A896",
            
            # Background colors
            "background": "#FAFAFA",
            "surface": "#FFFFFF",
            "surface_variant": "#F5F5F5",
            "surface_container": "#F8F9FA",
            
            # Text colors
            "on_background": "#212121",
            "on_surface": "#212121",
            "on_primary": "#FFFFFF",
            "on_secondary": "#000000",
            "text_primary": "#212121",
            "text_secondary": "#757575",
            "text_disabled": "#BDBDBD",
            
            # Status colors
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "info": "#2196F3",
            
            # Border and divider colors
            "border": "#E0E0E0",
            "divider": "#E0E0E0",
            "outline": "#79747E",
            
            # Interactive states
            "hover": "#F5F5F5",
            "pressed": "#EEEEEE",
            "focus": "#E3F2FD",
            "selected": "#E8F5E8",
            
            # Shadows
            "shadow": "rgba(0, 0, 0, 0.12)",
            "shadow_dark": "rgba(0, 0, 0, 0.24)",
            
            # Card colors
            "card_background": "#FFFFFF",
            "card_border": "#E0E0E0",
            "card_shadow": "rgba(0, 0, 0, 0.08)",
            
            # Drop zone colors
            "drop_zone_border": "#CCCCCC",
            "drop_zone_hover": "#4CAF50",
            "drop_zone_active": "#2E7D32",
            "drop_zone_background": "rgba(76, 175, 80, 0.1)",
        }
    
    def _create_dark_theme(self) -> Dict[str, Any]:
        """Create dark theme color scheme."""
        return {
            # Primary colors
            "primary": "#90CAF9",
            "primary_light": "#BBDEFB",
            "primary_dark": "#64B5F6",
            "primary_variant": "#42A5F5",
            
            # Secondary colors
            "secondary": "#80CBC4",
            "secondary_light": "#B2DFDB",
            "secondary_dark": "#4DB6AC",
            
            # Background colors
            "background": "#121212",
            "surface": "#1E1E1E",
            "surface_variant": "#2C2C2C",
            "surface_container": "#242424",
            
            # Text colors
            "on_background": "#FFFFFF",
            "on_surface": "#FFFFFF",
            "on_primary": "#000000",
            "on_secondary": "#000000",
            "text_primary": "#FFFFFF",
            "text_secondary": "#B3B3B3",
            "text_disabled": "#666666",
            
            # Status colors
            "success": "#66BB6A",
            "warning": "#FFB74D",
            "error": "#EF5350",
            "info": "#64B5F6",
            
            # Border and divider colors
            "border": "#404040",
            "divider": "#404040",
            "outline": "#938F99",
            
            # Interactive states
            "hover": "#2C2C2C",
            "pressed": "#383838",
            "focus": "#1E3A8A",
            "selected": "#2E7D32",
            
            # Shadows
            "shadow": "rgba(0, 0, 0, 0.3)",
            "shadow_dark": "rgba(0, 0, 0, 0.5)",
            
            # Card colors
            "card_background": "#1E1E1E",
            "card_border": "#404040",
            "card_shadow": "rgba(0, 0, 0, 0.2)",
            
            # Drop zone colors
            "drop_zone_border": "#666666",
            "drop_zone_hover": "#66BB6A",
            "drop_zone_active": "#4CAF50",
            "drop_zone_background": "rgba(102, 187, 106, 0.1)",
        }
    
    def _create_font_system(self) -> Dict[str, QFont]:
        """Create typography system."""
        fonts = {}
        
        # Base font
        base_font = QFont("Segoe UI", 10)
        base_font.setStyleHint(QFont.StyleHint.SansSerif)
        
        # Typography scale
        fonts["display_large"] = QFont(base_font)
        fonts["display_large"].setPointSize(24)
        fonts["display_large"].setWeight(QFont.Weight.Bold)
        
        fonts["display_medium"] = QFont(base_font)
        fonts["display_medium"].setPointSize(20)
        fonts["display_medium"].setWeight(QFont.Weight.Bold)
        
        fonts["display_small"] = QFont(base_font)
        fonts["display_small"].setPointSize(18)
        fonts["display_small"].setWeight(QFont.Weight.Bold)
        
        fonts["headline_large"] = QFont(base_font)
        fonts["headline_large"].setPointSize(16)
        fonts["headline_large"].setWeight(QFont.Weight.Bold)
        
        fonts["headline_medium"] = QFont(base_font)
        fonts["headline_medium"].setPointSize(14)
        fonts["headline_medium"].setWeight(QFont.Weight.Bold)
        
        fonts["headline_small"] = QFont(base_font)
        fonts["headline_small"].setPointSize(12)
        fonts["headline_small"].setWeight(QFont.Weight.Bold)
        
        fonts["title_large"] = QFont(base_font)
        fonts["title_large"].setPointSize(14)
        fonts["title_large"].setWeight(QFont.Weight.Medium)
        
        fonts["title_medium"] = QFont(base_font)
        fonts["title_medium"].setPointSize(12)
        fonts["title_medium"].setWeight(QFont.Weight.Medium)
        
        fonts["title_small"] = QFont(base_font)
        fonts["title_small"].setPointSize(11)
        fonts["title_small"].setWeight(QFont.Weight.Medium)
        
        fonts["body_large"] = QFont(base_font)
        fonts["body_large"].setPointSize(11)
        
        fonts["body_medium"] = QFont(base_font)
        fonts["body_medium"].setPointSize(10)
        
        fonts["body_small"] = QFont(base_font)
        fonts["body_small"].setPointSize(9)
        
        fonts["label_large"] = QFont(base_font)
        fonts["label_large"].setPointSize(10)
        fonts["label_large"].setWeight(QFont.Weight.Medium)
        
        fonts["label_medium"] = QFont(base_font)
        fonts["label_medium"].setPointSize(9)
        fonts["label_medium"].setWeight(QFont.Weight.Medium)
        
        fonts["label_small"] = QFont(base_font)
        fonts["label_small"].setPointSize(8)
        fonts["label_small"].setWeight(QFont.Weight.Medium)
        
        # Monospace font for code/data
        fonts["monospace"] = QFont("Consolas", 9)
        fonts["monospace"].setStyleHint(QFont.StyleHint.Monospace)
        
        return fonts
    
    def set_theme(self, mode: ThemeMode):
        """Set the current theme mode."""
        self.current_mode = mode
        
        if mode == ThemeMode.AUTO:
            # Detect system theme (simplified - would need platform-specific code)
            self.current_theme = "dark"  # Default to dark for now
        else:
            self.current_theme = mode.value
        
        self.logger.info(f"Theme changed to: {self.current_theme}")
        self.theme_changed.emit(self.current_theme)
    
    def get_color(self, color_name: str) -> str:
        """Get color value from current theme."""
        theme = self.themes.get(self.current_theme, self.themes["light"])
        return theme.get(color_name, "#000000")
    
    def get_font(self, font_name: str) -> QFont:
        """Get font from typography system."""
        return self.fonts.get(font_name, self.fonts["body_medium"])
    
    def create_card_style(self, elevated: bool = True) -> str:
        """Create CSS style for card components."""
        theme = self.themes[self.current_theme]
        
        shadow = "box-shadow: 0 2px 8px " + theme["card_shadow"] + ";" if elevated else ""
        
        return f"""
        QWidget {{
            background-color: {theme["card_background"]};
            border: 1px solid {theme["card_border"]};
            border-radius: 8px;
            {shadow}
        }}
        """
    
    def create_button_style(self, button_type: str = "primary") -> str:
        """Create CSS style for buttons."""
        theme = self.themes[self.current_theme]
        
        if button_type == "primary":
            return f"""
            QPushButton {{
                background-color: {theme["primary"]};
                color: {theme["on_primary"]};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {theme["primary_light"]};
            }}
            QPushButton:pressed {{
                background-color: {theme["primary_dark"]};
            }}
            QPushButton:disabled {{
                background-color: {theme["text_disabled"]};
                color: {theme["text_secondary"]};
            }}
            """
        elif button_type == "secondary":
            return f"""
            QPushButton {{
                background-color: transparent;
                color: {theme["primary"]};
                border: 1px solid {theme["primary"]};
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {theme["focus"]};
            }}
            QPushButton:pressed {{
                background-color: {theme["pressed"]};
            }}
            """
        else:  # text button
            return f"""
            QPushButton {{
                background-color: transparent;
                color: {theme["primary"]};
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {theme["hover"]};
            }}
            QPushButton:pressed {{
                background-color: {theme["pressed"]};
            }}
            """
    
    def create_input_style(self) -> str:
        """Create CSS style for input fields."""
        theme = self.themes[self.current_theme]
        
        return f"""
        QLineEdit, QTextEdit, QComboBox {{
            background-color: {theme["surface"]};
            color: {theme["text_primary"]};
            border: 1px solid {theme["border"]};
            border-radius: 4px;
            padding: 8px 12px;
            selection-background-color: {theme["primary"]};
        }}
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
            border: 2px solid {theme["primary"]};
            background-color: {theme["focus"]};
        }}
        QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled {{
            background-color: {theme["surface_variant"]};
            color: {theme["text_disabled"]};
        }}
        """
    
    def create_drop_zone_style(self, is_active: bool = False, is_hover: bool = False) -> str:
        """Create CSS style for drag-drop zones."""
        theme = self.themes[self.current_theme]
        
        if is_active:
            border_color = theme["drop_zone_active"]
            background_color = theme["drop_zone_background"]
        elif is_hover:
            border_color = theme["drop_zone_hover"]
            background_color = theme["drop_zone_background"]
        else:
            border_color = theme["drop_zone_border"]
            background_color = "transparent"
        
        return f"""
        QFrame {{
            border: 2px dashed {border_color};
            border-radius: 8px;
            background-color: {background_color};
            padding: 20px;
        }}
        """
    
    def create_status_bar_style(self) -> str:
        """Create CSS style for status bar."""
        theme = self.themes[self.current_theme]
        
        return f"""
        QStatusBar {{
            background-color: {theme["surface_container"]};
            color: {theme["text_primary"]};
            border-top: 1px solid {theme["border"]};
            padding: 4px 8px;
        }}
        QStatusBar QLabel {{
            color: {theme["text_secondary"]};
            padding: 2px 4px;
        }}
        QProgressBar {{
            border: 1px solid {theme["border"]};
            border-radius: 3px;
            background-color: {theme["surface_variant"]};
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {theme["primary"]};
            border-radius: 2px;
        }}
        """
    
    def create_menu_style(self) -> str:
        """Create CSS style for menus."""
        theme = self.themes[self.current_theme]
        
        return f"""
        QMenuBar {{
            background-color: {theme["surface"]};
            color: {theme["text_primary"]};
            border-bottom: 1px solid {theme["border"]};
            padding: 4px;
        }}
        QMenuBar::item {{
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 4px;
        }}
        QMenuBar::item:selected {{
            background-color: {theme["hover"]};
        }}
        QMenuBar::item:pressed {{
            background-color: {theme["pressed"]};
        }}
        QMenu {{
            background-color: {theme["surface"]};
            color: {theme["text_primary"]};
            border: 1px solid {theme["border"]};
            border-radius: 6px;
            padding: 4px;
        }}
        QMenu::item {{
            padding: 8px 16px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{
            background-color: {theme["hover"]};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {theme["divider"]};
            margin: 4px 8px;
        }}
        """
    
    def create_tab_style(self) -> str:
        """Create CSS style for tab widgets."""
        theme = self.themes[self.current_theme]
        
        return f"""
        QTabWidget::pane {{
            border: 1px solid {theme["border"]};
            background-color: {theme["surface"]};
            border-radius: 6px;
        }}
        QTabBar::tab {{
            background-color: {theme["surface_variant"]};
            color: {theme["text_secondary"]};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            min-width: 80px;
        }}
        QTabBar::tab:selected {{
            background-color: {theme["surface"]};
            color: {theme["text_primary"]};
            border-bottom: 2px solid {theme["primary"]};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {theme["hover"]};
        }}
        """
    
    def create_shadow_effect(self, blur_radius: int = 8, offset: tuple = (0, 2)) -> QGraphicsDropShadowEffect:
        """Create drop shadow effect for widgets."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(blur_radius)
        shadow.setOffset(offset[0], offset[1])
        
        theme = self.themes[self.current_theme]
        shadow_color = QColor(theme["shadow"])
        shadow.setColor(shadow_color)
        
        return shadow
    
    def animate_widget(self, widget: QWidget, animation_type: AnimationType, 
                      duration: Optional[int] = None, callback: Optional[callable] = None):
        """Animate widget with specified animation type."""
        if duration is None:
            duration = self.animation_duration
        
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setEasingCurve(self.animation_easing)
        
        if callback:
            animation.finished.connect(callback)
        
        current_geometry = widget.geometry()
        
        if animation_type == AnimationType.FADE_IN:
            widget.setWindowOpacity(0.0)
            fade_animation = QPropertyAnimation(widget, b"windowOpacity")
            fade_animation.setDuration(duration)
            fade_animation.setStartValue(0.0)
            fade_animation.setEndValue(1.0)
            fade_animation.start()
            
        elif animation_type == AnimationType.SLIDE_IN:
            start_geometry = QRect(current_geometry.x() - 50, current_geometry.y(),
                                 current_geometry.width(), current_geometry.height())
            animation.setStartValue(start_geometry)
            animation.setEndValue(current_geometry)
            animation.start()
            
        elif animation_type == AnimationType.SCALE_IN:
            start_geometry = QRect(current_geometry.center().x(), current_geometry.center().y(),
                                 0, 0)
            animation.setStartValue(start_geometry)
            animation.setEndValue(current_geometry)
            animation.start()
    
    def apply_theme_to_application(self, app: QApplication):
        """Apply current theme to the entire application."""
        theme = self.themes[self.current_theme]
        
        # Create application palette
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, QColor(theme["background"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(theme["text_primary"]))
        
        # Base colors (for input fields)
        palette.setColor(QPalette.ColorRole.Base, QColor(theme["surface"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme["surface_variant"]))
        
        # Text colors
        palette.setColor(QPalette.ColorRole.Text, QColor(theme["text_primary"]))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(theme["text_primary"]))
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, QColor(theme["surface"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme["text_primary"]))
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(theme["primary"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(theme["on_primary"]))
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, 
                        QColor(theme["text_disabled"]))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, 
                        QColor(theme["text_disabled"]))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, 
                        QColor(theme["text_disabled"]))
        
        app.setPalette(palette)
        
        # Set application font
        app.setFont(self.get_font("body_medium"))
        
        self.logger.info(f"Applied {self.current_theme} theme to application")


# Global theme manager instance
theme_manager = ThemeManager()