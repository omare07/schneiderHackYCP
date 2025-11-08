#!/usr/bin/env python3
"""
AI-Powered Spectral Analysis Desktop Application
Main application entry point

Author: MRG Labs Development Team
Version: 1.0.0
License: Proprietary
"""

import sys
import logging
import traceback
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

from config.settings import ConfigManager
from ui.main_window import ModernMainWindow
from ui.themes import ThemeManager, ThemeMode, theme_manager
from utils.logging import setup_logging
from utils.error_handling import GlobalExceptionHandler


class SpectralAnalyzerApp:
    """Main application class for the Spectral Analyzer."""
    
    def __init__(self):
        """Initialize the application."""
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None
        self.config_manager: Optional[ConfigManager] = None
        self.exception_handler: Optional[GlobalExceptionHandler] = None
        
    def initialize(self) -> bool:
        """
        Initialize the application components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Set up logging first
            setup_logging()
            self.logger = logging.getLogger(__name__)
            self.logger.info("Starting Spectral Analyzer Application")
            
            # Initialize Qt Application
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("Spectral Analyzer")
            self.app.setApplicationVersion("1.0.0")
            self.app.setOrganizationName("MRG Labs")
            # Note: High DPI scaling is enabled by default in PyQt6
            # No need to set AA_EnableHighDpiScaling or AA_UseHighDpiPixmaps
            
            
            # Initialize configuration manager
            self.config_manager = ConfigManager()
            
            # Set up global exception handler
            self.exception_handler = GlobalExceptionHandler()
            sys.excepthook = self.exception_handler.handle_exception
            
            # Initialize theme system
            theme_mode = self.config_manager.get_setting('ui', 'theme', 'light')
            if theme_mode == 'dark':
                theme_manager.set_theme(ThemeMode.DARK)
            elif theme_mode == 'auto':
                theme_manager.set_theme(ThemeMode.AUTO)
            else:
                theme_manager.set_theme(ThemeMode.LIGHT)
            
            # Apply theme to application
            theme_manager.apply_theme_to_application(self.app)
            
            # Initialize modern main window
            self.main_window = ModernMainWindow(self.config_manager)
            
            # Set application icon
            icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
            if icon_path.exists():
                self.app.setWindowIcon(QIcon(str(icon_path)))
            
            self.logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            self.logger.error(traceback.format_exc())
            self._show_critical_error("Initialization Error", str(e))
            return False
    
    def run(self) -> int:
        """
        Run the application main loop.
        
        Returns:
            int: Application exit code
        """
        if not self.app or not self.main_window:
            return 1
            
        try:
            # Show main window
            self.main_window.show()
            
            # Start application event loop
            return self.app.exec()
            
        except Exception as e:
            self.logger.error(f"Application runtime error: {e}")
            self.logger.error(traceback.format_exc())
            self._show_critical_error("Runtime Error", str(e))
            return 1
    
    def shutdown(self):
        """Clean shutdown of the application."""
        try:
            self.logger.info("Shutting down Spectral Analyzer Application")
            
            if self.main_window:
                self.main_window.cleanup()
                
            if self.config_manager:
                self.config_manager.save_settings()
                
            self.logger.info("Application shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _show_critical_error(self, title: str, message: str):
        """Show critical error dialog."""
        if self.app:
            QMessageBox.critical(None, title, message)
        else:
            print(f"CRITICAL ERROR - {title}: {message}")


def main():
    """Main entry point."""
    app = SpectralAnalyzerApp()
    
    try:
        if not app.initialize():
            return 1
            
        exit_code = app.run()
        app.shutdown()
        return exit_code
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        app.shutdown()
        return 0
        
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())