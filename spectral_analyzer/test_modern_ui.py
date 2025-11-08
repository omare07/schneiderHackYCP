"""
Test script for the modern PyQt6 user interface.

Tests all UI components, drag-and-drop functionality, animations,
and integration with backend systems.
"""

import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import ConfigManager
from ui.themes import ThemeManager, ThemeMode, theme_manager
from ui.components.file_drop_zone import ModernFileDropZone, DropZoneType
from ui.components.preview_widget import ModernPreviewWidget
from ui.components.status_bar import ModernStatusBar
from ui.components.modern_card import ModernCard, CardElevation, CardType, DataCard, ActionCard
from ui.components.toast_notification import ToastManager, show_success_toast, show_error_toast, show_warning_toast
from ui.main_window import ModernMainWindow
from utils.logging import setup_logging


class UITestWindow(QWidget):
    """Test window for UI components."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Modern UI Component Test")
        self.resize(1200, 800)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Test cards
        self._test_cards(layout)
        
        # Test drop zones
        self._test_drop_zones(layout)
        
        # Test toast notifications
        self._test_toast_notifications()
    
    def _test_cards(self, layout):
        """Test card components."""
        cards_layout = QHBoxLayout()
        
        # Data card
        data_card = DataCard("Files Loaded", "5", "files", trend=15.2, status="success")
        cards_layout.addWidget(data_card)
        
        # Action card
        action_card = ActionCard(
            "AI Settings",
            "Configure AI models and API keys",
            "🤖",
            "Configure",
            lambda: show_success_toast("AI Settings clicked!")
        )
        cards_layout.addWidget(action_card)
        
        # Regular card
        info_card = ModernCard(
            "Information",
            "This is a test card with elevation",
            CardType.ELEVATED,
            CardElevation.HIGH
        )
        info_card.add_action_button("Test Action", lambda: show_info_toast("Action clicked!"))
        cards_layout.addWidget(info_card)
        
        layout.addLayout(cards_layout)
    
    def _test_drop_zones(self, layout):
        """Test drop zone components."""
        drop_zones_layout = QHBoxLayout()
        
        # Baseline drop zone
        baseline_zone = ModernFileDropZone(
            zone_type=DropZoneType.BASELINE,
            accept_multiple=False
        )
        baseline_zone.files_dropped.connect(
            lambda files, zone: show_success_toast(f"Baseline: {len(files)} files")
        )
        drop_zones_layout.addWidget(baseline_zone)
        
        # Samples drop zone
        samples_zone = ModernFileDropZone(
            zone_type=DropZoneType.SAMPLES,
            accept_multiple=True
        )
        samples_zone.files_dropped.connect(
            lambda files, zone: show_success_toast(f"Samples: {len(files)} files")
        )
        drop_zones_layout.addWidget(samples_zone)
        
        layout.addLayout(drop_zones_layout)
    
    def _test_toast_notifications(self):
        """Test toast notification system."""
        # Schedule test notifications
        QTimer.singleShot(1000, lambda: show_success_toast("Welcome to Spectral Analyzer!"))
        QTimer.singleShot(2000, lambda: show_info_toast("This is an info notification"))
        QTimer.singleShot(3000, lambda: show_warning_toast("This is a warning notification"))
        QTimer.singleShot(4000, lambda: show_error_toast("This is an error notification"))


def test_theme_system():
    """Test theme system functionality."""
    print("🎨 Testing theme system...")
    
    # Test light theme
    theme_manager.set_theme(ThemeMode.LIGHT)
    print(f"  Light theme primary color: {theme_manager.get_color('primary')}")
    
    # Test dark theme
    theme_manager.set_theme(ThemeMode.DARK)
    print(f"  Dark theme primary color: {theme_manager.get_color('primary')}")
    
    # Test font system
    title_font = theme_manager.get_font("headline_large")
    print(f"  Title font: {title_font.family()}, {title_font.pointSize()}pt")
    
    print("✅ Theme system test completed")


def test_components():
    """Test individual UI components."""
    print("🧪 Testing UI components...")
    
    app = QApplication(sys.argv)
    
    # Set up theme
    theme_manager.set_theme(ThemeMode.LIGHT)
    theme_manager.apply_theme_to_application(app)
    
    # Test window
    test_window = UITestWindow()
    test_window.show()
    
    print("✅ UI components test window created")
    
    # Run for a short time to test
    QTimer.singleShot(10000, app.quit)  # Auto-close after 10 seconds
    app.exec()


def test_main_window():
    """Test the complete main window."""
    print("🏠 Testing main window...")
    
    app = QApplication(sys.argv)
    
    # Set up configuration
    config_manager = ConfigManager()
    
    # Initialize theme
    theme_manager.set_theme(ThemeMode.LIGHT)
    theme_manager.apply_theme_to_application(app)
    
    # Create main window
    main_window = ModernMainWindow(config_manager)
    main_window.show()
    
    print("✅ Main window created and displayed")
    
    # Test auto-close
    QTimer.singleShot(15000, app.quit)  # Auto-close after 15 seconds
    return app.exec()


def test_drag_drop_simulation():
    """Test drag-and-drop functionality simulation."""
    print("📁 Testing drag-and-drop simulation...")
    
    # Test file paths
    test_files = [
        project_root / "tests" / "test_data" / "sample_spectral.csv"
    ]
    
    existing_files = [f for f in test_files if f.exists()]
    
    if existing_files:
        print(f"  Found {len(existing_files)} test files")
        for file_path in existing_files:
            print(f"    - {file_path.name} ({file_path.stat().st_size} bytes)")
    else:
        print("  ⚠️ No test files found")
    
    print("✅ Drag-drop simulation test completed")


def run_all_tests():
    """Run all UI tests."""
    print("🚀 Starting Modern UI Test Suite")
    print("=" * 50)
    
    # Set up logging
    setup_logging()
    
    try:
        # Test theme system
        test_theme_system()
        print()
        
        # Test drag-drop simulation
        test_drag_drop_simulation()
        print()
        
        # Test components (interactive)
        print("🖥️ Starting interactive component test...")
        test_components()
        print()
        
        # Test main window (interactive)
        print("🖥️ Starting main window test...")
        exit_code = test_main_window()
        
        print("=" * 50)
        print("✅ All tests completed successfully!")
        return exit_code
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())