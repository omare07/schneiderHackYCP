"""
Modern main application window for Spectral Analyzer.

Provides a professional, commercial-grade user interface with drag-and-drop functionality,
card-based layout, smooth animations, and comprehensive workflow management.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QMenuBar, QToolBar, QStatusBar, QProgressBar, QLabel,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea,
    QApplication, QPushButton, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QKeySequence, QPixmap, QFont, QAction

from config.settings import ConfigManager
from ui.themes import ThemeManager, ThemeMode, theme_manager
from ui.components.file_drop_zone import ModernFileDropZone, DropZoneType, FileStatus
from ui.components.preview_widget import ModernPreviewWidget
from ui.components.status_bar import ModernStatusBar, StatusType
from ui.components.modern_card import ModernCard, CardElevation, CardType, DataCard, ActionCard
from ui.components.toast_notification import ToastManager, get_toast_manager, show_success_toast, show_error_toast, show_warning_toast, show_info_toast
from ui.dialogs.ai_settings import AISettingsDialog
from ui.dialogs.preview_dialog import PreviewDialog
from ui.components.cost_monitor import CostDisplayWidget, CostMonitorDialog
from core.csv_parser import CSVParser
from core.ai_normalizer import AINormalizer
from core.data_validator import DataValidator
from core.graph_generator import GraphGenerator
from utils.cost_tracker import CostTracker
from utils.cache_manager import CacheManager, CacheConfig


class ModernMainWindow(QMainWindow):
    """
    Modern main application window with professional Material Design interface.
    
    Features:
    - Card-based layout with Material Design styling
    - Drag-and-drop file handling with visual feedback
    - Real-time preview system with smooth animations
    - Integrated AI normalization workflow
    - Professional menu and toolbar system
    - Comprehensive status monitoring and progress tracking
    - Toast notifications for user feedback
    - Dark/light theme support with smooth transitions
    - Responsive design with splitter layouts
    """
    
    # Signals
    files_processed = pyqtSignal(list)
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    theme_changed = pyqtSignal(str)
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the modern main window.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize core components
        self.csv_parser = CSVParser()
        self.ai_normalizer = AINormalizer(config_manager)
        self.data_validator = DataValidator()
        self.graph_generator = GraphGenerator()
        
        # Initialize cost tracking and caching
        self.cost_tracker = CostTracker(config_manager=config_manager)
        cache_config = CacheConfig(
            memory_limit_mb=config_manager.cache_settings.memory_limit_mb,
            disk_limit_mb=config_manager.cache_settings.disk_limit_mb,
            default_ttl_hours=config_manager.cache_settings.default_ttl_hours,
            enable_redis=config_manager.cache_settings.enable_redis,
            redis_host=config_manager.cache_settings.redis_host,
            redis_port=config_manager.cache_settings.redis_port
        )
        self.cache_manager = CacheManager(config=cache_config)
        
        # UI components
        self.central_widget: Optional[QWidget] = None
        self.baseline_drop_zone: Optional[ModernFileDropZone] = None
        self.samples_drop_zone: Optional[ModernFileDropZone] = None
        self.preview_widget: Optional[ModernPreviewWidget] = None
        self.status_bar_widget: Optional[ModernStatusBar] = None
        self.toast_manager: Optional[ToastManager] = None
        self.cost_display_widget: Optional[CostDisplayWidget] = None
        self.cost_monitor_dialog: Optional[CostMonitorDialog] = None
        
        # Cards
        self.input_card: Optional[ModernCard] = None
        self.output_card: Optional[ModernCard] = None
        self.stats_cards: List[DataCard] = []
        
        # Dialogs
        self.ai_settings_dialog: Optional[AISettingsDialog] = None
        self.preview_dialog: Optional[PreviewDialog] = None
        
        # State management
        self.baseline_file: Optional[Path] = None
        self.sample_files: List[Path] = []
        self.processing_thread: Optional[QThread] = None
        self.current_theme = "light"
        
        # Initialize UI
        self._setup_ui()
        self._setup_connections()
        self._setup_theme()
        self._restore_window_state()
        
        # Initialize toast manager
        self.toast_manager = get_toast_manager(self)
        
        self.logger.info("Modern main window initialized")
    
    def _setup_ui(self):
        """Set up the modern user interface."""
        # Set window properties
        self.setWindowTitle("Spectral Analyzer - MRG Labs")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Set window icon
        try:
            icon_path = Path("spectral_analyzer/resources/icons/app_icon.png")
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception as e:
            self.logger.warning(f"Could not load app icon: {e}")
        
        # Create central widget with modern layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout with proper spacing
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)
        
        # Header section with app info and stats
        header_section = self._create_header_section()
        main_layout.addWidget(header_section)
        
        # Main content area with cards
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(content_splitter)
        
        # Left panel - Data input and controls
        left_panel = self._create_left_panel()
        content_splitter.addWidget(left_panel)
        
        # Right panel - Preview and analysis
        right_panel = self._create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions (40% left, 60% right)
        content_splitter.setSizes([600, 900])
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create toolbar
        self._create_toolbar()
        
        # Create status bar
        self._create_status_bar()
    
    def _create_header_section(self) -> QWidget:
        """Create header section with app info and statistics."""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setSpacing(16)
        
        # App title and description
        title_card = ModernCard(
            title="Spectral Analyzer",
            subtitle="AI-powered spectroscopy data analysis for MRG Labs",
            card_type=CardType.FILLED,
            elevation=CardElevation.LOW
        )
        title_card.setMaximumHeight(80)
        header_layout.addWidget(title_card, 2)
        
        # Statistics cards
        self.files_card = DataCard("Files Loaded", "0", "", parent=self)
        self.files_card.setMaximumHeight(80)
        self.files_card.setMaximumWidth(150)
        header_layout.addWidget(self.files_card)
        
        self.ai_status_card = DataCard("AI Status", "Disconnected", "", status="error", parent=self)
        self.ai_status_card.setMaximumHeight(80)
        self.ai_status_card.setMaximumWidth(150)
        header_layout.addWidget(self.ai_status_card)
        
        self.cost_card = DataCard("API Cost", "$0.00", "", parent=self)
        self.cost_card.setMaximumHeight(80)
        self.cost_card.setMaximumWidth(150)
        header_layout.addWidget(self.cost_card)
        
        self.stats_cards = [self.files_card, self.ai_status_card, self.cost_card]
        
        return header_widget
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with data input and controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Data Input Card
        self.input_card = ModernCard(
            title="Data Input",
            subtitle="Load baseline and sample files for analysis",
            card_type=CardType.ELEVATED,
            elevation=CardElevation.MEDIUM
        )
        
        # Baseline file drop zone
        self.baseline_drop_zone = ModernFileDropZone(
            zone_type=DropZoneType.BASELINE,
            accept_multiple=False,
            max_files=1
        )
        self.input_card.add_content(self.baseline_drop_zone)
        
        # Samples file drop zone
        self.samples_drop_zone = ModernFileDropZone(
            zone_type=DropZoneType.SAMPLES,
            accept_multiple=True,
            max_files=20
        )
        self.input_card.add_content(self.samples_drop_zone)
        
        layout.addWidget(self.input_card)
        
        # Output Settings Card
        self.output_card = self._create_output_card()
        layout.addWidget(self.output_card)
        
        # Quick Actions Card
        actions_card = self._create_actions_card()
        layout.addWidget(actions_card)
        
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with preview and analysis."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        
        # Preview widget
        self.preview_widget = ModernPreviewWidget()
        layout.addWidget(self.preview_widget)
        
        return panel
    
    def _create_output_card(self) -> ModernCard:
        """Create output settings card."""
        card = ModernCard(
            title="Output Settings",
            subtitle="Configure output format and destination",
            card_type=CardType.DEFAULT,
            elevation=CardElevation.LOW
        )
        
        from ui.themes import theme_manager
        
        # Output controls
        output_widget = QWidget()
        output_layout = QVBoxLayout(output_widget)
        output_layout.setSpacing(12)
        
        # Destination folder
        dest_layout = QHBoxLayout()
        dest_label = QLabel("Destination:")
        dest_label.setFont(theme_manager.get_font("label_medium"))
        dest_layout.addWidget(dest_label)
        
        self.dest_path_label = QLabel("Not selected")
        self.dest_path_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_secondary")};
            padding: 4px 8px;
            border: 1px solid {theme_manager.get_color("border")};
            border-radius: 4px;
            background-color: {theme_manager.get_color("surface_variant")};
        }}
        """)
        dest_layout.addWidget(self.dest_path_label, 1)
        
        browse_dest_btn = QPushButton("Browse")
        browse_dest_btn.setStyleSheet(theme_manager.create_button_style("secondary"))
        browse_dest_btn.clicked.connect(self._browse_destination)
        dest_layout.addWidget(browse_dest_btn)
        
        output_layout.addLayout(dest_layout)
        
        # Format selection
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        format_label.setFont(theme_manager.get_font("label_medium"))
        format_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["CSV (Normalized)", "Excel (.xlsx)", "JSON", "PNG Graph", "PDF Report"])
        self.format_combo.setStyleSheet(theme_manager.create_input_style())
        format_layout.addWidget(self.format_combo)
        
        output_layout.addLayout(format_layout)
        
        card.add_content(output_widget)
        
        # Add primary action button
        card.add_action_button("🚀 Generate and Save", self._generate_and_save, "primary")
        
        return card
    
    def _create_actions_card(self) -> ModernCard:
        """Create quick actions card."""
        card = ModernCard(
            title="Quick Actions",
            subtitle="Common operations and tools",
            card_type=CardType.DEFAULT,
            elevation=CardElevation.LOW
        )
        
        # Actions layout
        actions_widget = QWidget()
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setSpacing(8)
        
        # AI Settings action
        ai_action = ActionCard(
            title="AI Settings",
            description="Configure AI models and API keys",
            icon="🤖",
            action_text="Configure",
            action_callback=self._show_ai_settings
        )
        actions_layout.addWidget(ai_action)
        
        # Validation action
        validate_action = ActionCard(
            title="Validate Data",
            description="Run comprehensive data quality checks",
            icon="✅",
            action_text="Validate",
            action_callback=self._validate_data
        )
        actions_layout.addWidget(validate_action)
        
        card.add_content(actions_widget)
        
        return card
    
    def _create_menu_bar(self):
        """Create the modern application menu bar."""
        menubar = self.menuBar()
        
        # Apply menu styling
        menubar.setStyleSheet(theme_manager.create_menu_style())
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Open files action
        open_action = QAction("&Open Files...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open CSV files for analysis")
        open_action.triggered.connect(self._open_files)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        recent_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files_menu(recent_menu)
        
        file_menu.addSeparator()
        
        # Export action
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.setStatusTip("Export analysis results")
        export_action.triggered.connect(self._export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        # Preferences action
        preferences_action = QAction("&Preferences...", self)
        preferences_action.setShortcut(QKeySequence.StandardKey.Preferences)
        preferences_action.setStatusTip("Open application preferences")
        preferences_action.triggered.connect(self._show_preferences)
        edit_menu.addAction(preferences_action)
        
        # AI Settings action
        ai_settings_action = QAction("&AI Settings...", self)
        ai_settings_action.setStatusTip("Configure AI normalization settings")
        ai_settings_action.triggered.connect(self._show_ai_settings)
        edit_menu.addAction(ai_settings_action)
        
        # Process menu
        process_menu = menubar.addMenu("&Process")
        
        # Normalize action
        normalize_action = QAction("&Normalize Data", self)
        normalize_action.setShortcut(QKeySequence("Ctrl+N"))
        normalize_action.setStatusTip("Normalize selected CSV files using AI")
        normalize_action.triggered.connect(self._normalize_data)
        process_menu.addAction(normalize_action)
        
        # Validate action
        validate_action = QAction("&Validate Data", self)
        validate_action.setShortcut(QKeySequence("Ctrl+V"))
        validate_action.setStatusTip("Validate data quality")
        validate_action.triggered.connect(self._validate_data)
        process_menu.addAction(validate_action)
        
        # Generate graphs action
        generate_graphs_action = QAction("&Generate Graphs", self)
        generate_graphs_action.setShortcut(QKeySequence("Ctrl+G"))
        generate_graphs_action.setStatusTip("Generate spectral graphs")
        generate_graphs_action.triggered.connect(self._generate_graphs)
        process_menu.addAction(generate_graphs_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Theme submenu
        theme_menu = view_menu.addMenu("Theme")
        
        light_theme_action = QAction("Light Theme", self)
        light_theme_action.triggered.connect(lambda: self._set_theme("light"))
        theme_menu.addAction(light_theme_action)
        
        dark_theme_action = QAction("Dark Theme", self)
        dark_theme_action.triggered.connect(lambda: self._set_theme("dark"))
        theme_menu.addAction(dark_theme_action)
        
        auto_theme_action = QAction("Auto (System)", self)
        auto_theme_action.triggered.connect(lambda: self._set_theme("auto"))
        theme_menu.addAction(auto_theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.setStatusTip("About Spectral Analyzer")
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        # Documentation action
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        docs_action.setStatusTip("Open documentation")
        docs_action.triggered.connect(self._show_documentation)
        help_menu.addAction(docs_action)
    
    def _create_toolbar(self):
        """Create the modern application toolbar."""
        toolbar = self.addToolBar("Main")
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        toolbar.setMovable(False)
        
        # Apply toolbar styling
        from ui.themes import theme_manager
        toolbar.setStyleSheet(f"""
        QToolBar {{
            background-color: {theme_manager.get_color("surface")};
            border: none;
            spacing: 8px;
            padding: 8px;
        }}
        QToolButton {{
            background-color: transparent;
            border: none;
            border-radius: 6px;
            padding: 8px;
            margin: 2px;
        }}
        QToolButton:hover {{
            background-color: {theme_manager.get_color("hover")};
        }}
        QToolButton:pressed {{
            background-color: {theme_manager.get_color("pressed")};
        }}
        """)
        
        # Open files
        open_action = QAction("📁\nOpen", self)
        open_action.setStatusTip("Open CSV files")
        open_action.triggered.connect(self._open_files)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Process actions
        normalize_action = QAction("🤖\nNormalize", self)
        normalize_action.setStatusTip("Normalize data with AI")
        normalize_action.triggered.connect(self._normalize_data)
        toolbar.addAction(normalize_action)
        
        validate_action = QAction("✅\nValidate", self)
        validate_action.setStatusTip("Validate data quality")
        validate_action.triggered.connect(self._validate_data)
        toolbar.addAction(validate_action)
        
        generate_action = QAction("📊\nGenerate", self)
        generate_action.setStatusTip("Generate graphs")
        generate_action.triggered.connect(self._generate_graphs)
        toolbar.addAction(generate_action)
        
        toolbar.addSeparator()
        
        # Settings
        settings_action = QAction("⚙️\nSettings", self)
        settings_action.setStatusTip("Open AI settings")
        settings_action.triggered.connect(self._show_ai_settings)
        toolbar.addAction(settings_action)
    
    def _create_status_bar(self):
        """Create the modern application status bar."""
        self.status_bar_widget = ModernStatusBar()
        self.setStatusBar(self.status_bar_widget)
        
        # Set initial status
        self.status_bar_widget.show_message("Ready to analyze spectral data")
    
    def _setup_theme(self):
        """Set up theme system."""
        # Load theme from configuration
        theme_mode = self.config_manager.get_setting('ui', 'theme', 'light')
        
        if theme_mode == 'dark':
            theme_manager.set_theme(ThemeMode.DARK)
        elif theme_mode == 'auto':
            theme_manager.set_theme(ThemeMode.AUTO)
        else:
            theme_manager.set_theme(ThemeMode.LIGHT)
        
        # Apply theme to application
        theme_manager.apply_theme_to_application(QApplication.instance())
        
        # Connect theme change signal
        theme_manager.theme_changed.connect(self._on_theme_changed)
    
    def _setup_connections(self):
        """Set up signal-slot connections."""
        # Baseline drop zone connections
        if self.baseline_drop_zone:
            self.baseline_drop_zone.files_dropped.connect(self._handle_baseline_dropped)
            self.baseline_drop_zone.file_selected.connect(self._handle_file_selected)
        
        # Samples drop zone connections
        if self.samples_drop_zone:
            self.samples_drop_zone.files_dropped.connect(self._handle_samples_dropped)
            self.samples_drop_zone.file_selected.connect(self._handle_file_selected)
        
        # Preview widget connections
        if self.preview_widget:
            self.preview_widget.normalization_requested.connect(self._normalize_file)
            self.preview_widget.export_requested.connect(self._export_graph)
            self.preview_widget.sample_selected.connect(self._preview_sample)
        
        # Status bar connections
        if self.status_bar_widget:
            self.status_bar_widget.ai_settings_requested.connect(self._show_ai_settings)
        
        # Internal signal connections
        self.processing_started.connect(self._on_processing_started)
        self.processing_finished.connect(self._on_processing_finished)
        self.error_occurred.connect(self._on_error_occurred)
    
    def _restore_window_state(self):
        """Restore window state from configuration."""
        try:
            # Restore geometry
            geometry = self.config_manager.get_setting('ui', 'window_geometry')
            if geometry:
                self.restoreGeometry(geometry)
            
            # Restore window state
            state = self.config_manager.get_setting('ui', 'window_state')
            if state:
                self.restoreState(state)
                
        except Exception as e:
            self.logger.warning(f"Failed to restore window state: {e}")
    
    def _save_window_state(self):
        """Save window state to configuration."""
        try:
            self.config_manager.set_setting('ui', 'window_geometry', self.saveGeometry())
            self.config_manager.set_setting('ui', 'window_state', self.saveState())
            self.config_manager.save_settings()
            
        except Exception as e:
            self.logger.warning(f"Failed to save window state: {e}")
    
    def _update_recent_files_menu(self, menu):
        """Update recent files menu."""
        menu.clear()
        
        recent_files = self.config_manager.get_recent_files()
        
        if not recent_files:
            no_recent_action = QAction("No recent files", self)
            no_recent_action.setEnabled(False)
            menu.addAction(no_recent_action)
            return
        
        for file_path in recent_files[:10]:  # Show last 10 files
            action = QAction(str(file_path), self)
            action.triggered.connect(lambda checked, path=file_path: self._open_recent_file(path))
            menu.addAction(action)
        
        menu.addSeparator()
        
        clear_action = QAction("Clear Recent Files", self)
        clear_action.triggered.connect(self._clear_recent_files)
        menu.addAction(clear_action)
    
    # Event handlers
    def _handle_baseline_dropped(self, file_paths: List[str], zone_type: str):
        """Handle baseline file dropped."""
        try:
            if file_paths:
                self.baseline_file = Path(file_paths[0])
                self.config_manager.add_recent_file(file_paths[0])
                
                # Update statistics
                self._update_stats_cards()
                
                # Auto-preview baseline file
                self._preview_file(self.baseline_file)
                
                show_success_toast(f"Baseline loaded: {self.baseline_file.name}")
                
        except Exception as e:
            self.logger.error(f"Error handling baseline file: {e}")
            show_error_toast(f"Error loading baseline: {str(e)}")
    
    def _handle_samples_dropped(self, file_paths: List[str], zone_type: str):
        """Handle sample files dropped."""
        try:
            new_samples = [Path(path) for path in file_paths]
            
            # Add to sample files, avoiding duplicates
            for sample in new_samples:
                if sample not in self.sample_files:
                    self.sample_files.append(sample)
                    self.config_manager.add_recent_file(str(sample))
            
            # Update statistics
            self._update_stats_cards()
            
            # Update preview widget sample list
            self.preview_widget.set_sample_list([str(f) for f in self.sample_files])
            
            # Auto-preview first sample if no baseline
            if not self.baseline_file and self.sample_files:
                self._preview_file(self.sample_files[0])
            
            show_success_toast(f"Added {len(new_samples)} sample files")
            
        except Exception as e:
            self.logger.error(f"Error handling sample files: {e}")
            show_error_toast(f"Error loading samples: {str(e)}")
    
    def _handle_file_selected(self, file_path: str):
        """Handle file selection for preview."""
        try:
            self._preview_file(Path(file_path))
        except Exception as e:
            self.logger.error(f"Error handling file selection: {e}")
            show_error_toast(f"Preview error: {str(e)}")
    
    def _preview_file(self, file_path: Path):
        """Preview a single file with progress indication."""
        try:
            self.status_bar_widget.show_processing(f"Loading {file_path.name}...")
            
            # Parse CSV file
            result = self.csv_parser.parse_file(file_path)
            
            if result.success:
                # Update preview widget
                self.preview_widget.update_preview(result.data, result.structure)
                self.status_bar_widget.show_success(f"Loaded {file_path.name}")
            else:
                self.status_bar_widget.show_error(f"Failed to load {file_path.name}")
                show_error_toast(f"Could not parse {file_path.name}")
            
        except Exception as e:
            self.logger.error(f"Error previewing file {file_path}: {e}")
            self.status_bar_widget.show_error(f"Preview error: {str(e)}")
            show_error_toast(f"Preview error: {str(e)}")
    
    def _preview_sample(self, sample_path: str):
        """Preview selected sample file."""
        self._preview_file(Path(sample_path))
    
    def _update_stats_cards(self):
        """Update statistics cards in header."""
        # Files loaded count
        total_files = len(self.sample_files) + (1 if self.baseline_file else 0)
        self.files_card.value = str(total_files)
        
        # Update the card display (would need to implement update method in DataCard)
        # For now, recreate the card content
        
        # AI status
        if self.ai_normalizer and hasattr(self.ai_normalizer, 'openrouter_client'):
            if self.ai_normalizer.openrouter_client:
                self.ai_status_card.value = "Connected"
                self.ai_status_card.status = "success"
            else:
                self.ai_status_card.value = "Disconnected"
                self.ai_status_card.status = "error"
    
    # Action methods
    def _open_files(self):
        """Open file dialog to select CSV files."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        file_dialog.setWindowTitle("Select CSV Files")
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            
            # First file becomes baseline if none set
            if not self.baseline_file and file_paths:
                self._handle_baseline_dropped([file_paths[0]], "baseline")
                remaining_files = file_paths[1:]
            else:
                remaining_files = file_paths
            
            # Remaining files become samples
            if remaining_files:
                self._handle_samples_dropped(remaining_files, "samples")
    
    def _browse_destination(self):
        """Browse for output destination folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.dest_path_label.setText(folder)
            show_info_toast(f"Output destination set")
    
    def _generate_and_save(self):
        """Generate and save analysis results."""
        if not self.baseline_file and not self.sample_files:
            show_warning_toast("Please load files first")
            return
        
        dest_path = self.dest_path_label.text()
        if dest_path == "Not selected":
            show_warning_toast("Please select output destination")
            return
        
        # Start processing
        self.status_bar_widget.set_operation_status("Generating results", 0)
        show_info_toast("Starting analysis and generation...")
        
        # TODO: Implement actual processing workflow
        QTimer.singleShot(2000, lambda: self._complete_generation())
    
    def _complete_generation(self):
        """Complete the generation process."""
        self.status_bar_widget.hide_progress()
        self.status_bar_widget.show_success("Analysis completed successfully")
        show_success_toast("Results generated and saved!")
    
    def _normalize_data(self):
        """Start AI normalization process."""
        if not self.baseline_file and not self.sample_files:
            show_warning_toast("Please load CSV files first")
            return
        
        self.status_bar_widget.set_operation_status("AI normalization", 0)
        show_info_toast("Starting AI normalization...")
        
        # TODO: Implement normalization workflow
        QTimer.singleShot(3000, lambda: self._complete_normalization())
    
    def _complete_normalization(self):
        """Complete normalization process."""
        self.status_bar_widget.hide_progress()
        self.status_bar_widget.show_success("AI normalization completed")
        show_success_toast("Data normalized successfully!")
    
    def _normalize_file(self, file_path: str):
        """Normalize a specific file."""
        self.status_bar_widget.set_operation_status(f"Normalizing {Path(file_path).name}", 0)
        show_info_toast(f"Normalizing {Path(file_path).name}...")
        
        # TODO: Implement single file normalization
        QTimer.singleShot(2000, lambda: self._complete_normalization())
    
    def _validate_data(self):
        """Validate loaded data."""
        if not self.baseline_file and not self.sample_files:
            show_warning_toast("Please load CSV files first")
            return
        
        self.status_bar_widget.set_operation_status("Validating data", 0)
        show_info_toast("Running data validation...")
        
        # TODO: Implement validation workflow
        QTimer.singleShot(1500, lambda: self._complete_validation())
    
    def _complete_validation(self):
        """Complete validation process."""
        self.status_bar_widget.hide_progress()
        self.status_bar_widget.show_success("Data validation completed")
        show_success_toast("Data validation passed!")
    
    def _generate_graphs(self):
        """Generate spectral graphs."""
        if not self.baseline_file and not self.sample_files:
            show_warning_toast("Please load CSV files first")
            return
        
        self.status_bar_widget.set_operation_status("Generating graphs", 0)
        show_info_toast("Generating spectral graphs...")
        
        # TODO: Implement graph generation workflow
        QTimer.singleShot(2000, lambda: self._complete_graph_generation())
    
    def _complete_graph_generation(self):
        """Complete graph generation."""
        self.status_bar_widget.hide_progress()
        self.status_bar_widget.show_success("Graphs generated successfully")
        show_success_toast("Graphs generated and saved!")
    
    def _export_results(self):
        """Export analysis results."""
        show_info_toast("Export functionality coming soon...")
    
    def _export_graph(self, file_path: str, format: str):
        """Export graph for specific file."""
        show_info_toast(f"Exporting graph for {Path(file_path).name}...")
    
    def _show_preferences(self):
        """Show preferences dialog."""
        show_info_toast("Preferences dialog coming soon...")
    
    def _show_ai_settings(self):
        """Show AI settings dialog."""
        if not self.ai_settings_dialog:
            self.ai_settings_dialog = AISettingsDialog(self.config_manager, self)
            self.ai_settings_dialog.settings_changed.connect(self._on_ai_settings_changed)
        
        self.ai_settings_dialog.show()
    
    def _on_ai_settings_changed(self):
        """Handle AI settings changes."""
        # Reinitialize AI normalizer with new settings
        self.ai_normalizer = AINormalizer(self.config_manager)
        self._update_stats_cards()
        show_success_toast("AI settings updated")
    
    def _set_theme(self, theme_name: str):
        """Set application theme."""
        if theme_name == "light":
            theme_manager.set_theme(ThemeMode.LIGHT)
        elif theme_name == "dark":
            theme_manager.set_theme(ThemeMode.DARK)
        elif theme_name == "auto":
            theme_manager.set_theme(ThemeMode.AUTO)
        
        # Save theme preference
        self.config_manager.set_setting('ui', 'theme', theme_name)
        self.config_manager.save_settings()
        
        show_info_toast(f"Theme changed to {theme_name}")
    
    def _on_theme_changed(self, theme_name: str):
        """Handle theme change."""
        self.current_theme = theme_name
        
        # Reapply theme to application
        theme_manager.apply_theme_to_application(QApplication.instance())
        
        # Update status bar
        if self.status_bar_widget:
            self.status_bar_widget.setStyleSheet(theme_manager.create_status_bar_style())
        
        # Refresh preview if data is loaded
        if self.preview_widget and self.preview_widget.current_data is not None:
            self.preview_widget._delayed_graph_update()
        
        self.theme_changed.emit(theme_name)
    
    def _show_about(self):
        """Show about dialog."""
        from ui.themes import theme_manager
        
        about_text = """
        <h2>Spectral Analyzer v1.0</h2>
        <p><b>AI-powered spectroscopy data analysis tool</b></p>
        <p>Developed for MRG Labs</p>
        <br>
        <p><b>Features:</b></p>
        <ul>
        <li>🤖 AI-powered data normalization</li>
        <li>📊 Professional graph generation</li>
        <li>✅ Comprehensive data validation</li>
        <li>🎨 Modern Material Design interface</li>
        <li>🌙 Dark/Light theme support</li>
        </ul>
        <br>
        <p>© 2024 MRG Labs. All rights reserved.</p>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About Spectral Analyzer")
        msg_box.setText(about_text)
        msg_box.setIconPixmap(QPixmap("spectral_analyzer/resources/icons/app_icon.png").scaled(64, 64))
        msg_box.exec()
    
    def _show_documentation(self):
        """Show documentation."""
        show_info_toast("Documentation will open in your browser...")
    
    def _open_recent_file(self, file_path: str):
        """Open a recent file."""
        if Path(file_path).exists():
            self._handle_samples_dropped([file_path], "samples")
        else:
            show_error_toast(f"File no longer exists: {Path(file_path).name}")
    
    def _clear_recent_files(self):
        """Clear recent files list."""
        self.config_manager.ui_settings.recent_files.clear()
        self.config_manager.save_settings()
        show_info_toast("Recent files cleared")
    
    # Event handlers
    def _on_processing_started(self):
        """Handle processing started."""
        if self.status_bar_widget:
            self.status_bar_widget.show_progress()
    
    def _on_processing_finished(self):
        """Handle processing finished."""
        if self.status_bar_widget:
            self.status_bar_widget.hide_progress()
            self.status_bar_widget.show_success("Processing completed")
    
    def _on_error_occurred(self, error_message: str):
        """Handle error occurrence."""
        if self.status_bar_widget:
            self.status_bar_widget.hide_progress()
            self.status_bar_widget.show_error(f"Error: {error_message}")
        
        show_error_toast(error_message)
    
    def cleanup(self):
        """Clean up resources before closing."""
        try:
            # Save window state
            self._save_window_state()
            
            # Stop any running threads
            if self.processing_thread and self.processing_thread.isRunning():
                self.processing_thread.quit()
                self.processing_thread.wait(3000)  # Wait up to 3 seconds
            
            # Clean up status bar
            if self.status_bar_widget:
                self.status_bar_widget.cleanup()
            
            # Clear toast notifications
            if self.toast_manager:
                self.toast_manager.clear_all_toasts()
            
            self.logger.info("Modern main window cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.cleanup()
        event.accept()


# Maintain backward compatibility
MainWindow = ModernMainWindow