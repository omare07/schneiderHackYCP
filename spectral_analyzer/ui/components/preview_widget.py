"""
Modern preview widget for real-time spectral data visualization.

Provides interactive preview of CSV data with professional graph display,
data statistics, normalization controls, and smooth animations.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QGroupBox, QGridLayout, QScrollArea, QSplitter, QTabWidget, QComboBox,
    QCheckBox, QSlider, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPixmap, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from core.csv_parser import CSVStructure
from core.graph_generator import (
    GraphGenerator, SpectralGraphGenerator, GraphConfig as LegacyGraphConfig,
    GraphConfig, PlotType, ColorScheme
)
from ui.components.modern_card import ModernCard, CardElevation, CardType
from ui.components.toast_notification import show_success_toast, show_error_toast, show_info_toast


class ModernPreviewWidget(QWidget):
    """
    Modern preview widget for spectral data visualization and analysis.
    
    Features:
    - Real-time graph preview with smooth updates
    - Interactive graph controls and customization
    - Data structure analysis with professional display
    - Statistics display with visual indicators
    - Normalization controls with AI integration
    - Sample selector for multi-file workflows
    - Professional Material Design styling
    """
    
    # Signals
    normalization_requested = pyqtSignal(str)  # File path
    export_requested = pyqtSignal(str, str)    # File path, format
    sample_selected = pyqtSignal(str)          # Sample file path
    graph_settings_changed = pyqtSignal(dict) # Graph configuration
    
    def __init__(self, parent=None):
        """Initialize the modern preview widget."""
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.graph_generator = GraphGenerator()  # Legacy generator for compatibility
        self.spectral_generator = SpectralGraphGenerator()  # New professional generator
        
        # Current data
        self.current_data: Optional[pd.DataFrame] = None
        self.current_structure: Optional[CSVStructure] = None
        self.current_file_path: Optional[str] = None
        self.sample_files: List[str] = []
        
        # Graph configuration (legacy)
        self.graph_config = LegacyGraphConfig()
        # Note: axis.invert_x will be set through the checkbox UI control
        
        # Professional graph configuration
        self.professional_config = GraphConfig()
        
        # UI components
        self.graph_canvas: Optional[FigureCanvas] = None
        self.graph_card: Optional[ModernCard] = None
        self.info_card: Optional[ModernCard] = None
        self.controls_card: Optional[ModernCard] = None
        self.sample_selector: Optional[QComboBox] = None
        
        # Animation timers
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_graph_update)
        
        self._setup_ui()
        
        self.logger.debug("Modern preview widget initialized")
    
    def _setup_ui(self):
        """Set up the modern UI structure."""
        from ui.themes import theme_manager
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Sample selector section
        self._create_sample_selector(layout)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(main_splitter)
        
        # Graph section (top)
        self.graph_card = self._create_graph_card()
        main_splitter.addWidget(self.graph_card)
        
        # Info section (bottom)
        info_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Data info card
        self.info_card = self._create_info_card()
        info_splitter.addWidget(self.info_card)
        
        # Controls card
        self.controls_card = self._create_controls_card()
        info_splitter.addWidget(self.controls_card)
        
        info_splitter.setSizes([400, 300])
        main_splitter.addWidget(info_splitter)
        
        # Set splitter proportions (70% graph, 30% info)
        main_splitter.setSizes([700, 300])
        
        # Initially show placeholder
        self._show_placeholder()
    
    def _create_sample_selector(self, layout: QVBoxLayout):
        """Create sample selector section."""
        from ui.themes import theme_manager
        
        selector_widget = QWidget()
        selector_layout = QHBoxLayout(selector_widget)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sample selector label
        selector_label = QLabel("Sample:")
        selector_label.setFont(theme_manager.get_font("label_medium"))
        selector_layout.addWidget(selector_label)
        
        # Sample dropdown
        self.sample_selector = QComboBox()
        self.sample_selector.setMinimumWidth(200)
        self.sample_selector.setStyleSheet(theme_manager.create_input_style())
        self.sample_selector.currentTextChanged.connect(self._on_sample_changed)
        selector_layout.addWidget(self.sample_selector)
        
        selector_layout.addStretch()
        
        # Initially hidden
        selector_widget.setVisible(False)
        self.sample_selector_widget = selector_widget
        
        layout.addWidget(selector_widget)
    
    def _create_graph_card(self) -> ModernCard:
        """Create the graph preview card."""
        card = ModernCard(
            title="Spectral Preview",
            subtitle="Real-time visualization of loaded data",
            card_type=CardType.ELEVATED,
            elevation=CardElevation.MEDIUM
        )
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.figure.patch.set_facecolor('white')
        self.graph_canvas = FigureCanvas(self.figure)
        
        # Graph controls toolbar
        controls_widget = self._create_graph_controls()
        
        # Add to card
        card.add_content(self.graph_canvas)
        card.add_content(controls_widget)
        
        return card
    
    def _create_graph_controls(self) -> QWidget:
        """Create graph control toolbar."""
        from ui.themes import theme_manager
        
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 8, 0, 0)
        
        # Plot type selector
        plot_type_label = QLabel("Plot Type:")
        plot_type_label.setFont(theme_manager.get_font("label_small"))
        controls_layout.addWidget(plot_type_label)
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([plot_type.value.title() for plot_type in PlotType])
        self.plot_type_combo.setCurrentText("Line")
        self.plot_type_combo.setStyleSheet(theme_manager.create_input_style())
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_changed)
        controls_layout.addWidget(self.plot_type_combo)
        
        controls_layout.addWidget(QLabel("|"))
        
        # Color scheme selector
        color_label = QLabel("Colors:")
        color_label.setFont(theme_manager.get_font("label_small"))
        controls_layout.addWidget(color_label)
        
        self.color_combo = QComboBox()
        self.color_combo.addItems([scheme.value.title() for scheme in ColorScheme])
        self.color_combo.setCurrentText("Default")
        self.color_combo.setStyleSheet(theme_manager.create_input_style())
        self.color_combo.currentTextChanged.connect(self._on_color_scheme_changed)
        controls_layout.addWidget(self.color_combo)
        
        controls_layout.addStretch()
        
        # Action buttons
        self.normalize_btn = QPushButton("🤖 Normalize with AI")
        self.normalize_btn.setStyleSheet(theme_manager.create_button_style("primary"))
        self.normalize_btn.setEnabled(False)
        self.normalize_btn.clicked.connect(self._request_normalization)
        controls_layout.addWidget(self.normalize_btn)
        
        self.export_btn = QPushButton("📊 Export Graph")
        self.export_btn.setStyleSheet(theme_manager.create_button_style("secondary"))
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_graph)
        controls_layout.addWidget(self.export_btn)
        
        return controls_widget
    
    def _create_info_card(self) -> ModernCard:
        """Create the data information card."""
        card = ModernCard(
            title="Data Information",
            subtitle="Structure analysis and statistics",
            card_type=CardType.DEFAULT,
            elevation=CardElevation.LOW
        )
        
        # Create tab widget for different info views
        tab_widget = QTabWidget()
        
        # Structure tab
        structure_tab = self._create_structure_tab()
        tab_widget.addTab(structure_tab, "📋 Structure")
        
        # Statistics tab
        stats_tab = self._create_statistics_tab()
        tab_widget.addTab(stats_tab, "📊 Statistics")
        
        # Issues tab
        issues_tab = self._create_issues_tab()
        tab_widget.addTab(issues_tab, "⚠️ Issues")
        
        # Apply tab styling
        from ui.themes import theme_manager
        tab_widget.setStyleSheet(theme_manager.create_tab_style())
        
        card.add_content(tab_widget)
        return card
    
    def _create_controls_card(self) -> ModernCard:
        """Create the controls card."""
        card = ModernCard(
            title="Graph Controls",
            subtitle="Customize visualization settings",
            card_type=CardType.DEFAULT,
            elevation=CardElevation.LOW
        )
        
        # Controls layout
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(12)
        
        # Graph options
        self._create_graph_options(controls_layout)
        
        # Data processing options
        self._create_processing_options(controls_layout)
        
        card.add_content(controls_widget)
        return card
    
    def _create_graph_options(self, layout: QVBoxLayout):
        """Create graph customization options."""
        from ui.themes import theme_manager
        
        # Graph options group
        options_group = QGroupBox("Display Options")
        options_group.setFont(theme_manager.get_font("label_medium"))
        options_layout = QVBoxLayout(options_group)
        
        # Grid toggle
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._on_grid_toggled)
        options_layout.addWidget(self.grid_check)
        
        # Legend toggle
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(True)
        self.legend_check.toggled.connect(self._on_legend_toggled)
        options_layout.addWidget(self.legend_check)
        
        # Invert X-axis toggle
        self.invert_x_check = QCheckBox("Invert X-axis (IR Convention)")
        self.invert_x_check.setChecked(True)
        self.invert_x_check.toggled.connect(self._on_invert_x_toggled)
        options_layout.addWidget(self.invert_x_check)
        
        layout.addWidget(options_group)
    
    def _create_processing_options(self, layout: QVBoxLayout):
        """Create data processing options."""
        from ui.themes import theme_manager
        
        # Processing options group
        processing_group = QGroupBox("Data Processing")
        processing_group.setFont(theme_manager.get_font("label_medium"))
        processing_layout = QVBoxLayout(processing_group)
        
        # Baseline correction
        self.baseline_check = QCheckBox("Baseline Correction")
        self.baseline_check.toggled.connect(self._on_baseline_toggled)
        processing_layout.addWidget(self.baseline_check)
        
        # Normalization
        self.normalize_check = QCheckBox("Normalize Data")
        self.normalize_check.toggled.connect(self._on_normalize_toggled)
        processing_layout.addWidget(self.normalize_check)
        
        # Smoothing
        smoothing_layout = QHBoxLayout()
        self.smoothing_check = QCheckBox("Smoothing:")
        self.smoothing_check.toggled.connect(self._on_smoothing_toggled)
        smoothing_layout.addWidget(self.smoothing_check)
        
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 20)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.setEnabled(False)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        smoothing_layout.addWidget(self.smoothing_slider)
        
        processing_layout.addLayout(smoothing_layout)
        
        layout.addWidget(processing_group)
    
    def _create_structure_tab(self) -> QWidget:
        """Create the data structure information tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        from ui.themes import theme_manager
        
        # Structure info text
        self.structure_text = QTextEdit()
        self.structure_text.setReadOnly(True)
        self.structure_text.setMaximumHeight(200)
        self.structure_text.setFont(theme_manager.get_font("monospace"))
        self.structure_text.setStyleSheet(theme_manager.create_input_style())
        layout.addWidget(self.structure_text)
        
        return tab
    
    def _create_statistics_tab(self) -> QWidget:
        """Create the statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        from ui.themes import theme_manager
        
        # Statistics display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setFont(theme_manager.get_font("monospace"))
        self.stats_text.setStyleSheet(theme_manager.create_input_style())
        layout.addWidget(self.stats_text)
        
        return tab
    
    def _create_issues_tab(self) -> QWidget:
        """Create the issues/validation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        from ui.themes import theme_manager
        
        # Issues display
        self.issues_text = QTextEdit()
        self.issues_text.setReadOnly(True)
        self.issues_text.setMaximumHeight(200)
        self.issues_text.setFont(theme_manager.get_font("monospace"))
        self.issues_text.setStyleSheet(theme_manager.create_input_style())
        layout.addWidget(self.issues_text)
        
        return tab
    
    def _show_placeholder(self):
        """Show placeholder content when no data is loaded."""
        # Clear graph
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        from ui.themes import theme_manager
        text_color = theme_manager.get_color("text_secondary")
        bg_color = theme_manager.get_color("card_background")
        
        ax.text(0.5, 0.5, '📊\n\nNo data loaded\n\nDrag and drop CSV files to preview',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, color=text_color,
                bbox=dict(boxstyle='round,pad=1', 
                         facecolor=theme_manager.get_color("surface_variant"), 
                         alpha=0.8, edgecolor='none'))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Set figure background
        self.figure.patch.set_facecolor(bg_color)
        self.graph_canvas.draw()
        
        # Clear info sections
        if hasattr(self, 'structure_text'):
            self.structure_text.setText("No data structure information available")
        if hasattr(self, 'stats_text'):
            self.stats_text.setText("No statistics available")
        if hasattr(self, 'issues_text'):
            self.issues_text.setText("No validation issues to report")
        
        # Disable controls
        self.normalize_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        
        # Hide sample selector
        if hasattr(self, 'sample_selector_widget'):
            self.sample_selector_widget.setVisible(False)
    
    def update_preview(self, data: pd.DataFrame, structure: CSVStructure):
        """
        Update preview with new data and smooth animations.
        
        Args:
            data: DataFrame containing spectral data
            structure: CSV structure analysis
        """
        try:
            self.logger.info(f"Updating preview for {structure.file_path}")
            
            self.current_data = data
            self.current_structure = structure
            self.current_file_path = str(structure.file_path)
            
            # Delay graph update for smooth performance
            self.update_timer.start(100)  # 100ms delay
            
            # Update information sections immediately
            self._update_structure_info(structure)
            self._update_statistics(data)
            self._update_issues_info(structure)
            
            # Enable controls
            self.normalize_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
            # Update card title
            self.graph_card.set_title(f"Preview: {Path(self.current_file_path).name}")
            
            self.logger.debug("Preview updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update preview: {e}")
            self._show_error(f"Preview update failed: {e}")
    
    def _delayed_graph_update(self):
        """Perform delayed graph update for smooth performance."""
        if self.current_data is not None:
            self._update_graph(self.current_data)
    
    def _update_graph(self, data: pd.DataFrame):
        """Update the graph display with modern styling."""
        try:
            from ui.themes import theme_manager
            
            # Clear previous plot
            self.figure.clear()
            
            # Set figure background
            bg_color = theme_manager.get_color("card_background")
            self.figure.patch.set_facecolor(bg_color)
            
            # Create new plot
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(bg_color)
            
            # Check for wavenumber and intensity columns
            if 'wavenumber' in data.columns:
                x_data = data['wavenumber']
                
                # Find intensity columns
                intensity_cols = [col for col in data.columns 
                                if col in ['absorbance', 'transmittance', 'intensity']]
                
                if intensity_cols:
                    # Apply current graph configuration
                    self.graph_config.title = f"Preview: {Path(self.current_file_path).name if self.current_file_path else 'Unknown'}"
                    
                    # Plot data based on selected plot type
                    if self.graph_config.plot_type == PlotType.LINE:
                        for i, col in enumerate(intensity_cols):
                            y_data = data[col]
                            color = theme_manager.get_color("primary") if i == 0 else f"C{i}"
                            ax.plot(x_data, y_data, color=color, linewidth=2, alpha=0.8, label=col.capitalize())
                    
                    # Apply styling
                    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, 
                                 color=theme_manager.get_color("text_primary"))
                    ax.set_ylabel(intensity_cols[0].capitalize(), fontsize=12,
                                 color=theme_manager.get_color("text_primary"))
                    ax.set_title(self.graph_config.title, fontsize=14, fontweight='bold',
                                color=theme_manager.get_color("text_primary"))
                    
                    # Grid
                    if self.graph_config.style.grid:
                        ax.grid(True, alpha=0.3, color=theme_manager.get_color("text_disabled"))
                    
                    # Legend
                    if self.graph_config.style.legend and len(intensity_cols) > 1:
                        ax.legend(fontsize=10)
                    
                    # Invert x-axis for IR spectroscopy convention
                    if self.graph_config.axis.invert_x:
                        ax.invert_xaxis()
                    
                    # Style ticks
                    ax.tick_params(colors=theme_manager.get_color("text_secondary"), labelsize=10)
                    
                    # Remove top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color(theme_manager.get_color("border"))
                    ax.spines['bottom'].set_color(theme_manager.get_color("border"))
                    
                else:
                    self._show_graph_error("No intensity data found")
            else:
                self._show_graph_error("No wavenumber data found")
            
            # Refresh canvas with animation
            self.figure.tight_layout(pad=2.0)
            self.graph_canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Failed to update graph: {e}")
            self._show_graph_error(f"Graph error: {e}")
    
    def _show_graph_error(self, message: str):
        """Show error message in graph area."""
        from ui.themes import theme_manager
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        bg_color = theme_manager.get_color("card_background")
        error_color = theme_manager.get_color("error")
        
        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.5, f'❌\n\nGraph Error\n\n{message}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color=error_color,
                bbox=dict(boxstyle='round,pad=1', 
                         facecolor=theme_manager.get_color("surface_variant"), 
                         alpha=0.8, edgecolor=error_color))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.figure.patch.set_facecolor(bg_color)
        self.graph_canvas.draw()
    
    def _update_structure_info(self, structure: CSVStructure):
        """Update structure information display."""
        try:
            info_lines = [
                f"📁 File: {structure.file_path.name}",
                f"🔧 Format: {structure.format_type.value}",
                f"📝 Encoding: {structure.encoding}",
                f"🔗 Delimiter: '{structure.delimiter}'",
                f"📋 Has Header: {structure.has_header}",
                f"📊 Rows: {structure.row_count:,}",
                f"📈 Columns: {structure.column_count}",
                f"🎯 Confidence: {structure.confidence:.1f}%",
                "",
                "📋 Column Information:"
            ]
            
            for col in structure.columns:
                info_lines.append(
                    f"  {col.index}: {col.name} ({col.data_type.value}) "
                    f"- Confidence: {col.confidence:.2f}"
                )
                if col.numeric_range:
                    info_lines.append(f"    📏 Range: {col.numeric_range[0]:.3f} to {col.numeric_range[1]:.3f}")
            
            self.structure_text.setText("\n".join(info_lines))
            
        except Exception as e:
            self.logger.error(f"Failed to update structure info: {e}")
            self.structure_text.setText(f"❌ Error displaying structure info: {e}")
    
    def _update_statistics(self, data: pd.DataFrame):
        """Update statistics display with visual indicators."""
        try:
            stats_lines = [
                f"📊 Data Shape: {data.shape[0]:,} rows × {data.shape[1]} columns",
                f"💾 Memory Usage: {data.memory_usage(deep=True).sum() / 1024:.1f} KB",
                "",
                "📈 Column Statistics:"
            ]
            
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        stats_lines.extend([
                            f"  📊 {col}:",
                            f"    📝 Count: {len(col_data):,}",
                            f"    📊 Mean: {col_data.mean():.6f}",
                            f"    📏 Std: {col_data.std():.6f}",
                            f"    ⬇️ Min: {col_data.min():.6f}",
                            f"    ⬆️ Max: {col_data.max():.6f}",
                            f"    ❓ Missing: {data[col].isnull().sum():,}",
                            ""
                        ])
                else:
                    stats_lines.extend([
                        f"  📝 {col}: (non-numeric)",
                        f"    🔢 Unique values: {data[col].nunique():,}",
                        f"    ❓ Missing: {data[col].isnull().sum():,}",
                        ""
                    ])
            
            self.stats_text.setText("\n".join(stats_lines))
            
        except Exception as e:
            self.logger.error(f"Failed to update statistics: {e}")
            self.stats_text.setText(f"❌ Error calculating statistics: {e}")
    
    def _update_issues_info(self, structure: CSVStructure):
        """Update issues/validation information."""
        try:
            if structure.issues:
                issues_text = "⚠️ Data Issues Found:\n\n"
                for i, issue in enumerate(structure.issues, 1):
                    issues_text += f"{i}. {issue}\n"
            else:
                issues_text = "✅ No structural issues detected.\n\n"
                issues_text += "ℹ️ Note: This is a basic structure analysis. "
                issues_text += "Run full validation for comprehensive quality checks."
            
            # Add format-specific recommendations
            if structure.format_type.value == "unknown":
                issues_text += "\n\n💡 Recommendations:\n"
                issues_text += "• Consider using AI normalization to identify column types\n"
                issues_text += "• Verify that this is spectroscopy data\n"
                issues_text += "• Check file format and column naming conventions"
            
            self.issues_text.setText(issues_text)
            
        except Exception as e:
            self.logger.error(f"Failed to update issues info: {e}")
            self.issues_text.setText(f"❌ Error displaying issues: {e}")
    
    def _show_error(self, message: str):
        """Show error message in all sections."""
        error_text = f"❌ Error: {message}"
        
        if hasattr(self, 'structure_text'):
            self.structure_text.setText(error_text)
        if hasattr(self, 'stats_text'):
            self.stats_text.setText(error_text)
        if hasattr(self, 'issues_text'):
            self.issues_text.setText(error_text)
        
        self._show_graph_error(message)
        show_error_toast(f"Preview error: {message}")
    
    def set_sample_list(self, samples: List[str]):
        """Set the list of available samples."""
        self.sample_files = samples
        self.sample_selector.clear()
        
        if samples:
            for sample in samples:
                self.sample_selector.addItem(Path(sample).name, sample)
            self.sample_selector_widget.setVisible(True)
        else:
            self.sample_selector_widget.setVisible(False)
    
    def _on_sample_changed(self, sample_name: str):
        """Handle sample selection change."""
        current_data = self.sample_selector.currentData()
        if current_data:
            self.sample_selected.emit(current_data)
    
    def _on_plot_type_changed(self, plot_type_name: str):
        """Handle plot type change."""
        try:
            plot_type = PlotType(plot_type_name.lower())
            self.graph_config.plot_type = plot_type
            self._delayed_graph_update()
        except ValueError:
            self.logger.warning(f"Unknown plot type: {plot_type_name}")
    
    def _on_color_scheme_changed(self, scheme_name: str):
        """Handle color scheme change."""
        try:
            color_scheme = ColorScheme(scheme_name.lower())
            self.graph_config.style.color_scheme = color_scheme
            self._delayed_graph_update()
        except ValueError:
            self.logger.warning(f"Unknown color scheme: {scheme_name}")
    
    def _on_grid_toggled(self, checked: bool):
        """Handle grid toggle."""
        self.graph_config.style.grid = checked
        self._delayed_graph_update()
    
    def _on_legend_toggled(self, checked: bool):
        """Handle legend toggle."""
        self.graph_config.style.legend = checked
        self._delayed_graph_update()
    
    def _on_invert_x_toggled(self, checked: bool):
        """Handle X-axis inversion toggle."""
        self.graph_config.axis.invert_x = checked
        self._delayed_graph_update()
    
    def _on_baseline_toggled(self, checked: bool):
        """Handle baseline correction toggle."""
        self.graph_config.baseline_correction = checked
        self._delayed_graph_update()
    
    def _on_normalize_toggled(self, checked: bool):
        """Handle data normalization toggle."""
        self.graph_config.normalization = "min_max" if checked else None
        self._delayed_graph_update()
    
    def _on_smoothing_toggled(self, checked: bool):
        """Handle smoothing toggle."""
        self.smoothing_slider.setEnabled(checked)
        if checked:
            self._on_smoothing_changed(self.smoothing_slider.value())
        else:
            # Remove smoothing
            self._delayed_graph_update()
    
    def _on_smoothing_changed(self, value: int):
        """Handle smoothing parameter change."""
        # Apply smoothing to current data
        self._delayed_graph_update()
    
    def _browse_files(self):
        """Open file dialog to browse for files."""
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        file_dialog.setWindowTitle("Select CSV Files")
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            # Process through the same pipeline as drag-drop
            show_info_toast(f"Selected {len(file_paths)} files")
    
    def _clear_files(self):
        """Clear all files."""
        self.current_files = []
        self.file_statuses = {}
        self._update_status()
        self.files_cleared.emit()
    
    def _request_normalization(self):
        """Request AI normalization for current file."""
        if self.current_file_path:
            self.normalization_requested.emit(self.current_file_path)
            show_info_toast("AI normalization requested...")
    
    def _export_graph(self):
        """Export current graph."""
        if self.current_file_path:
            self.export_requested.emit(self.current_file_path, "png")
            show_info_toast("Graph export requested...")
    
    def clear_preview(self):
        """Clear the preview display."""
        self.current_data = None
        self.current_structure = None
        self.current_file_path = None
        self.sample_files.clear()
        self._show_placeholder()
    
    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get the currently displayed data."""
        return self.current_data
    
    def get_current_structure(self) -> Optional[CSVStructure]:
        """Get the current structure information."""
        return self.current_structure


# Maintain backward compatibility
PreviewWidget = ModernPreviewWidget