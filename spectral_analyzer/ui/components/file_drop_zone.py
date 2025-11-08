"""
Modern file drop zone widget with professional drag-and-drop functionality.

Provides an intuitive interface for users to drag and drop CSV files
with visual feedback, animations, and comprehensive file management.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QFrame, QSizePolicy, QFileDialog, QMessageBox, QProgressBar,
    QScrollArea, QGraphicsOpacityEffect
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QMimeData, QUrl, QPropertyAnimation, QEasingCurve,
    QTimer, QRect, QSize
)
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QFont, QPalette, QPainter, QPainterPath,
    QColor, QPixmap, QIcon
)

from ui.components.modern_card import ModernCard, CardElevation, CardType
from ui.components.toast_notification import show_success_toast, show_error_toast, show_warning_toast


class DropZoneType(Enum):
    """Drop zone types for different file handling."""
    BASELINE = "baseline"
    SAMPLES = "samples"
    GENERAL = "general"


class FileStatus(Enum):
    """File processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    AI_NORMALIZED = "ai_normalized"
    ERROR = "error"


class ModernFileDropZone(ModernCard):
    """
    Modern drag-and-drop file zone with professional styling.
    
    Features:
    - Visual drag-and-drop interface with animations
    - File list management with status indicators
    - CSV file validation and preview
    - Multiple drop zone types (baseline, samples)
    - Progress tracking and status updates
    - Professional Material Design styling
    """
    
    # Signals
    files_dropped = pyqtSignal(list, str)  # List of file paths, zone type
    file_selected = pyqtSignal(str)        # Selected file path
    files_cleared = pyqtSignal()
    file_status_changed = pyqtSignal(str, str)  # file path, status
    
    def __init__(self, zone_type: DropZoneType = DropZoneType.GENERAL,
                 accept_multiple: bool = True, max_files: int = 10,
                 parent=None):
        """
        Initialize the modern file drop zone.
        
        Args:
            zone_type: Type of drop zone (baseline, samples, general)
            accept_multiple: Whether to accept multiple files
            max_files: Maximum number of files allowed
            parent: Parent widget
        """
        # Set instance variables BEFORE calling super().__init__()
        self.logger = logging.getLogger(__name__)
        self.zone_type = zone_type
        self.accept_multiple = accept_multiple
        self.max_files = max_files
        
        # Initialize card with appropriate title
        title = self._get_zone_title(zone_type)
        subtitle = self._get_zone_subtitle(zone_type, accept_multiple)
        
        super().__init__(
            title=title,
            subtitle=subtitle,
            card_type=CardType.OUTLINED,
            elevation=CardElevation.LOW,
            parent=parent
        )
        
        # File management
        self.current_files: List[Path] = []
        self.file_statuses: Dict[str, FileStatus] = {}
        
        # Animation properties
        self.drop_animation = None
        self.pulse_animation = None
        self.is_drag_active = False
        
        # UI components
        self.drop_area = None
        self.file_list = None
        self.progress_bar = None
        self.status_label = None
        
        self._setup_ui()
        self._setup_drag_drop()
        self._setup_animations()
        
        self.logger.debug(f"Modern file drop zone initialized: {zone_type.value}")
    
    def _get_zone_title(self, zone_type: DropZoneType) -> str:
        """Get title based on zone type."""
        titles = {
            DropZoneType.BASELINE: "Baseline File",
            DropZoneType.SAMPLES: "Sample Files",
            DropZoneType.GENERAL: "CSV Files"
        }
        return titles.get(zone_type, "Files")
    
    def _get_zone_subtitle(self, zone_type: DropZoneType, accept_multiple: bool) -> str:
        """Get subtitle based on zone type and settings."""
        if zone_type == DropZoneType.BASELINE:
            return "Drop your baseline CSV file here"
        elif zone_type == DropZoneType.SAMPLES:
            return "Drop your sample CSV files here (multiple files supported)"
        else:
            if accept_multiple:
                return "Drop CSV files here or click to browse"
            else:
                return "Drop a CSV file here or click to browse"
    
    def _setup_ui(self):
        """Set up the modern UI structure."""
        # Call parent's _setup_ui first to initialize content_layout
        super()._setup_ui()
        
        from ui.themes import theme_manager
        
        # Drop area
        self.drop_area = self._create_drop_area()
        self.add_content(self.drop_area)
        
        # File list section
        self._create_file_list_section()
        
        # Progress section
        self._create_progress_section()
        
        # Control buttons
        self._create_control_buttons()
        
        # Status label
        self.status_label = QLabel("Ready to accept files")
        self.status_label.setFont(theme_manager.get_font("body_small"))
        self.status_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_secondary")};
            padding: 4px 0px;
        }}
        """)
        self.add_content(self.status_label)
    
    def _create_drop_area(self) -> QFrame:
        """Create the visual drop area."""
        from ui.themes import theme_manager
        
        drop_area = QFrame()
        drop_area.setMinimumHeight(120)
        drop_area.setFrameStyle(QFrame.Shape.NoFrame)
        
        # Layout for drop area content
        drop_layout = QVBoxLayout(drop_area)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.setSpacing(8)
        
        # Drop icon
        icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setFont(theme_manager.get_font("display_medium"))
        icon_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_secondary")};
        }}
        """)
        drop_layout.addWidget(icon_label)
        
        # Instructions
        instruction_text = "Drag and drop files here"
        if self.zone_type == DropZoneType.BASELINE:
            instruction_text = "Drop baseline CSV file here"
        elif self.zone_type == DropZoneType.SAMPLES:
            instruction_text = "Drop sample CSV files here"
        
        instruction_label = QLabel(instruction_text)
        instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction_label.setFont(theme_manager.get_font("body_medium"))
        instruction_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_secondary")};
        }}
        """)
        instruction_label.setWordWrap(True)
        drop_layout.addWidget(instruction_label)
        
        # Browse link
        browse_label = QLabel("or <a href='#'>click to browse</a>")
        browse_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        browse_label.setFont(theme_manager.get_font("body_small"))
        browse_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_secondary")};
        }}
        QLabel a {{
            color: {theme_manager.get_color("primary")};
            text-decoration: none;
        }}
        QLabel a:hover {{
            text-decoration: underline;
        }}
        """)
        browse_label.linkActivated.connect(self._browse_files)
        drop_layout.addWidget(browse_label)
        
        # Supported formats
        format_label = QLabel("Supported: CSV files")
        format_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        format_label.setFont(theme_manager.get_font("label_small"))
        format_label.setStyleSheet(f"""
        QLabel {{
            color: {theme_manager.get_color("text_disabled")};
        }}
        """)
        drop_layout.addWidget(format_label)
        
        # Apply initial styling
        self._update_drop_area_style(drop_area, False, False)
        
        return drop_area
    
    def _create_file_list_section(self):
        """Create file list section."""
        from ui.themes import theme_manager
        
        # File list with custom styling
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        self.file_list.setMinimumHeight(80)
        self.file_list.itemClicked.connect(self._on_file_selected)
        self.file_list.setStyleSheet(f"""
        QListWidget {{
            background-color: {theme_manager.get_color("surface_variant")};
            border: 1px solid {theme_manager.get_color("border")};
            border-radius: 6px;
            padding: 4px;
        }}
        QListWidget::item {{
            padding: 8px;
            border-radius: 4px;
            margin: 1px;
        }}
        QListWidget::item:selected {{
            background-color: {theme_manager.get_color("primary")};
            color: {theme_manager.get_color("on_primary")};
        }}
        QListWidget::item:hover {{
            background-color: {theme_manager.get_color("hover")};
        }}
        """)
        
        # Initially hidden
        self.file_list.setVisible(False)
        self.add_content(self.file_list)
    
    def _create_progress_section(self):
        """Create progress tracking section."""
        from ui.themes import theme_manager
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
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
        self.add_content(self.progress_bar)
    
    def _create_control_buttons(self):
        """Create control buttons."""
        from ui.themes import theme_manager
        
        # Button container
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 8, 0, 0)
        
        # Browse button
        self.browse_btn = QPushButton("Browse Files")
        self.browse_btn.setStyleSheet(theme_manager.create_button_style("secondary"))
        self.browse_btn.clicked.connect(self._browse_files)
        button_layout.addWidget(self.browse_btn)
        
        button_layout.addStretch()
        
        # Clear button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet(theme_manager.create_button_style("text"))
        self.clear_btn.clicked.connect(self._clear_files)
        self.clear_btn.setEnabled(False)
        button_layout.addWidget(self.clear_btn)
        
        self.add_content(button_widget)
    
    def _setup_drag_drop(self):
        """Set up drag and drop functionality."""
        self.setAcceptDrops(True)
        if self.drop_area:
            self.drop_area.setAcceptDrops(True)
    
    def _setup_animations(self):
        """Set up animations for visual feedback."""
        # Drop animation for visual feedback
        self.drop_animation = QPropertyAnimation(self.drop_area, b"geometry")
        self.drop_animation.setDuration(200)
        self.drop_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Pulse animation for active state
        self.pulse_animation = QPropertyAnimation(self, b"geometry")
        self.pulse_animation.setDuration(1000)
        self.pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.pulse_animation.setLoopCount(-1)  # Infinite loop
    
    def _update_drop_area_style(self, drop_area: QFrame, is_hover: bool, is_active: bool):
        """Update drop area styling based on state."""
        from ui.themes import theme_manager
        
        if is_active:
            border_color = theme_manager.get_color("drop_zone_active")
            background_color = theme_manager.get_color("drop_zone_background")
            border_style = "solid"
        elif is_hover:
            border_color = theme_manager.get_color("drop_zone_hover")
            background_color = theme_manager.get_color("drop_zone_background")
            border_style = "dashed"
        else:
            border_color = theme_manager.get_color("drop_zone_border")
            background_color = "transparent"
            border_style = "dashed"
        
        drop_area.setStyleSheet(f"""
        QFrame {{
            border: 2px {border_style} {border_color};
            border-radius: 8px;
            background-color: {background_color};
        }}
        """)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event with enhanced validation."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            csv_files = [url.toLocalFile() for url in urls 
                        if url.toLocalFile().lower().endswith('.csv')]
            
            # Validate file count for zone type
            if self._validate_file_count(csv_files):
                event.acceptProposedAction()
                self.is_drag_active = True
                self._update_drop_area_style(self.drop_area, True, False)
                self._start_pulse_animation()
            else:
                event.ignore()
                if len(csv_files) > self.max_files:
                    show_warning_toast(f"Maximum {self.max_files} files allowed for {self.zone_type.value}")
                elif not self.accept_multiple and len(csv_files) > 1:
                    show_warning_toast(f"Only one file allowed for {self.zone_type.value}")
        else:
            event.ignore()
    
    def _validate_file_count(self, csv_files: List[str]) -> bool:
        """Validate file count based on zone settings."""
        if not csv_files:
            return False
        
        if not self.accept_multiple and len(csv_files) > 1:
            return False
        
        if len(csv_files) > self.max_files:
            return False
        
        # For baseline zone, only allow one file
        if self.zone_type == DropZoneType.BASELINE and len(csv_files) > 1:
            return False
        
        return True
    
    def _start_pulse_animation(self):
        """Start pulse animation for active drag state."""
        if not self.pulse_animation.state() == QPropertyAnimation.State.Running:
            current_rect = self.geometry()
            expanded_rect = QRect(
                current_rect.x() - 2, current_rect.y() - 2,
                current_rect.width() + 4, current_rect.height() + 4
            )
            
            self.pulse_animation.setStartValue(current_rect)
            self.pulse_animation.setEndValue(expanded_rect)
            self.pulse_animation.start()
    
    def _stop_pulse_animation(self):
        """Stop pulse animation."""
        if self.pulse_animation.state() == QPropertyAnimation.State.Running:
            self.pulse_animation.stop()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave event."""
        self.is_drag_active = False
        self._update_drop_area_style(self.drop_area, False, False)
        self._stop_pulse_animation()
        event.accept()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event with comprehensive processing."""
        self.is_drag_active = False
        self._update_drop_area_style(self.drop_area, False, True)
        self._stop_pulse_animation()
        
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_paths = []
            invalid_files = []
            
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.csv'):
                    if Path(file_path).exists():
                        file_paths.append(file_path)
                    else:
                        invalid_files.append(file_path)
                        self.logger.warning(f"File does not exist: {file_path}")
            
            if file_paths:
                # Validate file count again
                if self._validate_file_count(file_paths):
                    self._process_dropped_files(file_paths)
                    event.acceptProposedAction()
                    
                    # Show success feedback
                    file_count = len(file_paths)
                    if file_count == 1:
                        show_success_toast(f"Added {Path(file_paths[0]).name}")
                    else:
                        show_success_toast(f"Added {file_count} files")
                else:
                    event.ignore()
            else:
                if invalid_files:
                    show_error_toast("Some files could not be found")
                else:
                    show_error_toast("No valid CSV files found")
                event.ignore()
        else:
            event.ignore()
        
        # Reset drop area style after a delay
        QTimer.singleShot(500, lambda: self._update_drop_area_style(self.drop_area, False, False))
    
    def _process_dropped_files(self, file_paths: List[str]):
        """Process dropped files and update UI."""
        try:
            # Convert to Path objects
            new_files = [Path(path) for path in file_paths]
            
            # For baseline zone, replace existing file
            if self.zone_type == DropZoneType.BASELINE:
                self.current_files = new_files
            else:
                # Add to existing files, avoiding duplicates
                for file_path in new_files:
                    if file_path not in self.current_files:
                        self.current_files.append(file_path)
            
            # Initialize file statuses
            for file_path in new_files:
                self.file_statuses[str(file_path)] = FileStatus.PENDING
            
            # Update UI
            self._update_file_list()
            self._update_status()
            
            # Emit signal
            self.files_dropped.emit(file_paths, self.zone_type.value)
            
        except Exception as e:
            self.logger.error(f"Error processing dropped files: {e}")
            show_error_toast(f"Error processing files: {str(e)}")
    
    def _browse_files(self):
        """Open file dialog to browse for CSV files."""
        file_dialog = QFileDialog(self)
        if self.accept_multiple:
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        else:
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        
        file_dialog.setNameFilter("CSV files (*.csv);;All files (*)")
        file_dialog.setWindowTitle("Select CSV Files")
        
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self._process_dropped_files(file_paths)
    
    def _clear_files(self):
        """Clear all files from the list."""
        self.current_files.clear()
        self.file_statuses.clear()
        self._update_file_list()
        self._update_status()
        self.files_cleared.emit()
    
    def _on_file_selected(self, item: QListWidgetItem):
        """Handle file selection in the list."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.file_selected.emit(str(file_path))
    
    def _update_file_list(self):
        """Update the file list display."""
        self.file_list.clear()
        
        if not self.current_files:
            self.file_list.setVisible(False)
            return
        
        self.file_list.setVisible(True)
        
        for file_path in self.current_files:
            item = QListWidgetItem()
            
            # Get file status
            status = self.file_statuses.get(str(file_path), FileStatus.PENDING)
            status_icon = self._get_status_icon(status)
            
            # Format display text
            try:
                file_size = file_path.stat().st_size
                size_str = self._format_file_size(file_size)
                display_text = f"{status_icon} {file_path.name} ({size_str})"
            except Exception:
                display_text = f"{status_icon} {file_path.name}"
            
            item.setText(display_text)
            item.setToolTip(str(file_path))
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            
            # Set color based on status
            if status == FileStatus.ERROR:
                item.setForeground(QColor("#F44336"))
            elif status == FileStatus.AI_NORMALIZED:
                item.setForeground(QColor("#9C27B0"))
            elif status == FileStatus.VALIDATED:
                item.setForeground(QColor("#4CAF50"))
            
            self.file_list.addItem(item)
        
        # Auto-select first file
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
    
    def _get_status_icon(self, status: FileStatus) -> str:
        """Get icon for file status."""
        icons = {
            FileStatus.PENDING: "⏳",
            FileStatus.PROCESSING: "⚙️",
            FileStatus.VALIDATED: "✅",
            FileStatus.AI_NORMALIZED: "✨",
            FileStatus.ERROR: "❌"
        }
        return icons.get(status, "📄")
    
    def _update_status(self):
        """Update status label."""
        file_count = len(self.current_files)
        
        if file_count == 0:
            self.status_label.setText("Ready to accept files")
            self.clear_btn.setEnabled(False)
        elif file_count == 1:
            self.status_label.setText("1 file loaded")
            self.clear_btn.setEnabled(True)
        else:
            self.status_label.setText(f"{file_count} files loaded")
            self.clear_btn.setEnabled(True)
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def get_current_files(self) -> List[Path]:
        """Get the current list of files."""
        return self.current_files.copy()
    
    def get_selected_file(self) -> Optional[Path]:
        """Get the currently selected file."""
        current_item = self.file_list.currentItem()
        if current_item:
            return current_item.data(Qt.ItemDataRole.UserRole)
        return None
    
    def set_file_status(self, file_path: Path, status: FileStatus):
        """Set status for a specific file."""
        self.file_statuses[str(file_path)] = status
        self.file_status_changed.emit(str(file_path), status.value)
        self._update_file_list()
    
    def remove_file(self, file_path: Path):
        """Remove a specific file from the list."""
        try:
            self.current_files.remove(file_path)
            self.file_statuses.pop(str(file_path), None)
            self._update_file_list()
            self._update_status()
        except ValueError:
            self.logger.warning(f"File not found in list: {file_path}")
    
    def show_progress(self, progress: int, message: str = ""):
        """Show progress bar with optional message."""
        self.progress_bar.setValue(progress)
        self.progress_bar.setVisible(True)
        if message:
            self.status_label.setText(message)
    
    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.setVisible(False)
        self._update_status()


# Maintain backward compatibility with old class name
FileDropZone = ModernFileDropZone