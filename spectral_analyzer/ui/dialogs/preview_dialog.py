"""
Preview dialog for AI normalization results.

Shows normalization plan preview with confidence indicators,
column mappings, and user approval interface.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QSplitter, QTabWidget,
    QWidget, QProgressBar, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from core.ai_normalizer import NormalizationPlan, ConfidenceLevel
from core.csv_parser import CSVStructure


class PreviewDialog(QDialog):
    """
    Normalization preview dialog.
    
    Features:
    - AI normalization plan display
    - Confidence level indicators
    - Column mapping preview
    - User approval interface
    - Manual adjustment options
    """
    
    # Signals
    plan_approved = pyqtSignal(object)  # NormalizationPlan
    plan_rejected = pyqtSignal()
    manual_mapping_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the preview dialog."""
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        
        # Current data
        self.normalization_plan: Optional[NormalizationPlan] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.csv_structure: Optional[CSVStructure] = None
        
        # UI components
        self.confidence_label: Optional[QLabel] = None
        self.confidence_bar: Optional[QProgressBar] = None
        self.mapping_table: Optional[QTableWidget] = None
        self.issues_text: Optional[QTextEdit] = None
        self.preview_text: Optional[QTextEdit] = None
        
        self._setup_ui()
        
        self.logger.debug("Preview dialog initialized")
    
    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("AI Normalization Preview")
        self.setModal(True)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        
        # Header section
        header_section = self._create_header_section()
        layout.addWidget(header_section)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Mapping details
        left_panel = self._create_mapping_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - Preview and issues
        right_panel = self._create_preview_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([400, 400])
        layout.addWidget(main_splitter)
        
        # Button section
        button_section = self._create_button_section()
        layout.addWidget(button_section)
    
    def _create_header_section(self) -> QWidget:
        """Create the header section with confidence display."""
        section = QGroupBox("AI Analysis Results")
        layout = QVBoxLayout(section)
        
        # Confidence display
        confidence_layout = QHBoxLayout()
        
        confidence_layout.addWidget(QLabel("Confidence Level:"))
        
        self.confidence_label = QLabel("Unknown")
        confidence_font = QFont()
        confidence_font.setBold(True)
        self.confidence_label.setFont(confidence_font)
        confidence_layout.addWidget(self.confidence_label)
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximumWidth(200)
        self.confidence_bar.setRange(0, 100)
        confidence_layout.addWidget(self.confidence_bar)
        
        confidence_layout.addStretch()
        layout.addLayout(confidence_layout)
        
        # Explanation text
        self.explanation_label = QLabel()
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.explanation_label)
        
        return section
    
    def _create_mapping_panel(self) -> QWidget:
        """Create the column mapping panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Mapping table
        mapping_group = QGroupBox("Column Mappings")
        mapping_layout = QVBoxLayout(mapping_group)
        
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(4)
        self.mapping_table.setHorizontalHeaderLabels([
            "Original Column", "Target Column", "Data Type", "Confidence"
        ])
        self.mapping_table.horizontalHeader().setStretchLastSection(True)
        mapping_layout.addWidget(self.mapping_table)
        
        layout.addWidget(mapping_group)
        
        # Transformations section
        transform_group = QGroupBox("Data Transformations")
        transform_layout = QVBoxLayout(transform_group)
        
        self.transformations_text = QTextEdit()
        self.transformations_text.setMaximumHeight(100)
        self.transformations_text.setReadOnly(True)
        transform_layout.addWidget(self.transformations_text)
        
        layout.addWidget(transform_group)
        
        return panel
    
    def _create_preview_panel(self) -> QWidget:
        """Create the preview and issues panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        tab_widget = QTabWidget()
        
        # Data preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Consolas", 9))
        preview_layout.addWidget(self.preview_text)
        
        tab_widget.addTab(preview_tab, "Data Preview")
        
        # Issues tab
        issues_tab = QWidget()
        issues_layout = QVBoxLayout(issues_tab)
        
        self.issues_text = QTextEdit()
        self.issues_text.setReadOnly(True)
        issues_layout.addWidget(self.issues_text)
        
        tab_widget.addTab(issues_tab, "Issues & Notes")
        
        layout.addWidget(tab_widget)
        
        return panel
    
    def _create_button_section(self) -> QWidget:
        """Create the button section."""
        section = QWidget()
        layout = QHBoxLayout(section)
        
        # Options
        self.auto_apply_check = QCheckBox("Remember this decision for similar files")
        layout.addWidget(self.auto_apply_check)
        
        layout.addStretch()
        
        # Action buttons
        self.manual_button = QPushButton("Manual Mapping")
        self.manual_button.clicked.connect(self._request_manual_mapping)
        layout.addWidget(self.manual_button)
        
        self.reject_button = QPushButton("Reject")
        self.reject_button.clicked.connect(self._reject_plan)
        layout.addWidget(self.reject_button)
        
        self.approve_button = QPushButton("Apply Normalization")
        self.approve_button.clicked.connect(self._approve_plan)
        self.approve_button.setDefault(True)
        layout.addWidget(self.approve_button)
        
        return section
    
    def show_normalization_preview(self, plan: NormalizationPlan, 
                                 original_data: pd.DataFrame,
                                 csv_structure: CSVStructure):
        """
        Show normalization plan preview.
        
        Args:
            plan: AI normalization plan
            original_data: Original CSV data
            csv_structure: CSV structure analysis
        """
        try:
            self.normalization_plan = plan
            self.original_data = original_data
            self.csv_structure = csv_structure
            
            # Update confidence display
            self._update_confidence_display(plan)
            
            # Update mapping table
            self._update_mapping_table(plan)
            
            # Update transformations
            self._update_transformations(plan)
            
            # Update preview
            self._update_data_preview(original_data)
            
            # Update issues
            self._update_issues_display(plan)
            
            # Update button states based on confidence
            self._update_button_states(plan)
            
            self.logger.info(f"Showing normalization preview with {plan.confidence_level.value} confidence")
            
        except Exception as e:
            self.logger.error(f"Failed to show normalization preview: {e}")
            self._show_error(f"Preview error: {e}")
    
    def _update_confidence_display(self, plan: NormalizationPlan):
        """Update confidence level display."""
        confidence_score = int(plan.confidence_score)
        self.confidence_bar.setValue(confidence_score)
        
        # Set confidence label and color
        if plan.confidence_level == ConfidenceLevel.HIGH:
            self.confidence_label.setText(f"HIGH ({confidence_score}%)")
            self.confidence_label.setStyleSheet("color: green; font-weight: bold;")
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            explanation = "AI is very confident about the column mappings. Safe to apply automatically."
            
        elif plan.confidence_level == ConfidenceLevel.MEDIUM:
            self.confidence_label.setText(f"MEDIUM ({confidence_score}%)")
            self.confidence_label.setStyleSheet("color: orange; font-weight: bold;")
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
            explanation = "AI has reasonable confidence. Please review the mappings before applying."
            
        else:  # LOW
            self.confidence_label.setText(f"LOW ({confidence_score}%)")
            self.confidence_label.setStyleSheet("color: red; font-weight: bold;")
            self.confidence_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            explanation = "AI has low confidence. Manual review and adjustment recommended."
        
        self.explanation_label.setText(explanation)
    
    def _update_mapping_table(self, plan: NormalizationPlan):
        """Update the column mapping table."""
        self.mapping_table.setRowCount(len(plan.column_mappings))
        
        for row, mapping in enumerate(plan.column_mappings):
            # Original column
            original_item = QTableWidgetItem(mapping.original_name)
            self.mapping_table.setItem(row, 0, original_item)
            
            # Target column
            target_item = QTableWidgetItem(mapping.target_name)
            self.mapping_table.setItem(row, 1, target_item)
            
            # Data type
            type_item = QTableWidgetItem(mapping.data_type)
            self.mapping_table.setItem(row, 2, type_item)
            
            # Confidence
            confidence_item = QTableWidgetItem(f"{mapping.confidence:.1%}")
            
            # Color code confidence
            if mapping.confidence >= 0.8:
                confidence_item.setBackground(QColor(200, 255, 200))  # Light green
            elif mapping.confidence >= 0.6:
                confidence_item.setBackground(QColor(255, 255, 200))  # Light yellow
            else:
                confidence_item.setBackground(QColor(255, 200, 200))  # Light red
            
            self.mapping_table.setItem(row, 3, confidence_item)
        
        self.mapping_table.resizeColumnsToContents()
    
    def _update_transformations(self, plan: NormalizationPlan):
        """Update transformations display."""
        if plan.data_transformations:
            transformations_text = "Planned transformations:\n\n"
            for i, transform in enumerate(plan.data_transformations, 1):
                transformations_text += f"{i}. {transform}\n"
        else:
            transformations_text = "No data transformations required."
        
        self.transformations_text.setPlainText(transformations_text)
    
    def _update_data_preview(self, data: pd.DataFrame):
        """Update data preview display."""
        try:
            # Show first few rows of data
            preview_lines = [
                f"Data Shape: {data.shape[0]:,} rows × {data.shape[1]} columns",
                "",
                "First 10 rows:",
                ""
            ]
            
            # Convert to string representation
            preview_df = data.head(10)
            preview_str = preview_df.to_string(max_cols=10, max_colwidth=20)
            preview_lines.append(preview_str)
            
            if len(data) > 10:
                preview_lines.extend(["", f"... and {len(data) - 10:,} more rows"])
            
            self.preview_text.setPlainText("\n".join(preview_lines))
            
        except Exception as e:
            self.preview_text.setPlainText(f"Error generating preview: {e}")
    
    def _update_issues_display(self, plan: NormalizationPlan):
        """Update issues and notes display."""
        issues_lines = []
        
        # Add detected issues
        if plan.issues_detected:
            issues_lines.extend([
                "Issues Detected:",
                ""
            ])
            for i, issue in enumerate(plan.issues_detected, 1):
                issues_lines.append(f"{i}. {issue}")
            issues_lines.append("")
        
        # Add analysis notes
        if plan.metadata.get('analysis_notes'):
            issues_lines.extend([
                "Analysis Notes:",
                "",
                plan.metadata['analysis_notes'],
                ""
            ])
        
        # Add model information
        issues_lines.extend([
            "AI Model Information:",
            "",
            f"Model: {plan.ai_model}",
            f"Analysis Time: {plan.timestamp}",
        ])
        
        # Add mapping details
        issues_lines.extend([
            "",
            "Mapping Details:",
            ""
        ])
        
        for mapping in plan.column_mappings:
            if mapping.notes:
                issues_lines.append(f"• {mapping.original_name} → {mapping.target_name}: {mapping.notes}")
        
        if not any(mapping.notes for mapping in plan.column_mappings):
            issues_lines.append("No additional mapping notes available.")
        
        self.issues_text.setPlainText("\n".join(issues_lines))
    
    def _update_button_states(self, plan: NormalizationPlan):
        """Update button states based on confidence level."""
        if plan.confidence_level == ConfidenceLevel.HIGH:
            self.approve_button.setText("Apply Normalization")
            self.approve_button.setStyleSheet("background-color: lightgreen;")
            
        elif plan.confidence_level == ConfidenceLevel.MEDIUM:
            self.approve_button.setText("Apply with Caution")
            self.approve_button.setStyleSheet("background-color: lightyellow;")
            
        else:  # LOW
            self.approve_button.setText("Apply Anyway")
            self.approve_button.setStyleSheet("background-color: lightcoral;")
            self.manual_button.setStyleSheet("background-color: lightblue; font-weight: bold;")
    
    def _show_error(self, message: str):
        """Show error message in the dialog."""
        self.confidence_label.setText("ERROR")
        self.confidence_label.setStyleSheet("color: red; font-weight: bold;")
        self.explanation_label.setText(message)
        
        self.preview_text.setPlainText(f"Error: {message}")
        self.issues_text.setPlainText(f"Error: {message}")
        
        # Disable approve button
        self.approve_button.setEnabled(False)
    
    def _approve_plan(self):
        """Approve the normalization plan."""
        if self.normalization_plan:
            self.plan_approved.emit(self.normalization_plan)
            self.accept()
    
    def _reject_plan(self):
        """Reject the normalization plan."""
        self.plan_rejected.emit()
        self.reject()
    
    def _request_manual_mapping(self):
        """Request manual mapping interface."""
        self.manual_mapping_requested.emit()
        self.reject()
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from the dialog."""
        return {
            'remember_decision': self.auto_apply_check.isChecked()
        }