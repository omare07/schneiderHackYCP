"""
UI Components package for Spectral Analyzer.

Contains reusable UI components including:
- File drop zone for drag-and-drop functionality
- Preview widget for real-time data visualization
- Status bar widget for application status
"""

from .file_drop_zone import FileDropZone
from .preview_widget import PreviewWidget
from .status_bar import StatusBarWidget

__all__ = ['FileDropZone', 'PreviewWidget', 'StatusBarWidget']