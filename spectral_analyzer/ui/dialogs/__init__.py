"""
Dialog windows package for Spectral Analyzer.

Contains dialog windows for:
- AI settings and configuration
- Data normalization preview
- Application preferences
- Export options
"""

from .ai_settings import AISettingsDialog
from .preview_dialog import PreviewDialog

__all__ = ['AISettingsDialog', 'PreviewDialog']