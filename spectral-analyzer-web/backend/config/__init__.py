"""
Configuration management package for Spectral Analyzer.

This package handles all configuration-related functionality including:
- Application settings management
- API configuration and key storage
- User preferences
- Network deployment settings
"""

from .settings import ConfigManager
from .api_config import APIConfig

__all__ = ['ConfigManager', 'APIConfig']