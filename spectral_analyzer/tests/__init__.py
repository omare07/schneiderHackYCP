"""
Test suite for Spectral Analyzer.

Comprehensive testing framework including:
- Unit tests for core modules
- Integration tests for workflows
- UI tests for components
- Test data and fixtures
"""

import sys
from pathlib import Path

# Add project root to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))