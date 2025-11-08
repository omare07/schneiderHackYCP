#!/usr/bin/env python3
"""
Simple test script for AI normalization engine core functionality.

Tests the AI normalizer and OpenRouter client without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add the spectral_analyzer directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that we can import the core components."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import json
        import hashlib
        from datetime import datetime
        from typing import Dict, List, Optional, Any
        from dataclasses import dataclass
        from enum import Enum
        
        print("✓ Basic dependencies imported successfully")
        
        # Test our custom components
        from core.ai_normalizer import (
            ConfidenceLevel, ColumnMapping, TransformationStep, 
            AIAnalysis, NormalizationResult, NormalizationPlan
        )
        
        print("✓ AI normalizer data classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_classes():
    """Test the data classes work correctly."""
    print("Testing data classes...")
    
    try:
        from core.ai_normalizer import (
            ConfidenceLevel, ColumnMapping, TransformationStep, 
            AIAnalysis, NormalizationResult, NormalizationPlan
        )
        
        # Test ConfidenceLevel enum
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
        
        # Test ColumnMapping
        mapping = ColumnMapping(
            original_name="Wavenumber",
            target_name="wavenumber",
            data_type="numeric",
            confidence=0.95,
            notes="Test mapping"
        )
        assert mapping.original_name == "Wavenumber"
        assert mapping.confidence == 0.95
        
        # Test TransformationStep
        step = TransformationStep(
            type="sort_by_wavenumber_desc",
            parameters={},
            reason="Test transformation",
            confidence=1.0
        )
        assert step.type == "sort_by_wavenumber_desc"
        
        print("✓ Data classes working correctly")
        return True
    except Exception as e:
        print(f"✗ Data classes test failed: {e}")
        return False


def test_openrouter_client_basic():
    """Test OpenRouter client basic functionality."""
    print("Testing OpenRouter client...")
    
    try:
        from utils.api_client import OpenRouterClient
        
        # Create client
        client = OpenRouterClient("test_api_key")
        
        # Test basic properties
        assert client.api_key == "test_api_key"
        assert client.default_model == "x-ai/grok-4-fast"
        assert client.base_url == "https://openrouter.ai/api/v1"
        
        # Test usage tracking
        client.track_usage(100, 0.001)
        client.track_cache_hit()
        
        stats = client.get_usage_stats()
        assert stats['total_tokens'] == 100
        assert stats['total_cost'] == 0.001
        assert stats['cache_hits'] == 1
        
        # Test reset
        client.reset_usage_stats()
        stats = client.get_usage_stats()
        assert stats['total_tokens'] == 0
        
        print("✓ OpenRouter client basic functionality working")
        return True
    except Exception as e:
        print(f"✗ OpenRouter client test failed: {e}")
        return False


def test_csv_analysis_prompt():
    """Test CSV analysis prompt creation."""
    print("Testing CSV analysis prompt...")
    
    try:
        from utils.api_client import OpenRouterClient
        
        client = OpenRouterClient("test_key")
        
        csv_preview = "wavenumber,absorbance\n4000,0.1\n3000,0.2"
        file_info = {"rows": 2, "columns": 2, "column_names": ["wavenumber", "absorbance"]}
        
        prompt = client._create_csv_analysis_prompt(csv_preview, file_info)
        
        # Check that prompt contains expected elements
        assert "spectroscopy" in prompt.lower()
        assert "wavenumber" in prompt.lower()
        assert "absorbance" in prompt.lower()
        assert csv_preview in prompt
        assert "JSON" in prompt
        assert "confidence" in prompt.lower()
        
        print("✓ CSV analysis prompt creation working")
        return True
    except Exception as e:
        print(f"✗ CSV analysis prompt test failed: {e}")
        return False


def run_simple_tests():
    """Run simple tests without full integration."""
    print("=" * 60)
    print("AI NORMALIZATION ENGINE - SIMPLE TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Classes Test", test_data_classes),
        ("OpenRouter Client Basic", test_openrouter_client_basic),
        ("CSV Analysis Prompt", test_csv_analysis_prompt),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"SIMPLE TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All simple tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        success = run_simple_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)