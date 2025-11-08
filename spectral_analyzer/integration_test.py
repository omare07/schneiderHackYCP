#!/usr/bin/env python3
"""
Integration test script for AI normalization engine.

Tests the complete workflow from CSV loading to AI normalization
without requiring external API calls.
"""

import asyncio
import pandas as pd
import sys
import os
from pathlib import Path

# Add the spectral_analyzer directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.ai_normalizer import AINormalizer, NormalizationResult, ConfidenceLevel
from utils.api_client import OpenRouterClient
from config.settings import ConfigManager


def create_test_data():
    """Create test CSV data for integration testing."""
    return pd.DataFrame({
        'Wavenumber': [4000, 3500, 3000, 2500, 2000, 1500, 1000, 500],
        'Absorbance': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    })


def create_problematic_data():
    """Create problematic CSV data to test transformation engine."""
    return pd.DataFrame({
        'Wave Number (cm-1)': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],  # Ascending order
        'Transmittance %': [90, 80, 70, 60, 50, 40, 30, 20],
        'Sample ID': ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1']
    })


async def test_ai_normalizer_initialization():
    """Test AINormalizer initialization."""
    print("Testing AINormalizer initialization...")
    
    try:
        normalizer = AINormalizer()
        print("✓ AINormalizer initialized successfully")
        
        # Test basic properties
        assert normalizer.expected_format is not None
        assert normalizer.usage_stats is not None
        print("✓ Basic properties verified")
        
        return True
    except Exception as e:
        print(f"✗ AINormalizer initialization failed: {e}")
        return False


def test_file_hash_generation():
    """Test file hash generation for caching."""
    print("Testing file hash generation...")
    
    try:
        normalizer = AINormalizer()
        test_data = create_test_data()
        
        hash1 = normalizer._generate_file_hash(test_data, "test.csv")
        hash2 = normalizer._generate_file_hash(test_data, "test.csv")
        
        assert hash1 == hash2, "Same data should produce same hash"
        assert len(hash1) == 64, "Hash should be 64 characters (SHA256)"
        
        print("✓ File hash generation working correctly")
        return True
    except Exception as e:
        print(f"✗ File hash generation failed: {e}")
        return False


def test_csv_preview_preparation():
    """Test CSV preview preparation for AI analysis."""
    print("Testing CSV preview preparation...")
    
    try:
        normalizer = AINormalizer()
        test_data = create_test_data()
        
        preview = normalizer._prepare_csv_preview(test_data, max_rows=5)
        
        assert isinstance(preview, str), "Preview should be string"
        assert "Wavenumber" in preview, "Preview should contain column names"
        assert "Absorbance" in preview, "Preview should contain column names"
        
        print("✓ CSV preview preparation working correctly")
        return True
    except Exception as e:
        print(f"✗ CSV preview preparation failed: {e}")
        return False


def test_transformation_engine():
    """Test the transformation engine with various transformations."""
    print("Testing transformation engine...")
    
    try:
        normalizer = AINormalizer()
        test_data = create_problematic_data()
        
        # Test transmittance to absorbance conversion
        transmittance_data = pd.DataFrame({
            'wavenumber': [4000, 3000, 2000, 1000],
            'transmittance': [90, 80, 70, 60]
        })
        
        converted = normalizer._apply_transformation(
            transmittance_data, "convert_transmittance_to_absorbance"
        )
        
        assert 'absorbance' in converted.columns, "Should have absorbance column"
        assert 'transmittance' not in converted.columns, "Should not have transmittance column"
        assert all(converted['absorbance'] > 0), "Absorbance values should be positive"
        
        # Test interpolation
        nan_data = pd.DataFrame({
            'wavenumber': [4000, 3000, 2000, 1000],
            'absorbance': [0.1, None, 0.3, 0.4]
        })
        
        interpolated = normalizer._apply_transformation(nan_data, "interpolate_missing_values")
        assert not interpolated.isnull().any().any(), "Should not have NaN values after interpolation"
        
        print("✓ Transformation engine working correctly")
        return True
    except Exception as e:
        print(f"✗ Transformation engine failed: {e}")
        return False


async def test_fallback_normalization():
    """Test fallback normalization when AI is not available."""
    print("Testing fallback normalization...")
    
    try:
        normalizer = AINormalizer()
        test_data = create_test_data()
        
        # Create fallback plan (simulating AI failure)
        fallback_plan = normalizer._create_fallback_plan(test_data, "test.csv", "AI unavailable")
        
        assert fallback_plan.confidence_level == ConfidenceLevel.LOW
        assert fallback_plan.ai_model == "fallback_heuristic"
        assert len(fallback_plan.column_mappings) > 0
        
        # Test applying fallback plan
        normalized_data = await normalizer.apply_normalization_plan(test_data, fallback_plan)
        assert isinstance(normalized_data, pd.DataFrame)
        
        print("✓ Fallback normalization working correctly")
        return True
    except Exception as e:
        print(f"✗ Fallback normalization failed: {e}")
        return False


def test_data_validation():
    """Test normalized data validation."""
    print("Testing data validation...")
    
    try:
        normalizer = AINormalizer()
        
        # Valid data
        valid_data = pd.DataFrame({
            'wavenumber': [4000, 3000, 2000, 1000],
            'absorbance': [0.1, 0.2, 0.3, 0.4]
        })
        
        validation = normalizer._validate_normalized_data(valid_data)
        assert validation['is_valid'] is True
        
        # Invalid data
        invalid_data = pd.DataFrame({
            'frequency': [4000, 3000, 2000, 1000],
            'absorbance': [0.1, 0.2, 0.3, 0.4]
        })
        
        validation = normalizer._validate_normalized_data(invalid_data)
        assert validation['is_valid'] is False
        assert "Missing wavenumber column" in validation['issues']
        
        print("✓ Data validation working correctly")
        return True
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        return False


def test_openrouter_client_creation():
    """Test OpenRouter client creation."""
    print("Testing OpenRouter client creation...")
    
    try:
        client = OpenRouterClient("test_api_key")
        
        assert client.api_key == "test_api_key"
        assert client.default_model == "x-ai/grok-4-fast"
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert "Bearer test_api_key" in client.headers["Authorization"]
        
        # Test usage tracking
        client.track_usage(100, 0.001)
        client.track_cache_hit()
        
        stats = client.get_usage_stats()
        assert stats['total_tokens'] == 100
        assert stats['total_cost'] == 0.001
        assert stats['cache_hits'] == 1
        
        print("✓ OpenRouter client creation working correctly")
        return True
    except Exception as e:
        print(f"✗ OpenRouter client creation failed: {e}")
        return False


def test_supported_transformations():
    """Test getting supported transformations."""
    print("Testing supported transformations...")
    
    try:
        normalizer = AINormalizer()
        transformations = normalizer.get_supported_transformations()
        
        assert isinstance(transformations, list)
        assert len(transformations) > 0
        
        # Check structure
        for transform in transformations:
            assert 'name' in transform
            assert 'description' in transform
            assert 'parameters' in transform
        
        print(f"✓ Found {len(transformations)} supported transformations")
        return True
    except Exception as e:
        print(f"✗ Supported transformations test failed: {e}")
        return False


async def run_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("AI NORMALIZATION ENGINE - INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("AINormalizer Initialization", test_ai_normalizer_initialization()),
        ("File Hash Generation", test_file_hash_generation()),
        ("CSV Preview Preparation", test_csv_preview_preparation()),
        ("Transformation Engine", test_transformation_engine()),
        ("Fallback Normalization", test_fallback_normalization()),
        ("Data Validation", test_data_validation()),
        ("OpenRouter Client Creation", test_openrouter_client_creation()),
        ("Supported Transformations", test_supported_transformations()),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)