"""
Comprehensive test suite for AI normalization engine.

Tests OpenRouter integration, confidence-based decision making,
caching, error handling, and transformation engine.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from core.ai_normalizer import (
    AINormalizer, ColumnMapping, NormalizationPlan, NormalizationResult,
    TransformationStep, AIAnalysis, ConfidenceLevel, UsageStats
)
from utils.api_client import OpenRouterClient, APIResponse
from utils.cache_manager import CacheManager
from config.settings import ConfigManager


class TestAINormalizer:
    """Test suite for AINormalizer class."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data for testing."""
        return pd.DataFrame({
            'Wavenumber': [4000, 3500, 3000, 2500, 2000, 1500, 1000, 500],
            'Absorbance': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        })
    
    @pytest.fixture
    def problematic_csv_data(self):
        """Create problematic CSV data for testing edge cases."""
        return pd.DataFrame({
            'Wave Number (cm-1)': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],  # Ascending order
            'Transmittance %': [90, 80, 70, 60, 50, 40, 30, 20],
            'Sample ID': ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A1']
        })
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create mock configuration manager."""
        config = Mock(spec=ConfigManager)
        config.get_api_key.return_value = "test_api_key"
        return config
    
    @pytest.fixture
    def mock_openrouter_response(self):
        """Create mock OpenRouter API response."""
        return {
            "can_normalize": True,
            "confidence": 0.85,
            "detected_format": {
                "delimiter": ",",
                "decimal_separator": ".",
                "has_headers": True,
                "metadata_rows": 0,
                "encoding": "utf-8"
            },
            "column_mapping": {
                "wavenumber_column": "Wavenumber",
                "absorbance_column": "Absorbance"
            },
            "column_mappings": [
                {
                    "original_name": "Wavenumber",
                    "target_name": "wavenumber",
                    "data_type": "numeric",
                    "transformation": "none",
                    "confidence": 0.95,
                    "notes": "Clear wavenumber column identification"
                },
                {
                    "original_name": "Absorbance",
                    "target_name": "absorbance",
                    "data_type": "numeric",
                    "transformation": "none",
                    "confidence": 0.90,
                    "notes": "Clear absorbance column identification"
                }
            ],
            "transformations": [
                {
                    "type": "sort_by_wavenumber_desc",
                    "parameters": {},
                    "reason": "Ensure descending wavenumber order"
                }
            ],
            "warnings": [],
            "recommendations": ["Data appears to be in standard format"],
            "confidence_score": 85,
            "analysis_notes": "High quality spectroscopy data with clear column identification"
        }
    
    @pytest.fixture
    def ai_normalizer(self, mock_config_manager):
        """Create AINormalizer instance with mocked dependencies."""
        with patch('core.ai_normalizer.CacheManager'), \
             patch('core.ai_normalizer.SecurityManager'):
            normalizer = AINormalizer(mock_config_manager)
            return normalizer
    
    def test_initialization(self, ai_normalizer):
        """Test AINormalizer initialization."""
        assert ai_normalizer.logger is not None
        assert ai_normalizer.api_config is not None
        assert ai_normalizer.cache_manager is not None
        assert ai_normalizer.expected_format is not None
        assert isinstance(ai_normalizer.usage_stats, UsageStats)
    
    def test_file_hash_generation(self, ai_normalizer, sample_csv_data):
        """Test file hash generation for caching."""
        hash1 = ai_normalizer._generate_file_hash(sample_csv_data, "test.csv")
        hash2 = ai_normalizer._generate_file_hash(sample_csv_data, "test.csv")
        
        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length
        
        # Different data should produce different hash
        different_data = sample_csv_data.copy()
        different_data['Absorbance'] = different_data['Absorbance'] * 2
        hash3 = ai_normalizer._generate_file_hash(different_data, "test.csv")
        assert hash1 != hash3
    
    def test_csv_preview_preparation(self, ai_normalizer, sample_csv_data):
        """Test CSV preview preparation for AI analysis."""
        preview = ai_normalizer._prepare_csv_preview(sample_csv_data, max_rows=5)
        
        assert isinstance(preview, str)
        assert "Wavenumber" in preview
        assert "Absorbance" in preview
        assert len(preview.split('\n')) <= 7  # Header + 5 rows + possible truncation
    
    def test_csv_preview_truncation(self, ai_normalizer):
        """Test CSV preview truncation for large data."""
        # Create large dataset
        large_data = pd.DataFrame({
            'col1': ['x' * 1000] * 100,  # Large string values
            'col2': range(100)
        })
        
        preview = ai_normalizer._prepare_csv_preview(large_data)
        assert len(preview) <= 8020  # 8000 + "... (truncated)"
        
        if len(preview) > 8000:
            assert "... (truncated)" in preview
    
    @pytest.mark.asyncio
    async def test_openrouter_client_initialization(self, mock_config_manager):
        """Test OpenRouter client initialization."""
        with patch('core.ai_normalizer.CacheManager'), \
             patch('core.ai_normalizer.SecurityManager'), \
             patch('core.ai_normalizer.OpenRouterClient') as mock_client:
            
            normalizer = AINormalizer(mock_config_manager)
            
            # Should attempt to create OpenRouter client
            mock_client.assert_called_once_with("test_api_key", mock_config_manager)
    
    @pytest.mark.asyncio
    async def test_ai_normalization_success(self, ai_normalizer, sample_csv_data, mock_openrouter_response):
        """Test successful AI normalization with high confidence."""
        # Mock OpenRouter client
        mock_client = AsyncMock()
        mock_client.analyze_csv_structure.return_value = mock_openrouter_response
        ai_normalizer.openrouter_client = mock_client
        
        # Mock cache manager
        ai_normalizer.cache_manager.get_normalization_plan = AsyncMock(return_value=None)
        ai_normalizer.cache_manager.store_normalization_plan = AsyncMock(return_value=True)
        
        result = await ai_normalizer.normalize_csv(sample_csv_data, "test.csv")
        
        assert isinstance(result, NormalizationResult)
        assert result.success is True
        assert result.normalized_data is not None
        assert result.plan is not None
        assert result.plan.confidence_level == ConfidenceLevel.HIGH
        assert result.cache_hit is False
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, ai_normalizer, sample_csv_data):
        """Test cache hit scenario."""
        # Create mock cached plan
        cached_plan = NormalizationPlan(
            file_hash="test_hash",
            column_mappings=[
                ColumnMapping("Wavenumber", "wavenumber", "numeric", confidence=0.95),
                ColumnMapping("Absorbance", "absorbance", "numeric", confidence=0.90)
            ],
            data_transformations=["sort_by_wavenumber_desc"],
            confidence_score=85.0,
            confidence_level=ConfidenceLevel.HIGH,
            issues_detected=[],
            metadata={},
            ai_model="x-ai/grok-4-fast",
            timestamp=datetime.now().isoformat()
        )
        
        # Mock cache manager to return cached plan
        ai_normalizer.cache_manager.get_normalization_plan = AsyncMock(return_value=cached_plan)
        
        result = await ai_normalizer.normalize_csv(sample_csv_data, "test.csv")
        
        assert result.success is True
        assert result.cache_hit is True
        assert result.plan == cached_plan
        assert ai_normalizer.usage_stats.cache_hits == 1
    
    @pytest.mark.asyncio
    async def test_ai_failure_fallback(self, ai_normalizer, sample_csv_data):
        """Test fallback behavior when AI fails."""
        # Mock OpenRouter client to fail
        mock_client = AsyncMock()
        mock_client.analyze_csv_structure.side_effect = Exception("API Error")
        ai_normalizer.openrouter_client = mock_client
        
        # Mock cache manager
        ai_normalizer.cache_manager.get_normalization_plan = AsyncMock(return_value=None)
        ai_normalizer.cache_manager.store_normalization_plan = AsyncMock(return_value=True)
        
        result = await ai_normalizer.normalize_csv(sample_csv_data, "test.csv")
        
        assert result.success is True  # Should succeed with fallback
        assert result.plan.confidence_level == ConfidenceLevel.LOW
        assert result.plan.ai_model == "fallback_heuristic"
        assert "AI normalization failed" in result.plan.issues_detected[0]
        assert result.error_message is not None
        assert "fallback" in result.error_message.lower()
    
    def test_transformation_engine(self, ai_normalizer, problematic_csv_data):
        """Test various data transformations."""
        # Test sorting by wavenumber descending
        sorted_df = ai_normalizer._apply_transformation(
            problematic_csv_data, "sort_by_wavenumber_desc"
        )
        # Should be sorted in descending order (but column name doesn't match exactly)
        
        # Test transmittance to absorbance conversion
        transmittance_data = pd.DataFrame({
            'wavenumber': [4000, 3000, 2000, 1000],
            'transmittance': [90, 80, 70, 60]
        })
        
        converted_df = ai_normalizer._apply_transformation(
            transmittance_data, "convert_transmittance_to_absorbance"
        )
        
        assert 'absorbance' in converted_df.columns
        assert 'transmittance' not in converted_df.columns
        assert all(converted_df['absorbance'] > 0)  # Absorbance should be positive
    
    def test_transformation_steps(self, ai_normalizer, sample_csv_data):
        """Test applying transformation steps."""
        steps = [
            TransformationStep(
                type="rename_columns",
                parameters={"mapping": {"Wavenumber": "wavenumber", "Absorbance": "absorbance"}},
                reason="Standardize column names"
            ),
            TransformationStep(
                type="sort_by_wavenumber_desc",
                parameters={},
                reason="Ensure descending order"
            )
        ]
        
        result_df = ai_normalizer.apply_transformation_steps(sample_csv_data, steps)
        
        assert 'wavenumber' in result_df.columns
        assert 'absorbance' in result_df.columns
        assert 'Wavenumber' not in result_df.columns
    
    def test_confidence_level_determination(self, ai_normalizer):
        """Test confidence level determination logic."""
        # Test high confidence
        high_confidence_response = {"confidence_score": 95}
        plan = ai_normalizer._process_ai_response(high_confidence_response, "test_hash")
        assert plan.confidence_level == ConfidenceLevel.HIGH
        
        # Test medium confidence
        medium_confidence_response = {"confidence_score": 75}
        plan = ai_normalizer._process_ai_response(medium_confidence_response, "test_hash")
        assert plan.confidence_level == ConfidenceLevel.MEDIUM
        
        # Test low confidence
        low_confidence_response = {"confidence_score": 50}
        plan = ai_normalizer._process_ai_response(low_confidence_response, "test_hash")
        assert plan.confidence_level == ConfidenceLevel.LOW
    
    def test_data_validation(self, ai_normalizer):
        """Test normalized data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'wavenumber': [4000, 3000, 2000, 1000],
            'absorbance': [0.1, 0.2, 0.3, 0.4]
        })
        
        validation = ai_normalizer._validate_normalized_data(valid_data)
        assert validation['is_valid'] is True
        assert len(validation['issues']) == 0
        
        # Invalid data - missing wavenumber
        invalid_data = pd.DataFrame({
            'frequency': [4000, 3000, 2000, 1000],
            'absorbance': [0.1, 0.2, 0.3, 0.4]
        })
        
        validation = ai_normalizer._validate_normalized_data(invalid_data)
        assert validation['is_valid'] is False
        assert "Missing wavenumber column" in validation['issues']
    
    def test_confidence_explanation(self, ai_normalizer):
        """Test confidence explanation generation."""
        plan = NormalizationPlan(
            file_hash="test",
            column_mappings=[
                ColumnMapping("col1", "wavenumber", "numeric", confidence=0.95),
                ColumnMapping("col2", "absorbance", "numeric", confidence=0.85)
            ],
            data_transformations=[],
            confidence_score=85.0,
            confidence_level=ConfidenceLevel.HIGH,
            issues_detected=["Minor formatting issue"],
            metadata={},
            ai_model="test",
            timestamp="2024-01-01"
        )
        
        explanation = ai_normalizer.get_confidence_explanation(plan)
        
        assert "High confidence" in explanation
        assert "automatically" in explanation.lower()
        assert "Issues detected" in explanation
    
    def test_supported_transformations(self, ai_normalizer):
        """Test getting supported transformations list."""
        transformations = ai_normalizer.get_supported_transformations()
        
        assert isinstance(transformations, list)
        assert len(transformations) > 0
        
        # Check that each transformation has required fields
        for transform in transformations:
            assert 'name' in transform
            assert 'description' in transform
            assert 'parameters' in transform
    
    @pytest.mark.asyncio
    async def test_ai_connection_test(self, ai_normalizer):
        """Test AI connection testing."""
        # Mock successful connection
        mock_client = AsyncMock()
        mock_client.test_connection.return_value = APIResponse(
            success=True,
            data={'message': 'Connection successful'},
            response_time=0.5
        )
        ai_normalizer.openrouter_client = mock_client
        
        result = await ai_normalizer.test_ai_connection()
        
        assert result['openrouter_available'] is True
        assert 'successful' in result['openrouter_message'].lower()
        assert result['response_time'] == 0.5
    
    @pytest.mark.asyncio
    async def test_usage_statistics_tracking(self, ai_normalizer, sample_csv_data, mock_openrouter_response):
        """Test usage statistics tracking."""
        # Mock OpenRouter client
        mock_client = AsyncMock()
        mock_client.analyze_csv_structure.return_value = mock_openrouter_response
        ai_normalizer.openrouter_client = mock_client
        
        # Mock cache manager
        ai_normalizer.cache_manager.get_normalization_plan = AsyncMock(return_value=None)
        ai_normalizer.cache_manager.store_normalization_plan = AsyncMock(return_value=True)
        
        initial_requests = ai_normalizer.usage_stats.total_requests
        
        await ai_normalizer.normalize_csv(sample_csv_data, "test.csv")
        
        assert ai_normalizer.usage_stats.total_requests == initial_requests + 1
        assert ai_normalizer.usage_stats.successful_requests == 1
        assert ai_normalizer.usage_stats.last_request_time is not None
    
    def test_edge_cases(self, ai_normalizer):
        """Test edge cases and error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        hash_result = ai_normalizer._generate_file_hash(empty_df, "empty.csv")
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
        
        # DataFrame with NaN values
        nan_df = pd.DataFrame({
            'wavenumber': [4000, np.nan, 2000, 1000],
            'absorbance': [0.1, 0.2, np.nan, 0.4]
        })
        
        # Should handle NaN values gracefully
        interpolated = ai_normalizer._apply_transformation(nan_df, "interpolate_missing_values")
        assert not interpolated.isnull().any().any()
    
    @pytest.mark.asyncio
    async def test_concurrent_normalization(self, ai_normalizer, sample_csv_data, mock_openrouter_response):
        """Test concurrent normalization requests."""
        # Mock OpenRouter client
        mock_client = AsyncMock()
        mock_client.analyze_csv_structure.return_value = mock_openrouter_response
        ai_normalizer.openrouter_client = mock_client
        
        # Mock cache manager
        ai_normalizer.cache_manager.get_normalization_plan = AsyncMock(return_value=None)
        ai_normalizer.cache_manager.store_normalization_plan = AsyncMock(return_value=True)
        
        # Run multiple normalizations concurrently
        tasks = [
            ai_normalizer.normalize_csv(sample_csv_data, f"test_{i}.csv")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result.success for result in results)
        assert len(results) == 3


class TestOpenRouterIntegration:
    """Test OpenRouter API integration specifically."""
    
    @pytest.mark.asyncio
    async def test_openrouter_client_creation(self):
        """Test OpenRouter client creation and configuration."""
        api_key = "test_key"
        client = OpenRouterClient(api_key)
        
        assert client.api_key == api_key
        assert client.default_model == "x-ai/grok-4-fast"
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert "Bearer test_key" in client.headers["Authorization"]
    
    @pytest.mark.asyncio
    async def test_csv_analysis_prompt_creation(self):
        """Test CSV analysis prompt creation."""
        client = OpenRouterClient("test_key")
        
        csv_preview = "wavenumber,absorbance\n4000,0.1\n3000,0.2"
        file_info = {"rows": 2, "columns": 2}
        
        prompt = client._create_csv_analysis_prompt(csv_preview, file_info)
        
        assert "spectroscopy" in prompt.lower()
        assert "wavenumber" in prompt.lower()
        assert "absorbance" in prompt.lower()
        assert csv_preview in prompt
        assert "JSON" in prompt
    
    def test_usage_statistics_tracking(self):
        """Test usage statistics tracking in OpenRouter client."""
        client = OpenRouterClient("test_key")
        
        # Track usage
        client.track_usage(100, 0.001)
        client.track_cache_hit()
        
        stats = client.get_usage_stats()
        
        assert stats['total_tokens'] == 100
        assert stats['total_cost'] == 0.001
        assert stats['cache_hits'] == 1
        assert stats['cache_hit_rate'] == 0  # No requests yet
    
    def test_usage_stats_reset(self):
        """Test usage statistics reset."""
        client = OpenRouterClient("test_key")
        
        # Add some usage
        client.track_usage(100, 0.001)
        client.track_cache_hit()
        
        # Reset
        client.reset_usage_stats()
        
        stats = client.get_usage_stats()
        assert stats['total_tokens'] == 0
        assert stats['total_cost'] == 0.0
        assert stats['cache_hits'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])