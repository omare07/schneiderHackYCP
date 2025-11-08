"""
API configuration and management for Spectral Analyzer.

Handles OpenRouter API configuration, authentication, and service management.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class APIProvider(Enum):
    """Supported API providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTier(Enum):
    """Model performance tiers."""
    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str
    provider: APIProvider
    tier: ModelTier
    cost_per_token: float
    max_tokens: int
    context_window: int
    supports_json: bool = True
    supports_streaming: bool = False
    description: str = ""


@dataclass
class APIEndpoint:
    """API endpoint configuration."""
    base_url: str
    chat_completion: str = "/chat/completions"
    models: str = "/models"
    usage: str = "/usage"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3


class APIConfig:
    """
    Centralized API configuration management.
    
    Manages API endpoints, model configurations, and service settings
    for different AI providers.
    """
    
    def __init__(self):
        """Initialize API configuration."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize model configurations
        self._init_model_configs()
        
        # Initialize API endpoints
        self._init_api_endpoints()
        
        # Current configuration
        self.current_provider = APIProvider.OPENROUTER
        self.current_model = "google/gemini-2.0-flash-exp"
        self.fallback_models = [
            "anthropic/claude-3-haiku",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-8b-instruct"
        ]
    
    def _init_model_configs(self):
        """Initialize supported model configurations."""
        self.models = {
            # OpenRouter Models
            "x-ai/grok-4-fast": ModelConfig(
                name="x-ai/grok-4-fast",
                provider=APIProvider.OPENROUTER,
                tier=ModelTier.FAST,
                cost_per_token=0.000001,
                max_tokens=4096,
                context_window=32768,
                supports_json=True,
                description="Fast Grok model optimized for structured data analysis"
            ),
            "anthropic/claude-3-haiku": ModelConfig(
                name="anthropic/claude-3-haiku",
                provider=APIProvider.OPENROUTER,
                tier=ModelTier.BALANCED,
                cost_per_token=0.000005,
                max_tokens=4096,
                context_window=200000,
                supports_json=True,
                description="Balanced Claude model for general tasks"
            ),
            "openai/gpt-3.5-turbo": ModelConfig(
                name="openai/gpt-3.5-turbo",
                provider=APIProvider.OPENROUTER,
                tier=ModelTier.FAST,
                cost_per_token=0.000002,
                max_tokens=4096,
                context_window=16384,
                supports_json=True,
                description="Fast GPT model for quick processing"
            ),
            "meta-llama/llama-3.1-8b-instruct": ModelConfig(
                name="meta-llama/llama-3.1-8b-instruct",
                provider=APIProvider.OPENROUTER,
                tier=ModelTier.FAST,
                cost_per_token=0.0000005,
                max_tokens=2048,
                context_window=8192,
                supports_json=True,
                description="Cost-effective Llama model for basic tasks"
            )
        }
    
    def _init_api_endpoints(self):
        """Initialize API endpoint configurations."""
        self.endpoints = {
            APIProvider.OPENROUTER: APIEndpoint(
                base_url="https://openrouter.ai/api/v1",
                headers={
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://spectral-analyzer.mrglabs.com",
                    "X-Title": "MRG Labs Spectral Analyzer"
                },
                timeout=30,
                max_retries=3
            ),
            APIProvider.OPENAI: APIEndpoint(
                base_url="https://api.openai.com/v1",
                headers={
                    "Content-Type": "application/json"
                },
                timeout=30,
                max_retries=3
            ),
            APIProvider.ANTHROPIC: APIEndpoint(
                base_url="https://api.anthropic.com/v1",
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                timeout=30,
                max_retries=3
            )
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig object or None if not found
        """
        return self.models.get(model_name)
    
    def get_endpoint_config(self, provider: APIProvider) -> Optional[APIEndpoint]:
        """
        Get endpoint configuration for a provider.
        
        Args:
            provider: API provider
            
        Returns:
            APIEndpoint object or None if not found
        """
        return self.endpoints.get(provider)
    
    def get_models_by_tier(self, tier: ModelTier) -> List[ModelConfig]:
        """
        Get all models of a specific tier.
        
        Args:
            tier: Model tier to filter by
            
        Returns:
            List of ModelConfig objects
        """
        return [model for model in self.models.values() if model.tier == tier]
    
    def get_models_by_provider(self, provider: APIProvider) -> List[ModelConfig]:
        """
        Get all models from a specific provider.
        
        Args:
            provider: API provider to filter by
            
        Returns:
            List of ModelConfig objects
        """
        return [model for model in self.models.values() if model.provider == provider]
    
    def estimate_cost(self, model_name: str, input_tokens: int, output_tokens: int = 0) -> float:
        """
        Estimate cost for API call.
        
        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_config = self.get_model_config(model_name)
        if not model_config:
            return 0.0
        
        total_tokens = input_tokens + output_tokens
        return total_tokens * model_config.cost_per_token
    
    def get_fallback_chain(self, primary_model: str) -> List[str]:
        """
        Get fallback model chain for a primary model.
        
        Args:
            primary_model: Primary model name
            
        Returns:
            List of fallback model names
        """
        primary_config = self.get_model_config(primary_model)
        if not primary_config:
            return self.fallback_models.copy()
        
        # Filter fallbacks by same or lower tier
        tier_order = {ModelTier.PREMIUM: 3, ModelTier.BALANCED: 2, ModelTier.FAST: 1}
        primary_tier_value = tier_order.get(primary_config.tier, 1)
        
        fallbacks = []
        for model_name in self.fallback_models:
            model_config = self.get_model_config(model_name)
            if model_config and tier_order.get(model_config.tier, 1) <= primary_tier_value:
                fallbacks.append(model_name)
        
        return fallbacks
    
    def validate_model_compatibility(self, model_name: str, requirements: Dict[str, Any]) -> bool:
        """
        Validate if a model meets specific requirements.
        
        Args:
            model_name: Name of the model to validate
            requirements: Dictionary of requirements to check
            
        Returns:
            bool: True if model meets all requirements
        """
        model_config = self.get_model_config(model_name)
        if not model_config:
            return False
        
        # Check JSON support requirement
        if requirements.get('json_support', False) and not model_config.supports_json:
            return False
        
        # Check minimum context window
        min_context = requirements.get('min_context_window', 0)
        if model_config.context_window < min_context:
            return False
        
        # Check maximum cost per token
        max_cost = requirements.get('max_cost_per_token', float('inf'))
        if model_config.cost_per_token > max_cost:
            return False
        
        # Check minimum max tokens
        min_max_tokens = requirements.get('min_max_tokens', 0)
        if model_config.max_tokens < min_max_tokens:
            return False
        
        return True
    
    def get_recommended_model(self, task_type: str = "normalization") -> str:
        """
        Get recommended model for a specific task type.
        
        Args:
            task_type: Type of task (normalization, analysis, etc.)
            
        Returns:
            Recommended model name
        """
        if task_type == "normalization":
            # For CSV normalization, prioritize fast models with JSON support
            requirements = {
                'json_support': True,
                'min_context_window': 8192,
                'max_cost_per_token': 0.000005
            }
            
            # Check models in order of preference
            preferred_models = [
                "x-ai/grok-4-fast",
                "openai/gpt-3.5-turbo",
                "meta-llama/llama-3.1-8b-instruct",
                "anthropic/claude-3-haiku"
            ]
            
            for model_name in preferred_models:
                if self.validate_model_compatibility(model_name, requirements):
                    return model_name
        
        # Default fallback
        return self.current_model
    
    def update_model_config(self, model_name: str, **kwargs):
        """
        Update configuration for a specific model.
        
        Args:
            model_name: Name of the model to update
            **kwargs: Configuration parameters to update
        """
        if model_name in self.models:
            model_config = self.models[model_name]
            for key, value in kwargs.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)
            
            self.logger.info(f"Updated configuration for model {model_name}")
        else:
            self.logger.warning(f"Model {model_name} not found in configuration")
    
    def add_custom_model(self, model_config: ModelConfig):
        """
        Add a custom model configuration.
        
        Args:
            model_config: ModelConfig object for the new model
        """
        self.models[model_config.name] = model_config
        self.logger.info(f"Added custom model configuration: {model_config.name}")
    
    def get_provider_status(self) -> Dict[APIProvider, bool]:
        """
        Get status of all API providers.
        
        Returns:
            Dictionary mapping providers to their availability status
        """
        # This would typically check actual API availability
        # For now, return default status
        return {
            APIProvider.OPENROUTER: True,
            APIProvider.OPENAI: True,
            APIProvider.ANTHROPIC: True
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary containing usage statistics
        """
        # Placeholder for usage tracking
        return {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'requests_by_model': {},
            'daily_usage': {},
            'monthly_usage': {}
        }