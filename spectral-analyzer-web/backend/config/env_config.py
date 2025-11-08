"""
Environment configuration for the web backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend directory
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')

# Application Settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Simple config manager compatible with existing code
class SimpleConfigManager:
    """Simple configuration manager for API keys"""
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider"""
        if provider.lower() in ['openrouter', 'openrouter.ai']:
            return OPENROUTER_API_KEY
        return None
    
    def set_api_key(self, provider: str, key: str):
        """Set API key (not implemented for env-based config)"""
        pass

# Global config manager instance
config_manager = SimpleConfigManager()