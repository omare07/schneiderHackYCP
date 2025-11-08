"""
Configuration management system for Spectral Analyzer.

Handles application settings, user preferences, and persistent configuration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from PyQt6.QtCore import QSettings

from utils.security import SecureKeyManager


@dataclass
class UISettings:
    """UI-related settings."""
    theme: str = "dark"
    window_geometry: Optional[str] = None
    window_state: Optional[str] = None
    splitter_states: Dict[str, str] = None
    recent_files: list = None
    max_recent_files: int = 10
    
    def __post_init__(self):
        if self.splitter_states is None:
            self.splitter_states = {}
        if self.recent_files is None:
            self.recent_files = []


@dataclass
class ProcessingSettings:
    """Data processing settings."""
    batch_size: int = 10
    max_concurrent_files: int = 4
    auto_normalize: bool = True
    confidence_threshold_high: float = 90.0
    confidence_threshold_medium: float = 70.0
    cache_enabled: bool = True
    cache_ttl_days: int = 30
    preview_rows: int = 100


@dataclass
class AISettings:
    """AI service configuration."""
    provider: str = "openrouter"
    model: str = "x-ai/grok-4-fast"
    api_base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: int = 30
    max_retries: int = 3
    cost_limit_daily: float = 10.0
    cost_limit_monthly: float = 200.0
    cost_warning_threshold: float = 0.8
    enable_cost_alerts: bool = True
    enable_fallback: bool = True
    fallback_models: list = None
    
    def __post_init__(self):
        if self.fallback_models is None:
            self.fallback_models = [
                "anthropic/claude-3-haiku",
                "openai/gpt-3.5-turbo"
            ]


@dataclass
class CacheSettings:
    """Cache configuration settings."""
    enable_caching: bool = True
    memory_limit_mb: int = 100
    disk_limit_mb: int = 1000
    default_ttl_hours: int = 24
    cleanup_interval_minutes: int = 60
    compression_enabled: bool = True
    compression_threshold_kb: int = 10
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    enable_background_cleanup: bool = True
    max_memory_entries: int = 1000


@dataclass
class NetworkSettings:
    """Network and deployment settings."""
    enable_network_cache: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    lims_integration_enabled: bool = False
    lims_base_url: str = ""
    sync_interval_minutes: int = 15
    offline_mode: bool = False


@dataclass
class SecuritySettings:
    """Security configuration."""
    encrypt_api_keys: bool = True
    require_password: bool = False
    session_timeout_minutes: int = 480  # 8 hours
    audit_logging: bool = True
    secure_cache: bool = True


class ConfigManager:
    """
    Centralized configuration management system.
    
    Handles loading, saving, and managing all application settings
    with support for different configuration sources and secure storage.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Custom configuration directory path
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up configuration directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = Path.home() / ".spectral_analyzer"
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration file paths
        self.settings_file = self.config_dir / "settings.json"
        self.user_prefs_file = self.config_dir / "user_preferences.json"
        
        # Qt Settings for system integration
        self.qt_settings = QSettings("MRG Labs", "Spectral Analyzer")
        
        # Security manager for API keys
        self.security_manager = SecureKeyManager()
        
        # Initialize settings
        self.ui_settings = UISettings()
        self.processing_settings = ProcessingSettings()
        self.ai_settings = AISettings()
        self.cache_settings = CacheSettings()
        self.network_settings = NetworkSettings()
        self.security_settings = SecuritySettings()
        
        # Load existing configuration
        self.load_settings()
    
    def load_settings(self) -> bool:
        """
        Load settings from configuration files.
        
        Returns:
            bool: True if settings loaded successfully
        """
        try:
            # Load main settings
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings_data = json.load(f)
                
                # Update settings objects
                if 'ui' in settings_data:
                    self.ui_settings = UISettings(**settings_data['ui'])
                if 'processing' in settings_data:
                    self.processing_settings = ProcessingSettings(**settings_data['processing'])
                if 'ai' in settings_data:
                    self.ai_settings = AISettings(**settings_data['ai'])
                if 'cache' in settings_data:
                    self.cache_settings = CacheSettings(**settings_data['cache'])
                if 'network' in settings_data:
                    self.network_settings = NetworkSettings(**settings_data['network'])
                if 'security' in settings_data:
                    self.security_settings = SecuritySettings(**settings_data['security'])
            
            # Load Qt-specific settings
            self._load_qt_settings()
            
            self.logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            return False
    
    def save_settings(self) -> bool:
        """
        Save current settings to configuration files.
        
        Returns:
            bool: True if settings saved successfully
        """
        try:
            # Prepare settings data
            settings_data = {
                'ui': asdict(self.ui_settings),
                'processing': asdict(self.processing_settings),
                'ai': asdict(self.ai_settings),
                'cache': asdict(self.cache_settings),
                'network': asdict(self.network_settings),
                'security': asdict(self.security_settings)
            }
            
            # Save main settings
            with open(self.settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)
            
            # Save Qt-specific settings
            self._save_qt_settings()
            
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            return False
    
    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            category: Settings category (ui, processing, ai, network, security)
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        try:
            settings_obj = getattr(self, f"{category}_settings")
            return getattr(settings_obj, key, default)
        except AttributeError:
            return default
    
    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """
        Set a specific setting value.
        
        Args:
            category: Settings category
            key: Setting key
            value: New value
            
        Returns:
            bool: True if setting was updated successfully
        """
        try:
            settings_obj = getattr(self, f"{category}_settings")
            if hasattr(settings_obj, key):
                setattr(settings_obj, key, value)
                return True
            return False
        except AttributeError:
            return False
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Retrieve encrypted API key for a provider.
        
        Args:
            provider: API provider name
            
        Returns:
            Decrypted API key or None
        """
        try:
            return self.security_manager.get_api_key(provider)
        except Exception as e:
            self.logger.error(f"Failed to retrieve API key for {provider}: {e}")
            return None
    
    def set_api_key(self, provider: str, api_key: str, password: Optional[str] = None) -> bool:
        """
        Store encrypted API key for a provider.
        
        Args:
            provider: API provider name
            api_key: API key to encrypt and store
            password: Optional password for encryption
            
        Returns:
            bool: True if key was stored successfully
        """
        try:
            return self.security_manager.set_api_key(provider, api_key, password)
        except Exception as e:
            self.logger.error(f"Failed to store API key for {provider}: {e}")
            return False
    
    def add_recent_file(self, file_path: str):
        """Add file to recent files list."""
        if file_path in self.ui_settings.recent_files:
            self.ui_settings.recent_files.remove(file_path)
        
        self.ui_settings.recent_files.insert(0, file_path)
        
        # Limit recent files
        if len(self.ui_settings.recent_files) > self.ui_settings.max_recent_files:
            self.ui_settings.recent_files = self.ui_settings.recent_files[:self.ui_settings.max_recent_files]
    
    def get_recent_files(self) -> list:
        """Get list of recent files."""
        return self.ui_settings.recent_files.copy()
    
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self.ui_settings = UISettings()
        self.processing_settings = ProcessingSettings()
        self.ai_settings = AISettings()
        self.cache_settings = CacheSettings()
        self.network_settings = NetworkSettings()
        self.security_settings = SecuritySettings()
        
        self.logger.info("Settings reset to defaults")
    
    def _load_qt_settings(self):
        """Load Qt-specific settings."""
        # Window geometry and state
        geometry = self.qt_settings.value("geometry")
        if geometry:
            self.ui_settings.window_geometry = geometry
        
        state = self.qt_settings.value("windowState")
        if state:
            self.ui_settings.window_state = state
    
    def _save_qt_settings(self):
        """Save Qt-specific settings."""
        if self.ui_settings.window_geometry:
            self.qt_settings.setValue("geometry", self.ui_settings.window_geometry)
        
        if self.ui_settings.window_state:
            self.qt_settings.setValue("windowState", self.ui_settings.window_state)
    
    def export_settings(self, export_path: Path) -> bool:
        """
        Export settings to a file for backup or sharing.
        
        Args:
            export_path: Path to export file
            
        Returns:
            bool: True if export successful
        """
        try:
            settings_data = {
                'ui': asdict(self.ui_settings),
                'processing': asdict(self.processing_settings),
                'ai': asdict(self.ai_settings),
                'cache': asdict(self.cache_settings),
                'network': asdict(self.network_settings),
                'security': asdict(self.security_settings),
                'version': '1.0.0',
                'export_timestamp': str(Path.ctime(Path.now()))
            }
            
            with open(export_path, 'w') as f:
                json.dump(settings_data, f, indent=2)
            
            self.logger.info(f"Settings exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, import_path: Path) -> bool:
        """
        Import settings from a file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            bool: True if import successful
        """
        try:
            with open(import_path, 'r') as f:
                settings_data = json.load(f)
            
            # Validate and import settings
            if 'ui' in settings_data:
                self.ui_settings = UISettings(**settings_data['ui'])
            if 'processing' in settings_data:
                self.processing_settings = ProcessingSettings(**settings_data['processing'])
            if 'ai' in settings_data:
                self.ai_settings = AISettings(**settings_data['ai'])
            if 'cache' in settings_data:
                self.cache_settings = CacheSettings(**settings_data['cache'])
            if 'network' in settings_data:
                self.network_settings = NetworkSettings(**settings_data['network'])
            if 'security' in settings_data:
                self.security_settings = SecuritySettings(**settings_data['security'])
            
            self.logger.info(f"Settings imported from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            return False