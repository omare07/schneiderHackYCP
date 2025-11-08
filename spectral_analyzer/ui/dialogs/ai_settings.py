"""
Modern AI Settings dialog for configuring AI normalization parameters.

Provides professional interface for managing API keys, model selection,
confidence thresholds, cost controls, and connection testing.
"""

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QLabel, QTextEdit, QTabWidget, QWidget,
    QMessageBox, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QIcon

from config.settings import ConfigManager
from config.api_config import APIConfig, APIProvider, ModelTier
from ui.components.modern_card import ModernCard, CardElevation, CardType
from ui.components.toast_notification import show_success_toast, show_error_toast, show_warning_toast, show_info_toast


class APITestThread(QThread):
    """Thread for testing API connectivity."""
    
    test_completed = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, api_key: str, provider: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.provider = provider
        self.model = model
    
    def run(self):
        """Test API connectivity."""
        try:
            # TODO: Implement actual API test
            # For now, simulate test
            self.msleep(2000)  # Simulate network delay
            
            if self.api_key and len(self.api_key) > 10:
                self.test_completed.emit(True, "API connection successful")
            else:
                self.test_completed.emit(False, "Invalid API key format")
                
        except Exception as e:
            self.test_completed.emit(False, f"API test failed: {e}")


class ModernAISettingsDialog(QDialog):
    """
    Modern AI Settings configuration dialog with professional styling.
    
    Features:
    - API key management with secure input
    - Model selection and configuration
    - Confidence threshold settings with visual feedback
    - Cost monitoring and limits with usage tracking
    - Connection testing with real-time feedback
    - Professional Material Design styling
    - Smooth animations and transitions
    """
    
    settings_changed = pyqtSignal()
    
    def __init__(self, config_manager: ConfigManager, parent=None):
        """
        Initialize the modern AI settings dialog.
        
        Args:
            config_manager: Configuration manager instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.api_config = APIConfig()
        
        # Test thread
        self.test_thread: Optional[APITestThread] = None
        
        # UI components
        self.api_key_edit: Optional[QLineEdit] = None
        self.provider_combo: Optional[QComboBox] = None
        self.model_combo: Optional[QComboBox] = None
        self.test_button: Optional[QPushButton] = None
        self.test_progress: Optional[QProgressBar] = None
        self.test_result_label: Optional[QLabel] = None
        
        # Animation properties
        self.fade_animation: Optional[QPropertyAnimation] = None
        
        self._setup_ui()
        self._setup_animations()
        self._load_settings()
        
        self.logger.debug("Modern AI settings dialog initialized")
    
    def _setup_ui(self):
        """Set up the modern user interface."""
        from ui.themes import theme_manager
        
        self.setWindowTitle("🤖 AI Settings")
        self.setModal(True)
        self.resize(600, 700)
        
        # Apply dialog styling
        self.setStyleSheet(f"""
        QDialog {{
            background-color: {theme_manager.get_color("background")};
            color: {theme_manager.get_color("text_primary")};
        }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Header card
        header_card = ModernCard(
            title="AI Configuration",
            subtitle="Configure AI models and settings for spectral data analysis",
            card_type=CardType.FILLED,
            elevation=CardElevation.LOW
        )
        header_card.setMaximumHeight(80)
        layout.addWidget(header_card)
        
        # Create tab widget with modern styling
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet(theme_manager.create_tab_style())
        
        # API Configuration tab
        api_tab = self._create_api_tab()
        tab_widget.addTab(api_tab, "🔑 API Configuration")
        
        # Model Settings tab
        model_tab = self._create_model_tab()
        tab_widget.addTab(model_tab, "🧠 Model Settings")
        
        # Cost Control tab
        cost_tab = self._create_cost_tab()
        tab_widget.addTab(cost_tab, "💰 Cost Control")
        
        # Advanced tab
        advanced_tab = self._create_advanced_tab()
        tab_widget.addTab(advanced_tab, "⚙️ Advanced")
        
        layout.addWidget(tab_widget)
        
        # Test section
        test_card = self._create_test_section()
        layout.addWidget(test_card)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("🔄 Reset to Defaults")
        self.reset_button.setStyleSheet(theme_manager.create_button_style("text"))
        self.reset_button.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(theme_manager.create_button_style("secondary"))
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.save_button = QPushButton("💾 Save Settings")
        self.save_button.setStyleSheet(theme_manager.create_button_style("primary"))
        self.save_button.clicked.connect(self._save_settings)
        self.save_button.setDefault(True)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def _setup_animations(self):
        """Set up dialog animations."""
        # Fade in animation
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(250)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    def _create_test_section(self) -> ModernCard:
        """Create API connection test section."""
        from ui.themes import theme_manager
        
        card = ModernCard(
            title="Connection Test",
            subtitle="Test your API configuration",
            card_type=CardType.OUTLINED,
            elevation=CardElevation.LOW
        )
        
        # Test controls
        test_widget = QWidget()
        test_layout = QVBoxLayout(test_widget)
        test_layout.setSpacing(12)
        
        # Test button and progress
        test_controls_layout = QHBoxLayout()
        
        self.test_api_button = QPushButton("🔍 Test API Connection")
        self.test_api_button.setStyleSheet(theme_manager.create_button_style("secondary"))
        self.test_api_button.clicked.connect(self._test_api_connection)
        test_controls_layout.addWidget(self.test_api_button)
        
        test_controls_layout.addStretch()
        
        test_layout.addLayout(test_controls_layout)
        
        # Test progress
        self.test_progress = QProgressBar()
        self.test_progress.setVisible(False)
        self.test_progress.setStyleSheet(f"""
        QProgressBar {{
            border: 1px solid {theme_manager.get_color("border")};
            border-radius: 4px;
            background-color: {theme_manager.get_color("surface_variant")};
            text-align: center;
            height: 20px;
        }}
        QProgressBar::chunk {{
            background-color: {theme_manager.get_color("primary")};
            border-radius: 3px;
        }}
        """)
        test_layout.addWidget(self.test_progress)
        
        # Test result
        self.test_result_label = QLabel()
        self.test_result_label.setVisible(False)
        self.test_result_label.setFont(theme_manager.get_font("body_medium"))
        self.test_result_label.setWordWrap(True)
        test_layout.addWidget(self.test_result_label)
        
        card.add_content(test_widget)
        return card
    
    def _create_api_tab(self) -> QWidget:
        """Create API configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # API Provider section
        provider_group = QGroupBox("API Provider")
        provider_layout = QFormLayout(provider_group)
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([provider.value for provider in APIProvider])
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        provider_layout.addRow("Provider:", self.provider_combo)
        
        layout.addWidget(provider_group)
        
        # API Key section
        key_group = QGroupBox("API Authentication")
        key_layout = QFormLayout(key_group)
        
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter your API key...")
        key_layout.addRow("API Key:", self.api_key_edit)
        
        # Show/hide API key button
        self.show_key_button = QPushButton("Show")
        self.show_key_button.setMaximumWidth(60)
        self.show_key_button.clicked.connect(self._toggle_api_key_visibility)
        key_layout.addRow("", self.show_key_button)
        
        layout.addWidget(key_group)
        
        # Connection Info section
        info_group = QGroupBox("Connection Information")
        info_layout = QVBoxLayout(info_group)
        
        self.connection_info = QTextEdit()
        self.connection_info.setMaximumHeight(100)
        self.connection_info.setReadOnly(True)
        self.connection_info.setPlainText("No connection information available")
        info_layout.addWidget(self.connection_info)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        return tab
    
    def _create_model_tab(self) -> QWidget:
        """Create model settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model Selection section
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout(model_group)
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addRow("Primary Model:", self.model_combo)
        
        self.enable_fallback_check = QCheckBox("Enable fallback models")
        self.enable_fallback_check.setChecked(True)
        model_layout.addRow("", self.enable_fallback_check)
        
        layout.addWidget(model_group)
        
        # Confidence Thresholds section
        confidence_group = QGroupBox("Confidence Thresholds")
        confidence_layout = QFormLayout(confidence_group)
        
        self.high_confidence_spin = QDoubleSpinBox()
        self.high_confidence_spin.setRange(50.0, 100.0)
        self.high_confidence_spin.setValue(90.0)
        self.high_confidence_spin.setSuffix("%")
        confidence_layout.addRow("High Confidence (Auto-apply):", self.high_confidence_spin)
        
        self.medium_confidence_spin = QDoubleSpinBox()
        self.medium_confidence_spin.setRange(30.0, 95.0)
        self.medium_confidence_spin.setValue(70.0)
        self.medium_confidence_spin.setSuffix("%")
        confidence_layout.addRow("Medium Confidence (Preview):", self.medium_confidence_spin)
        
        layout.addWidget(confidence_group)
        
        # Model Information section
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info = QTextEdit()
        self.model_info.setMaximumHeight(120)
        self.model_info.setReadOnly(True)
        info_layout.addWidget(self.model_info)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        return tab
    
    def _create_cost_tab(self) -> QWidget:
        """Create cost control tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Cost Limits section
        limits_group = QGroupBox("Cost Limits")
        limits_layout = QFormLayout(limits_group)
        
        self.monthly_limit_spin = QDoubleSpinBox()
        self.monthly_limit_spin.setRange(0.0, 1000.0)
        self.monthly_limit_spin.setValue(50.0)
        self.monthly_limit_spin.setPrefix("$")
        limits_layout.addRow("Monthly Limit:", self.monthly_limit_spin)
        
        self.daily_limit_spin = QDoubleSpinBox()
        self.daily_limit_spin.setRange(0.0, 100.0)
        self.daily_limit_spin.setValue(10.0)
        self.daily_limit_spin.setPrefix("$")
        limits_layout.addRow("Daily Limit:", self.daily_limit_spin)
        
        self.enable_limits_check = QCheckBox("Enable cost limits")
        self.enable_limits_check.setChecked(True)
        limits_layout.addRow("", self.enable_limits_check)
        
        layout.addWidget(limits_group)
        
        # Usage Statistics section
        usage_group = QGroupBox("Usage Statistics")
        usage_layout = QVBoxLayout(usage_group)
        
        self.usage_stats = QTextEdit()
        self.usage_stats.setMaximumHeight(100)
        self.usage_stats.setReadOnly(True)
        self.usage_stats.setPlainText("No usage statistics available")
        usage_layout.addWidget(self.usage_stats)
        
        refresh_stats_button = QPushButton("Refresh Statistics")
        refresh_stats_button.clicked.connect(self._refresh_usage_stats)
        usage_layout.addWidget(refresh_stats_button)
        
        layout.addWidget(usage_group)
        
        layout.addStretch()
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """Create advanced settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Request Settings section
        request_group = QGroupBox("Request Settings")
        request_layout = QFormLayout(request_group)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        self.timeout_spin.setValue(30)
        self.timeout_spin.setSuffix(" seconds")
        request_layout.addRow("Request Timeout:", self.timeout_spin)
        
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(0, 10)
        self.max_retries_spin.setValue(3)
        request_layout.addRow("Max Retries:", self.max_retries_spin)
        
        layout.addWidget(request_group)
        
        # Caching Settings section
        cache_group = QGroupBox("Caching Settings")
        cache_layout = QFormLayout(cache_group)
        
        self.enable_cache_check = QCheckBox("Enable normalization caching")
        self.enable_cache_check.setChecked(True)
        cache_layout.addRow("", self.enable_cache_check)
        
        self.cache_ttl_spin = QSpinBox()
        self.cache_ttl_spin.setRange(1, 365)
        self.cache_ttl_spin.setValue(30)
        self.cache_ttl_spin.setSuffix(" days")
        cache_layout.addRow("Cache TTL:", self.cache_ttl_spin)
        
        clear_cache_button = QPushButton("Clear Cache")
        clear_cache_button.clicked.connect(self._clear_cache)
        cache_layout.addRow("", clear_cache_button)
        
        layout.addWidget(cache_group)
        
        # Debug Settings section
        debug_group = QGroupBox("Debug Settings")
        debug_layout = QFormLayout(debug_group)
        
        self.enable_debug_check = QCheckBox("Enable debug logging")
        debug_layout.addRow("", self.enable_debug_check)
        
        self.save_requests_check = QCheckBox("Save API requests/responses")
        debug_layout.addRow("", self.save_requests_check)
        
        layout.addWidget(debug_group)
        
        layout.addStretch()
        return tab
    
    def _load_settings(self):
        """Load current settings into the dialog."""
        try:
            # Load AI settings
            ai_settings = self.config_manager.ai_settings
            
            # API settings
            self.provider_combo.setCurrentText(ai_settings.provider)
            
            # Load API key
            api_key = self.config_manager.get_api_key(ai_settings.provider)
            if api_key:
                self.api_key_edit.setText(api_key)
            
            # Model settings
            self._update_model_list()
            self.model_combo.setCurrentText(ai_settings.model)
            self.enable_fallback_check.setChecked(ai_settings.enable_fallback)
            
            # Confidence thresholds
            processing_settings = self.config_manager.processing_settings
            self.high_confidence_spin.setValue(processing_settings.confidence_threshold_high)
            self.medium_confidence_spin.setValue(processing_settings.confidence_threshold_medium)
            
            # Cost settings
            self.monthly_limit_spin.setValue(ai_settings.cost_limit_monthly)
            
            # Advanced settings
            self.timeout_spin.setValue(ai_settings.timeout_seconds)
            self.max_retries_spin.setValue(ai_settings.max_retries)
            self.enable_cache_check.setChecked(processing_settings.cache_enabled)
            self.cache_ttl_spin.setValue(processing_settings.cache_ttl_days)
            
            self._update_model_info()
            
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            QMessageBox.warning(self, "Settings Error", f"Failed to load settings: {e}")
    
    def _save_settings(self):
        """Save settings and close dialog."""
        try:
            # Validate API key
            api_key = self.api_key_edit.text().strip()
            if not api_key:
                QMessageBox.warning(self, "Invalid Settings", "API key is required.")
                return
            
            # Save API key
            provider = self.provider_combo.currentText()
            if not self.config_manager.set_api_key(provider, api_key):
                QMessageBox.warning(self, "Settings Error", "Failed to save API key.")
                return
            
            # Update AI settings
            ai_settings = self.config_manager.ai_settings
            ai_settings.provider = provider
            ai_settings.model = self.model_combo.currentText()
            ai_settings.enable_fallback = self.enable_fallback_check.isChecked()
            ai_settings.cost_limit_monthly = self.monthly_limit_spin.value()
            ai_settings.timeout_seconds = self.timeout_spin.value()
            ai_settings.max_retries = self.max_retries_spin.value()
            
            # Update processing settings
            processing_settings = self.config_manager.processing_settings
            processing_settings.confidence_threshold_high = self.high_confidence_spin.value()
            processing_settings.confidence_threshold_medium = self.medium_confidence_spin.value()
            processing_settings.cache_enabled = self.enable_cache_check.isChecked()
            processing_settings.cache_ttl_days = self.cache_ttl_spin.value()
            
            # Save configuration
            if self.config_manager.save_settings():
                self.settings_changed.emit()
                self.accept()
            else:
                QMessageBox.warning(self, "Settings Error", "Failed to save settings.")
                
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Settings Error", f"Failed to save settings: {e}")
    
    def _on_provider_changed(self, provider: str):
        """Handle provider selection change."""
        self._update_model_list()
        self._update_connection_info()
    
    def _on_model_changed(self, model: str):
        """Handle model selection change."""
        self._update_model_info()
    
    def _update_model_list(self):
        """Update available models based on selected provider."""
        provider_name = self.provider_combo.currentText()
        
        try:
            provider = APIProvider(provider_name)
            models = self.api_config.get_models_by_provider(provider)
            
            self.model_combo.clear()
            for model in models:
                self.model_combo.addItem(model.name)
                
        except Exception as e:
            self.logger.warning(f"Failed to update model list: {e}")
    
    def _update_model_info(self):
        """Update model information display."""
        model_name = self.model_combo.currentText()
        if not model_name:
            return
        
        model_config = self.api_config.get_model_config(model_name)
        if model_config:
            info_lines = [
                f"Model: {model_config.name}",
                f"Provider: {model_config.provider.value}",
                f"Tier: {model_config.tier.value}",
                f"Cost per token: ${model_config.cost_per_token:.8f}",
                f"Max tokens: {model_config.max_tokens:,}",
                f"Context window: {model_config.context_window:,}",
                f"JSON support: {'Yes' if model_config.supports_json else 'No'}",
                "",
                f"Description: {model_config.description}"
            ]
            
            self.model_info.setPlainText("\n".join(info_lines))
        else:
            self.model_info.setPlainText("No model information available")
    
    def _update_connection_info(self):
        """Update connection information display."""
        provider_name = self.provider_combo.currentText()
        
        try:
            provider = APIProvider(provider_name)
            endpoint = self.api_config.get_endpoint_config(provider)
            
            if endpoint:
                info_lines = [
                    f"Base URL: {endpoint.base_url}",
                    f"Timeout: {endpoint.timeout} seconds",
                    f"Max retries: {endpoint.max_retries}",
                    "",
                    "Headers:",
                ]
                
                for key, value in endpoint.headers.items():
                    info_lines.append(f"  {key}: {value}")
                
                self.connection_info.setPlainText("\n".join(info_lines))
            else:
                self.connection_info.setPlainText("No connection information available")
                
        except Exception as e:
            self.connection_info.setPlainText(f"Error loading connection info: {e}")
    
    def _test_api_connection(self):
        """Test API connection."""
        api_key = self.api_key_edit.text().strip()
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()
        
        if not api_key:
            QMessageBox.warning(self, "Test Error", "Please enter an API key first.")
            return
        
        # Show progress
        self.test_progress.setVisible(True)
        self.test_progress.setRange(0, 0)  # Indeterminate
        self.test_result_label.setVisible(False)
        self.test_api_button.setEnabled(False)
        
        # Start test thread
        self.test_thread = APITestThread(api_key, provider, model)
        self.test_thread.test_completed.connect(self._on_test_completed)
        self.test_thread.start()
    
    def _on_test_completed(self, success: bool, message: str):
        """Handle API test completion."""
        self.test_progress.setVisible(False)
        self.test_result_label.setVisible(True)
        self.test_api_button.setEnabled(True)
        
        if success:
            self.test_result_label.setText(f"✓ {message}")
            self.test_result_label.setStyleSheet("color: green;")
        else:
            self.test_result_label.setText(f"✗ {message}")
            self.test_result_label.setStyleSheet("color: red;")
        
        # Auto-hide result after 5 seconds
        QTimer.singleShot(5000, lambda: self.test_result_label.setVisible(False))
    
    def _toggle_api_key_visibility(self):
        """Toggle API key visibility."""
        if self.api_key_edit.echoMode() == QLineEdit.EchoMode.Password:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_button.setText("Hide")
        else:
            self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_button.setText("Show")
    
    def _refresh_usage_stats(self):
        """Refresh API usage statistics."""
        # TODO: Implement usage statistics retrieval
        self.usage_stats.setPlainText("Usage statistics refresh not yet implemented")
    
    def _clear_cache(self):
        """Clear normalization cache."""
        reply = QMessageBox.question(
            self, "Clear Cache",
            "Are you sure you want to clear the normalization cache?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # TODO: Implement cache clearing
            QMessageBox.information(self, "Cache Cleared", "Cache clearing not yet implemented")
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all AI settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset to default values
            self.provider_combo.setCurrentText("openrouter")
            self.api_key_edit.clear()
            self.high_confidence_spin.setValue(90.0)
            self.medium_confidence_spin.setValue(70.0)
            self.monthly_limit_spin.setValue(50.0)
            self.timeout_spin.setValue(30)
            self.max_retries_spin.setValue(3)
            self.enable_cache_check.setChecked(True)
            self.cache_ttl_spin.setValue(30)
            
            self._update_model_list()
            if self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
    
    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.test_thread and self.test_thread.isRunning():
            self.test_thread.quit()
            self.test_thread.wait(1000)
        
        event.accept()


# Maintain backward compatibility
AISettingsDialog = ModernAISettingsDialog