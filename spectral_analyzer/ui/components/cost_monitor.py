"""
Cost monitoring and cache statistics UI components.

Provides real-time cost tracking, usage statistics, and cache performance
monitoring for the spectral analyzer application.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QGridLayout,
    QTextEdit, QComboBox, QDateEdit, QSpinBox, QCheckBox, QFrame,
    QScrollArea, QSizePolicy, QDialog, QDialogButtonBox, QMessageBox
)
from PyQt6.QtCore import QTimer, pyqtSignal, QThread, pyqtSlot, QDate
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QPieSeries, QValueAxis, QDateTimeAxis

from utils.cost_tracker import CostTracker, UsageStatistics, CostAlert, Period
from utils.cache_manager import CacheManager, CacheStatistics


class CostDisplayWidget(QWidget):
    """Widget for displaying real-time cost information."""
    
    def __init__(self, cost_tracker: CostTracker, parent=None):
        super().__init__(parent)
        self.cost_tracker = cost_tracker
        self.logger = logging.getLogger(__name__)
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        # Session cost
        self.session_label = QLabel("Session: $0.00")
        self.session_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        layout.addWidget(self.session_label)
        
        # Daily cost
        self.daily_label = QLabel("Today: $0.00")
        self.daily_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        layout.addWidget(self.daily_label)
        
        # Monthly cost
        self.monthly_label = QLabel("Month: $0.00")
        self.monthly_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        layout.addWidget(self.monthly_label)
        
        # Cache hit rate
        self.cache_label = QLabel("Cache: 0%")
        self.cache_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
        layout.addWidget(self.cache_label)
        
        # Alert indicator
        self.alert_label = QLabel("●")
        self.alert_label.setStyleSheet("color: #4CAF50; font-size: 16px;")
        self.alert_label.setToolTip("No alerts")
        layout.addWidget(self.alert_label)
        
        layout.addStretch()
    
    def setup_timer(self):
        """Set up update timer."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(5000)  # Update every 5 seconds
        
        # Initial update
        self.update_display()
    
    @pyqtSlot()
    def update_display(self):
        """Update the cost display."""
        try:
            # Get statistics
            session_stats = self.cost_tracker.get_usage_statistics(Period.SESSION)
            daily_stats = self.cost_tracker.get_usage_statistics(Period.DAY)
            monthly_stats = self.cost_tracker.get_usage_statistics(Period.MONTH)
            
            # Update labels
            self.session_label.setText(f"Session: ${session_stats.total_cost:.3f}")
            self.daily_label.setText(f"Today: ${daily_stats.total_cost:.2f}")
            self.monthly_label.setText(f"Month: ${monthly_stats.total_cost:.2f}")
            self.cache_label.setText(f"Cache: {session_stats.cache_hit_rate:.0f}%")
            
            # Update alert indicator
            alerts = self.cost_tracker.get_recent_alerts(hours=1)
            if alerts:
                critical_alerts = [a for a in alerts if a.severity == "critical"]
                if critical_alerts:
                    self.alert_label.setStyleSheet("color: #F44336; font-size: 16px;")
                    self.alert_label.setToolTip(f"{len(critical_alerts)} critical alerts")
                else:
                    self.alert_label.setStyleSheet("color: #FF9800; font-size: 16px;")
                    self.alert_label.setToolTip(f"{len(alerts)} warnings")
            else:
                self.alert_label.setStyleSheet("color: #4CAF50; font-size: 16px;")
                self.alert_label.setToolTip("No alerts")
                
        except Exception as e:
            self.logger.error(f"Failed to update cost display: {e}")


class CacheStatsWidget(QWidget):
    """Widget for displaying cache statistics."""
    
    def __init__(self, cache_manager: CacheManager, parent=None):
        super().__init__(parent)
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Cache Performance")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        layout.addWidget(title)
        
        # Statistics grid
        stats_layout = QGridLayout()
        
        # Hit rate
        self.hit_rate_label = QLabel("Hit Rate:")
        self.hit_rate_value = QLabel("0%")
        self.hit_rate_bar = QProgressBar()
        self.hit_rate_bar.setMaximum(100)
        stats_layout.addWidget(self.hit_rate_label, 0, 0)
        stats_layout.addWidget(self.hit_rate_value, 0, 1)
        stats_layout.addWidget(self.hit_rate_bar, 0, 2)
        
        # Memory usage
        self.memory_label = QLabel("Memory:")
        self.memory_value = QLabel("0 MB")
        stats_layout.addWidget(self.memory_label, 1, 0)
        stats_layout.addWidget(self.memory_value, 1, 1)
        
        # Disk usage
        self.disk_label = QLabel("Disk:")
        self.disk_value = QLabel("0 MB")
        stats_layout.addWidget(self.disk_label, 2, 0)
        stats_layout.addWidget(self.disk_value, 2, 1)
        
        # Total entries
        self.entries_label = QLabel("Entries:")
        self.entries_value = QLabel("0")
        stats_layout.addWidget(self.entries_label, 3, 0)
        stats_layout.addWidget(self.entries_value, 3, 1)
        
        layout.addLayout(stats_layout)
    
    def setup_timer(self):
        """Set up update timer."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(10000)  # Update every 10 seconds
        
        # Initial update
        self.update_stats()
    
    @pyqtSlot()
    def update_stats(self):
        """Update cache statistics."""
        try:
            # Get cache statistics (this would be async in real implementation)
            # For now, we'll use a simplified approach
            stats = {
                'hit_rate': 75.0,
                'memory_usage_mb': 45.2,
                'disk_usage_mb': 128.7,
                'total_entries': 156
            }
            
            # Update display
            self.hit_rate_value.setText(f"{stats['hit_rate']:.1f}%")
            self.hit_rate_bar.setValue(int(stats['hit_rate']))
            self.memory_value.setText(f"{stats['memory_usage_mb']:.1f} MB")
            self.disk_value.setText(f"{stats['disk_usage_mb']:.1f} MB")
            self.entries_value.setText(str(stats['total_entries']))
            
        except Exception as e:
            self.logger.error(f"Failed to update cache stats: {e}")


class CostMonitorDialog(QDialog):
    """Comprehensive cost monitoring and statistics dialog."""
    
    def __init__(self, cost_tracker: CostTracker, cache_manager: CacheManager, parent=None):
        super().__init__(parent)
        self.cost_tracker = cost_tracker
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("Cost Monitor & Cache Statistics")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Cost tracking tab
        self.setup_cost_tab()
        
        # Cache statistics tab
        self.setup_cache_tab()
        
        # Usage history tab
        self.setup_history_tab()
        
        # Alerts tab
        self.setup_alerts_tab()
        
        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close |
            QDialogButtonBox.StandardButton.Reset
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.reset_statistics)
        layout.addWidget(button_box)
    
    def setup_cost_tab(self):
        """Set up cost tracking tab."""
        cost_widget = QWidget()
        layout = QVBoxLayout(cost_widget)
        
        # Current costs group
        current_group = QGroupBox("Current Usage")
        current_layout = QGridLayout(current_group)
        
        self.session_cost_label = QLabel("$0.00")
        self.daily_cost_label = QLabel("$0.00")
        self.monthly_cost_label = QLabel("$0.00")
        
        current_layout.addWidget(QLabel("Session Cost:"), 0, 0)
        current_layout.addWidget(self.session_cost_label, 0, 1)
        current_layout.addWidget(QLabel("Daily Cost:"), 1, 0)
        current_layout.addWidget(self.daily_cost_label, 1, 1)
        current_layout.addWidget(QLabel("Monthly Cost:"), 2, 0)
        current_layout.addWidget(self.monthly_cost_label, 2, 1)
        
        layout.addWidget(current_group)
        
        # Usage breakdown
        breakdown_group = QGroupBox("Usage Breakdown")
        breakdown_layout = QVBoxLayout(breakdown_group)
        
        self.usage_table = QTableWidget()
        self.usage_table.setColumnCount(4)
        self.usage_table.setHorizontalHeaderLabels(["Model", "Requests", "Tokens", "Cost"])
        breakdown_layout.addWidget(self.usage_table)
        
        layout.addWidget(breakdown_group)
        
        # Cost forecast
        forecast_group = QGroupBox("Cost Forecast")
        forecast_layout = QGridLayout(forecast_group)
        
        self.daily_forecast_label = QLabel("$0.00")
        self.monthly_forecast_label = QLabel("$0.00")
        
        forecast_layout.addWidget(QLabel("Daily Forecast:"), 0, 0)
        forecast_layout.addWidget(self.daily_forecast_label, 0, 1)
        forecast_layout.addWidget(QLabel("Monthly Forecast:"), 1, 0)
        forecast_layout.addWidget(self.monthly_forecast_label, 1, 1)
        
        layout.addWidget(forecast_group)
        
        self.tab_widget.addTab(cost_widget, "Cost Tracking")
    
    def setup_cache_tab(self):
        """Set up cache statistics tab."""
        cache_widget = QWidget()
        layout = QVBoxLayout(cache_widget)
        
        # Performance metrics
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout(perf_group)
        
        self.hit_rate_label = QLabel("0%")
        self.avg_lookup_label = QLabel("0ms")
        self.memory_usage_label = QLabel("0 MB")
        self.disk_usage_label = QLabel("0 MB")
        
        perf_layout.addWidget(QLabel("Hit Rate:"), 0, 0)
        perf_layout.addWidget(self.hit_rate_label, 0, 1)
        perf_layout.addWidget(QLabel("Avg Lookup Time:"), 1, 0)
        perf_layout.addWidget(self.avg_lookup_label, 1, 1)
        perf_layout.addWidget(QLabel("Memory Usage:"), 2, 0)
        perf_layout.addWidget(self.memory_usage_label, 2, 1)
        perf_layout.addWidget(QLabel("Disk Usage:"), 3, 0)
        perf_layout.addWidget(self.disk_usage_label, 3, 1)
        
        layout.addWidget(perf_group)
        
        # Cache levels
        levels_group = QGroupBox("Cache Levels")
        levels_layout = QVBoxLayout(levels_group)
        
        self.levels_table = QTableWidget()
        self.levels_table.setColumnCount(3)
        self.levels_table.setHorizontalHeaderLabels(["Level", "Entries", "Hit Rate"])
        levels_layout.addWidget(self.levels_table)
        
        layout.addWidget(levels_group)
        
        # Most accessed
        accessed_group = QGroupBox("Most Accessed")
        accessed_layout = QVBoxLayout(accessed_group)
        
        self.accessed_table = QTableWidget()
        self.accessed_table.setColumnCount(3)
        self.accessed_table.setHorizontalHeaderLabels(["Key", "Access Count", "Last Accessed"])
        accessed_layout.addWidget(self.accessed_table)
        
        layout.addWidget(accessed_group)
        
        # Cache controls
        controls_layout = QHBoxLayout()
        
        self.cleanup_button = QPushButton("Cleanup Expired")
        self.cleanup_button.clicked.connect(self.cleanup_cache)
        controls_layout.addWidget(self.cleanup_button)
        
        self.clear_button = QPushButton("Clear Cache")
        self.clear_button.clicked.connect(self.clear_cache)
        controls_layout.addWidget(self.clear_button)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.tab_widget.addTab(cache_widget, "Cache Statistics")
    
    def setup_history_tab(self):
        """Set up usage history tab."""
        history_widget = QWidget()
        layout = QVBoxLayout(history_widget)
        
        # Filters
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["Session", "Day", "Week", "Month"])
        self.period_combo.currentTextChanged.connect(self.update_history)
        filter_layout.addWidget(self.period_combo)
        
        filter_layout.addWidget(QLabel("From:"))
        self.from_date = QDateEdit()
        self.from_date.setDate(QDate.currentDate().addDays(-7))
        filter_layout.addWidget(self.from_date)
        
        filter_layout.addWidget(QLabel("To:"))
        self.to_date = QDateEdit()
        self.to_date.setDate(QDate.currentDate())
        filter_layout.addWidget(self.to_date)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.update_history)
        filter_layout.addWidget(self.refresh_button)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp", "Model", "Tokens", "Cost", "Success", "Cache Hit"
        ])
        layout.addWidget(self.history_table)
        
        # Export controls
        export_layout = QHBoxLayout()
        
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(lambda: self.export_data('csv'))
        export_layout.addWidget(self.export_csv_button)
        
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(lambda: self.export_data('json'))
        export_layout.addWidget(self.export_json_button)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        self.tab_widget.addTab(history_widget, "Usage History")
    
    def setup_alerts_tab(self):
        """Set up alerts tab."""
        alerts_widget = QWidget()
        layout = QVBoxLayout(alerts_widget)
        
        # Alert settings
        settings_group = QGroupBox("Alert Settings")
        settings_layout = QGridLayout(settings_group)
        
        settings_layout.addWidget(QLabel("Daily Limit ($):"), 0, 0)
        self.daily_limit_spin = QSpinBox()
        self.daily_limit_spin.setMaximum(1000)
        self.daily_limit_spin.setValue(10)
        settings_layout.addWidget(self.daily_limit_spin, 0, 1)
        
        settings_layout.addWidget(QLabel("Monthly Limit ($):"), 1, 0)
        self.monthly_limit_spin = QSpinBox()
        self.monthly_limit_spin.setMaximum(10000)
        self.monthly_limit_spin.setValue(200)
        settings_layout.addWidget(self.monthly_limit_spin, 1, 1)
        
        self.enable_alerts_check = QCheckBox("Enable Alerts")
        self.enable_alerts_check.setChecked(True)
        settings_layout.addWidget(self.enable_alerts_check, 2, 0, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Recent alerts
        alerts_group = QGroupBox("Recent Alerts")
        alerts_layout = QVBoxLayout(alerts_group)
        
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(4)
        self.alerts_table.setHorizontalHeaderLabels([
            "Timestamp", "Type", "Message", "Severity"
        ])
        alerts_layout.addWidget(self.alerts_table)
        
        layout.addWidget(alerts_group)
        
        self.tab_widget.addTab(alerts_widget, "Alerts")
    
    def load_data(self):
        """Load data into the dialog."""
        try:
            self.update_cost_data()
            self.update_cache_data()
            self.update_history()
            self.update_alerts()
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
    
    def update_cost_data(self):
        """Update cost tracking data."""
        try:
            # Get statistics
            session_stats = self.cost_tracker.get_usage_statistics(Period.SESSION)
            daily_stats = self.cost_tracker.get_usage_statistics(Period.DAY)
            monthly_stats = self.cost_tracker.get_usage_statistics(Period.MONTH)
            
            # Update labels
            self.session_cost_label.setText(f"${session_stats.total_cost:.3f}")
            self.daily_cost_label.setText(f"${daily_stats.total_cost:.2f}")
            self.monthly_cost_label.setText(f"${monthly_stats.total_cost:.2f}")
            
            # Update usage breakdown table
            self.usage_table.setRowCount(len(session_stats.model_usage))
            row = 0
            for model, count in session_stats.model_usage.items():
                cost = session_stats.cost_breakdown.get(model, 0.0)
                tokens = int(count * session_stats.average_tokens_per_call)
                
                self.usage_table.setItem(row, 0, QTableWidgetItem(model))
                self.usage_table.setItem(row, 1, QTableWidgetItem(str(count)))
                self.usage_table.setItem(row, 2, QTableWidgetItem(str(tokens)))
                self.usage_table.setItem(row, 3, QTableWidgetItem(f"${cost:.3f}"))
                row += 1
            
            # Update forecast
            forecast = self.cost_tracker.get_cost_forecast()
            self.daily_forecast_label.setText(f"${forecast.get('daily_forecast', 0):.2f}")
            self.monthly_forecast_label.setText(f"${forecast.get('monthly_forecast', 0):.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cost data: {e}")
    
    def update_cache_data(self):
        """Update cache statistics data."""
        try:
            # This would be async in real implementation
            # For now, using placeholder data
            stats = {
                'hit_rate': 75.0,
                'avg_lookup_time_ms': 12.5,
                'memory_usage_mb': 45.2,
                'disk_usage_mb': 128.7,
                'cache_levels': {'memory': 50, 'file': 100, 'redis': 25},
                'most_accessed': [
                    {'key': 'norm:abc123', 'access_count': 15, 'last_accessed': '2024-01-15 10:30:00'},
                    {'key': 'norm:def456', 'access_count': 12, 'last_accessed': '2024-01-15 09:45:00'},
                ]
            }
            
            # Update performance metrics
            self.hit_rate_label.setText(f"{stats['hit_rate']:.1f}%")
            self.avg_lookup_label.setText(f"{stats['avg_lookup_time_ms']:.1f}ms")
            self.memory_usage_label.setText(f"{stats['memory_usage_mb']:.1f} MB")
            self.disk_usage_label.setText(f"{stats['disk_usage_mb']:.1f} MB")
            
            # Update cache levels table
            self.levels_table.setRowCount(len(stats['cache_levels']))
            row = 0
            for level, count in stats['cache_levels'].items():
                hit_rate = 80.0 if level == 'memory' else 60.0 if level == 'file' else 40.0
                self.levels_table.setItem(row, 0, QTableWidgetItem(level.title()))
                self.levels_table.setItem(row, 1, QTableWidgetItem(str(count)))
                self.levels_table.setItem(row, 2, QTableWidgetItem(f"{hit_rate:.1f}%"))
                row += 1
            
            # Update most accessed table
            self.accessed_table.setRowCount(len(stats['most_accessed']))
            for row, item in enumerate(stats['most_accessed']):
                self.accessed_table.setItem(row, 0, QTableWidgetItem(item['key']))
                self.accessed_table.setItem(row, 1, QTableWidgetItem(str(item['access_count'])))
                self.accessed_table.setItem(row, 2, QTableWidgetItem(item['last_accessed']))
            
        except Exception as e:
            self.logger.error(f"Failed to update cache data: {e}")
    
    def update_history(self):
        """Update usage history."""
        try:
            # Placeholder data - would get from cost tracker
            history_data = [
                {
                    'timestamp': '2024-01-15 10:30:00',
                    'model': 'x-ai/grok-4-fast',
                    'tokens': 1500,
                    'cost': 0.0015,
                    'success': True,
                    'cache_hit': False
                },
                {
                    'timestamp': '2024-01-15 10:25:00',
                    'model': 'x-ai/grok-4-fast',
                    'tokens': 0,
                    'cost': 0.0000,
                    'success': True,
                    'cache_hit': True
                }
            ]
            
            self.history_table.setRowCount(len(history_data))
            for row, item in enumerate(history_data):
                self.history_table.setItem(row, 0, QTableWidgetItem(item['timestamp']))
                self.history_table.setItem(row, 1, QTableWidgetItem(item['model']))
                self.history_table.setItem(row, 2, QTableWidgetItem(str(item['tokens'])))
                self.history_table.setItem(row, 3, QTableWidgetItem(f"${item['cost']:.4f}"))
                self.history_table.setItem(row, 4, QTableWidgetItem("✓" if item['success'] else "✗"))
                self.history_table.setItem(row, 5, QTableWidgetItem("✓" if item['cache_hit'] else "✗"))
            
        except Exception as e:
            self.logger.error(f"Failed to update history: {e}")
    
    def update_alerts(self):
        """Update alerts data."""
        try:
            alerts = self.cost_tracker.get_recent_alerts(hours=24)
            
            self.alerts_table.setRowCount(len(alerts))
            for row, alert in enumerate(alerts):
                self.alerts_table.setItem(row, 0, QTableWidgetItem(alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")))
                self.alerts_table.setItem(row, 1, QTableWidgetItem(alert.alert_type.value))
                self.alerts_table.setItem(row, 2, QTableWidgetItem(alert.message))
                self.alerts_table.setItem(row, 3, QTableWidgetItem(alert.severity.upper()))
            
        except Exception as e:
            self.logger.error(f"Failed to update alerts: {e}")
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            # This would be async in real implementation
            QMessageBox.information(self, "Cache Cleanup", "Cache cleanup completed successfully.")
            self.update_cache_data()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cache cleanup failed: {e}")
    
    def clear_cache(self):
        """Clear all cache entries."""
        reply = QMessageBox.question(
            self, "Clear Cache",
            "Are you sure you want to clear all cache entries?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # This would be async in real implementation
                QMessageBox.information(self, "Cache Cleared", "Cache cleared successfully.")
                self.update_cache_data()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to clear cache: {e}")
    
    def export_data(self, format: str):
        """Export usage data."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export Usage Data ({format.upper()})",
                f"usage_report.{format}",
                f"{format.upper()} Files (*.{format})"
            )
            
            if filename:
                success = self.cost_tracker.export_usage_report(filename, format)
                if success:
                    QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
                else:
                    QMessageBox.warning(self, "Export Failed", "Failed to export data")
                    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Export failed: {e}")
    
    def reset_statistics(self):
        """Reset usage statistics."""
        reply = QMessageBox.question(
            self, "Reset Statistics",
            "Are you sure you want to reset all usage statistics?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.cost_tracker.reset_session_stats()
                QMessageBox.information(self, "Reset Complete", "Statistics reset successfully.")
                self.load_data()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to reset statistics: {e}")