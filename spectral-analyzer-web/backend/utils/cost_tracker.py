"""
Cost tracking and monitoring system for AI API usage.

Provides comprehensive cost tracking, usage statistics, and budget monitoring
for AI API calls with persistent storage and reporting capabilities.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import csv
import pandas as pd


class AlertType(Enum):
    """Types of cost alerts."""
    DAILY_LIMIT = "daily_limit"
    MONTHLY_LIMIT = "monthly_limit"
    HIGH_USAGE = "high_usage"
    BUDGET_WARNING = "budget_warning"
    UNUSUAL_PATTERN = "unusual_pattern"


class Period(Enum):
    """Time periods for usage statistics."""
    SESSION = "session"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class CostAlert:
    """Cost alert information."""
    alert_type: AlertType
    current_cost: float
    limit: float
    message: str
    timestamp: datetime
    severity: str = "warning"  # info, warning, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_type': self.alert_type.value,
            'current_cost': self.current_cost,
            'limit': self.limit,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity
        }


@dataclass
class UsageStatistics:
    """Comprehensive usage statistics for a time period."""
    period: str
    start_time: datetime
    end_time: datetime
    api_calls: int
    tokens_used: int
    total_cost: float
    cache_hits: int
    cache_misses: int
    cache_savings: float
    most_used_model: str
    cost_breakdown: Dict[str, float]
    model_usage: Dict[str, int]
    average_cost_per_call: float
    average_tokens_per_call: float
    peak_usage_hour: Optional[str] = None
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(1, total_requests)) * 100
    
    @property
    def cost_per_token(self) -> float:
        """Calculate average cost per token."""
        return self.total_cost / max(1, self.tokens_used)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'period': self.period,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'api_calls': self.api_calls,
            'tokens_used': self.tokens_used,
            'total_cost': self.total_cost,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_savings': self.cache_savings,
            'most_used_model': self.most_used_model,
            'cost_breakdown': self.cost_breakdown,
            'model_usage': self.model_usage,
            'average_cost_per_call': self.average_cost_per_call,
            'average_tokens_per_call': self.average_tokens_per_call,
            'cache_hit_rate': self.cache_hit_rate,
            'cost_per_token': self.cost_per_token,
            'peak_usage_hour': self.peak_usage_hour
        }


@dataclass
class APICallRecord:
    """Record of a single API call."""
    timestamp: datetime
    model: str
    provider: str
    tokens_used: int
    cost: float
    response_time: float
    success: bool
    cache_hit: bool = False
    operation_type: str = "normalization"
    file_hash: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'model': self.model,
            'provider': self.provider,
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'response_time': self.response_time,
            'success': self.success,
            'cache_hit': self.cache_hit,
            'operation_type': self.operation_type,
            'file_hash': self.file_hash,
            'error_message': self.error_message
        }


class CostTracker:
    """
    Comprehensive cost tracking and monitoring system.
    
    Features:
    - Real-time API usage tracking
    - Cost estimation and monitoring
    - Budget alerts and warnings
    - Usage statistics and reporting
    - Cache savings calculation
    - Export capabilities
    """
    
    def __init__(self, storage_path: Optional[str] = None, config_manager=None):
        """
        Initialize cost tracker.
        
        Args:
            storage_path: Path for persistent storage
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Storage setup
        if storage_path:
            self.storage_dir = Path(storage_path)
        else:
            self.storage_dir = Path.home() / ".spectral_analyzer" / "usage"
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "usage_tracking.db"
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_calls = []
        
        # Budget limits (from config or defaults)
        self.daily_limit = 10.0  # $10 per day
        self.monthly_limit = 200.0  # $200 per month
        self.warning_threshold = 0.8  # 80% of limit
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self._load_config()
        
        self.logger.debug("Cost tracker initialized")
    
    def _init_database(self):
        """Initialize SQLite database for usage tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # API calls table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        tokens_used INTEGER NOT NULL,
                        cost REAL NOT NULL,
                        response_time REAL NOT NULL,
                        success BOOLEAN NOT NULL,
                        cache_hit BOOLEAN DEFAULT FALSE,
                        operation_type TEXT DEFAULT 'normalization',
                        file_hash TEXT,
                        error_message TEXT
                    )
                """)
                
                # Usage summaries table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS usage_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        period_type TEXT NOT NULL,
                        period_start TEXT NOT NULL,
                        period_end TEXT NOT NULL,
                        api_calls INTEGER NOT NULL,
                        tokens_used INTEGER NOT NULL,
                        total_cost REAL NOT NULL,
                        cache_hits INTEGER NOT NULL,
                        cache_misses INTEGER NOT NULL,
                        cache_savings REAL NOT NULL,
                        most_used_model TEXT,
                        cost_breakdown TEXT,
                        model_usage TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                # Cost alerts table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cost_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT NOT NULL,
                        current_cost REAL NOT NULL,
                        limit_value REAL NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        acknowledged BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON api_calls(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON api_calls(model)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_period ON usage_summaries(period_type, period_start)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize cost tracking database: {e}")
    
    def _load_config(self):
        """Load configuration from config manager."""
        if self.config_manager:
            try:
                self.daily_limit = self.config_manager.get_setting('ai', 'daily_cost_limit', 10.0)
                self.monthly_limit = self.config_manager.get_setting('ai', 'monthly_cost_limit', 200.0)
                self.warning_threshold = self.config_manager.get_setting('ai', 'warning_threshold', 0.8)
            except Exception as e:
                self.logger.warning(f"Failed to load cost tracking config: {e}")
    
    def track_api_call(self, model: str, provider: str, tokens_used: int, 
                      cost: float, response_time: float, success: bool = True,
                      cache_hit: bool = False, operation_type: str = "normalization",
                      file_hash: Optional[str] = None, error_message: Optional[str] = None):
        """
        Track an API call with comprehensive details.
        
        Args:
            model: AI model used
            provider: API provider
            tokens_used: Number of tokens consumed
            cost: Cost of the API call
            response_time: Response time in seconds
            success: Whether the call was successful
            cache_hit: Whether this was a cache hit
            operation_type: Type of operation performed
            file_hash: Optional file hash for caching
            error_message: Error message if call failed
        """
        try:
            # Create call record
            call_record = APICallRecord(
                timestamp=datetime.now(),
                model=model,
                provider=provider,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                success=success,
                cache_hit=cache_hit,
                operation_type=operation_type,
                file_hash=file_hash,
                error_message=error_message
            )
            
            # Add to session tracking
            self.session_calls.append(call_record)
            
            # Store in database
            self._store_api_call(call_record)
            
            # Check for alerts
            self._check_cost_alerts()
            
            self.logger.debug(f"Tracked API call: {model}, cost: ${cost:.4f}, tokens: {tokens_used}")
            
        except Exception as e:
            self.logger.error(f"Failed to track API call: {e}")
    
    def record_api_usage(self, operation: str, cost: float, tokens_used: int,
                        model: str = "unknown", success: bool = True):
        """
        Record API usage - compatibility method for integration tests.
        
        Args:
            operation: Type of operation performed
            cost: Cost of the API call
            tokens_used: Number of tokens consumed
            model: AI model used (optional)
            success: Whether the operation was successful
        """
        self.track_api_call(
            model=model,
            provider="openrouter",
            tokens_used=tokens_used,
            cost=cost,
            response_time=0.0,
            success=success,
            cache_hit=False,
            operation_type=operation
        )
    
    def _store_api_call(self, call_record: APICallRecord):
        """Store API call record in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_calls 
                    (timestamp, model, provider, tokens_used, cost, response_time, 
                     success, cache_hit, operation_type, file_hash, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    call_record.timestamp.isoformat(),
                    call_record.model,
                    call_record.provider,
                    call_record.tokens_used,
                    call_record.cost,
                    call_record.response_time,
                    call_record.success,
                    call_record.cache_hit,
                    call_record.operation_type,
                    call_record.file_hash,
                    call_record.error_message
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store API call record: {e}")
    
    def get_usage_statistics(self, period: Union[str, Period] = Period.SESSION,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> UsageStatistics:
        """
        Get usage statistics for a specified period.
        
        Args:
            period: Time period for statistics
            start_time: Optional custom start time
            end_time: Optional custom end time
            
        Returns:
            UsageStatistics object with comprehensive data
        """
        try:
            if isinstance(period, str):
                period = Period(period)
            
            # Determine time range
            if period == Period.SESSION:
                start_time = self.session_start
                end_time = datetime.now()
                calls = self.session_calls
            else:
                if not start_time or not end_time:
                    start_time, end_time = self._get_period_range(period)
                calls = self._get_calls_in_range(start_time, end_time)
            
            # Calculate statistics
            stats = self._calculate_statistics(calls, period.value, start_time, end_time)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get usage statistics: {e}")
            return self._empty_statistics(period.value, start_time or datetime.now(), end_time or datetime.now())
    
    def _get_period_range(self, period: Period) -> tuple[datetime, datetime]:
        """Get start and end times for a period."""
        now = datetime.now()
        
        if period == Period.HOUR:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = now
        elif period == Period.DAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == Period.WEEK:
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == Period.MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == Period.YEAR:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = now
        else:
            start = now - timedelta(days=1)
            end = now
        
        return start, end
    
    def _get_calls_in_range(self, start_time: datetime, end_time: datetime) -> List[APICallRecord]:
        """Get API calls within a time range from database."""
        try:
            calls = []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, model, provider, tokens_used, cost, response_time,
                           success, cache_hit, operation_type, file_hash, error_message
                    FROM api_calls
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """, (start_time.isoformat(), end_time.isoformat()))
                
                for row in cursor.fetchall():
                    call = APICallRecord(
                        timestamp=datetime.fromisoformat(row[0]),
                        model=row[1],
                        provider=row[2],
                        tokens_used=row[3],
                        cost=row[4],
                        response_time=row[5],
                        success=bool(row[6]),
                        cache_hit=bool(row[7]),
                        operation_type=row[8] or "normalization",
                        file_hash=row[9],
                        error_message=row[10]
                    )
                    calls.append(call)
            
            return calls
            
        except Exception as e:
            self.logger.error(f"Failed to get calls in range: {e}")
            return []
    
    def _calculate_statistics(self, calls: List[APICallRecord], period: str,
                            start_time: datetime, end_time: datetime) -> UsageStatistics:
        """Calculate comprehensive statistics from API calls."""
        if not calls:
            return self._empty_statistics(period, start_time, end_time)
        
        # Basic counts
        api_calls = len(calls)
        successful_calls = [c for c in calls if c.success]
        cache_hits = len([c for c in calls if c.cache_hit])
        cache_misses = len([c for c in calls if not c.cache_hit])
        
        # Totals
        total_tokens = sum(c.tokens_used for c in calls)
        total_cost = sum(c.cost for c in calls)
        
        # Model usage
        model_usage = {}
        cost_breakdown = {}
        
        for call in calls:
            model_usage[call.model] = model_usage.get(call.model, 0) + 1
            cost_breakdown[call.model] = cost_breakdown.get(call.model, 0.0) + call.cost
        
        # Most used model
        most_used_model = max(model_usage.keys(), key=model_usage.get) if model_usage else "none"
        
        # Averages
        avg_cost_per_call = total_cost / max(1, len(successful_calls))
        avg_tokens_per_call = total_tokens / max(1, len(successful_calls))
        
        # Cache savings calculation
        avg_cost_per_call_no_cache = sum(c.cost for c in calls if not c.cache_hit) / max(1, cache_misses)
        cache_savings = cache_hits * avg_cost_per_call_no_cache
        
        # Peak usage hour (for day/week/month periods)
        peak_usage_hour = self._calculate_peak_usage_hour(calls) if len(calls) > 10 else None
        
        return UsageStatistics(
            period=period,
            start_time=start_time,
            end_time=end_time,
            api_calls=api_calls,
            tokens_used=total_tokens,
            total_cost=total_cost,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_savings=cache_savings,
            most_used_model=most_used_model,
            cost_breakdown=cost_breakdown,
            model_usage=model_usage,
            average_cost_per_call=avg_cost_per_call,
            average_tokens_per_call=avg_tokens_per_call,
            peak_usage_hour=peak_usage_hour
        )
    
    def _empty_statistics(self, period: str, start_time: datetime, end_time: datetime) -> UsageStatistics:
        """Create empty statistics object."""
        return UsageStatistics(
            period=period,
            start_time=start_time,
            end_time=end_time,
            api_calls=0,
            tokens_used=0,
            total_cost=0.0,
            cache_hits=0,
            cache_misses=0,
            cache_savings=0.0,
            most_used_model="none",
            cost_breakdown={},
            model_usage={},
            average_cost_per_call=0.0,
            average_tokens_per_call=0.0
        )
    
    def _calculate_peak_usage_hour(self, calls: List[APICallRecord]) -> Optional[str]:
        """Calculate peak usage hour from calls."""
        try:
            hour_counts = {}
            for call in calls:
                hour = call.timestamp.strftime("%H:00")
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            if hour_counts:
                return max(hour_counts.keys(), key=hour_counts.get)
            
        except Exception:
            pass
        
        return None
    
    def calculate_cache_savings(self, cache_hits: int, avg_cost_per_call: float) -> float:
        """
        Calculate money saved through caching.
        
        Args:
            cache_hits: Number of cache hits
            avg_cost_per_call: Average cost per API call
            
        Returns:
            Total savings in USD
        """
        return cache_hits * avg_cost_per_call
    
    def _check_cost_alerts(self):
        """Check for cost alerts and warnings."""
        try:
            # Get current usage
            daily_stats = self.get_usage_statistics(Period.DAY)
            monthly_stats = self.get_usage_statistics(Period.MONTH)
            
            alerts = []
            
            # Daily limit check
            if daily_stats.total_cost >= self.daily_limit:
                alerts.append(CostAlert(
                    alert_type=AlertType.DAILY_LIMIT,
                    current_cost=daily_stats.total_cost,
                    limit=self.daily_limit,
                    message=f"Daily cost limit exceeded: ${daily_stats.total_cost:.2f} / ${self.daily_limit:.2f}",
                    timestamp=datetime.now(),
                    severity="critical"
                ))
            elif daily_stats.total_cost >= self.daily_limit * self.warning_threshold:
                alerts.append(CostAlert(
                    alert_type=AlertType.BUDGET_WARNING,
                    current_cost=daily_stats.total_cost,
                    limit=self.daily_limit,
                    message=f"Daily cost warning: ${daily_stats.total_cost:.2f} / ${self.daily_limit:.2f} ({daily_stats.total_cost/self.daily_limit*100:.1f}%)",
                    timestamp=datetime.now(),
                    severity="warning"
                ))
            
            # Monthly limit check
            if monthly_stats.total_cost >= self.monthly_limit:
                alerts.append(CostAlert(
                    alert_type=AlertType.MONTHLY_LIMIT,
                    current_cost=monthly_stats.total_cost,
                    limit=self.monthly_limit,
                    message=f"Monthly cost limit exceeded: ${monthly_stats.total_cost:.2f} / ${self.monthly_limit:.2f}",
                    timestamp=datetime.now(),
                    severity="critical"
                ))
            elif monthly_stats.total_cost >= self.monthly_limit * self.warning_threshold:
                alerts.append(CostAlert(
                    alert_type=AlertType.BUDGET_WARNING,
                    current_cost=monthly_stats.total_cost,
                    limit=self.monthly_limit,
                    message=f"Monthly cost warning: ${monthly_stats.total_cost:.2f} / ${self.monthly_limit:.2f} ({monthly_stats.total_cost/self.monthly_limit*100:.1f}%)",
                    timestamp=datetime.now(),
                    severity="warning"
                ))
            
            # Store alerts
            for alert in alerts:
                self._store_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to check cost alerts: {e}")
    
    def _store_alert(self, alert: CostAlert):
        """Store cost alert in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cost_alerts 
                    (alert_type, current_cost, limit_value, message, timestamp, severity)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_type.value,
                    alert.current_cost,
                    alert.limit,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.severity
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store cost alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[CostAlert]:
        """Get recent cost alerts."""
        try:
            alerts = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT alert_type, current_cost, limit_value, message, timestamp, severity
                    FROM cost_alerts
                    WHERE timestamp > ? AND acknowledged = FALSE
                    ORDER BY timestamp DESC
                """, (cutoff_time.isoformat(),))
                
                for row in cursor.fetchall():
                    alert = CostAlert(
                        alert_type=AlertType(row[0]),
                        current_cost=row[1],
                        limit=row[2],
                        message=row[3],
                        timestamp=datetime.fromisoformat(row[4]),
                        severity=row[5]
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get recent alerts: {e}")
            return []
    
    def export_usage_report(self, filepath: str, format: str = 'csv',
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> bool:
        """
        Export detailed usage report.
        
        Args:
            filepath: Output file path
            format: Export format ('csv', 'json', 'excel')
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            bool: True if export successful
        """
        try:
            # Get data range
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Get call records
            calls = self._get_calls_in_range(start_date, end_date)
            
            if format.lower() == 'csv':
                return self._export_csv(calls, filepath)
            elif format.lower() == 'json':
                return self._export_json(calls, filepath)
            elif format.lower() == 'excel':
                return self._export_excel(calls, filepath)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to export usage report: {e}")
            return False
    
    def _export_csv(self, calls: List[APICallRecord], filepath: str) -> bool:
        """Export calls to CSV format."""
        try:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'model', 'provider', 'tokens_used', 'cost',
                    'response_time', 'success', 'cache_hit', 'operation_type',
                    'file_hash', 'error_message'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for call in calls:
                    writer.writerow(call.to_dict())
            
            self.logger.info(f"Usage report exported to CSV: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return False
    
    def _export_json(self, calls: List[APICallRecord], filepath: str) -> bool:
        """Export calls to JSON format."""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_calls': len(calls),
                'calls': [call.to_dict() for call in calls]
            }
            
            with open(filepath, 'w') as jsonfile:
                json.dump(data, jsonfile, indent=2)
            
            self.logger.info(f"Usage report exported to JSON: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return False
    
    def _export_excel(self, calls: List[APICallRecord], filepath: str) -> bool:
        """Export calls to Excel format."""
        try:
            # Convert to DataFrame
            data = [call.to_dict() for call in calls]
            df = pd.DataFrame(data)
            
            # Export to Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='API_Calls', index=False)
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Calls', 'Total Cost', 'Total Tokens', 'Cache Hits', 'Average Cost/Call'],
                    'Value': [
                        len(calls),
                        sum(c.cost for c in calls),
                        sum(c.tokens_used for c in calls),
                        len([c for c in calls if c.cache_hit]),
                        sum(c.cost for c in calls) / max(1, len(calls))
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"Usage report exported to Excel: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export Excel: {e}")
            return False
    
    def get_cost_forecast(self, days: int = 30) -> Dict[str, float]:
        """
        Get cost forecast based on recent usage patterns.
        
        Args:
            days: Number of days to forecast
            
        Returns:
            Dictionary with forecast data
        """
        try:
            # Get recent usage for trend analysis
            recent_stats = self.get_usage_statistics(Period.WEEK)
            
            if recent_stats.api_calls == 0:
                return {
                    'daily_forecast': 0.0,
                    'weekly_forecast': 0.0,
                    'monthly_forecast': 0.0,
                    'confidence': 'low'
                }
            
            # Calculate daily average
            days_in_period = (recent_stats.end_time - recent_stats.start_time).days or 1
            daily_avg_cost = recent_stats.total_cost / days_in_period
            daily_avg_calls = recent_stats.api_calls / days_in_period
            
            # Simple linear projection
            forecast = {
                'daily_forecast': daily_avg_cost,
                'weekly_forecast': daily_avg_cost * 7,
                'monthly_forecast': daily_avg_cost * 30,
                'daily_calls_forecast': daily_avg_calls,
                'confidence': 'medium' if recent_stats.api_calls > 10 else 'low'
            }
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Failed to generate cost forecast: {e}")
            return {'error': str(e)}
    
    def reset_session_stats(self):
        """Reset session statistics."""
        self.session_start = datetime.now()
        self.session_calls = []
        self.logger.info("Session statistics reset")
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """
        Clean up old tracking data.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                # Clean old API calls
                cursor = conn.execute(
                    "DELETE FROM api_calls WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                calls_deleted = cursor.rowcount
                
                # Clean old alerts
                cursor = conn.execute(
                    "DELETE FROM cost_alerts WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                alerts_deleted = cursor.rowcount
                
                conn.commit()
            
            self.logger.info(f"Cleaned up old data: {calls_deleted} calls, {alerts_deleted} alerts")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")