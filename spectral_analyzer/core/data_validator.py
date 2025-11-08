"""
Data validation system for spectroscopy data.

Provides comprehensive validation rules and quality checks for spectral data
to ensure data integrity and compatibility with analysis workflows.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import re


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationRule(Enum):
    """Available validation rules."""
    WAVENUMBER_RANGE = "wavenumber_range"
    WAVENUMBER_ORDER = "wavenumber_order"
    ABSORBANCE_RANGE = "absorbance_range"
    TRANSMITTANCE_RANGE = "transmittance_range"
    MISSING_VALUES = "missing_values"
    DUPLICATE_VALUES = "duplicate_values"
    DATA_CONTINUITY = "data_continuity"
    BASELINE_DRIFT = "baseline_drift"
    NOISE_LEVEL = "noise_level"
    SPECTRAL_RESOLUTION = "spectral_resolution"
    MINIMUM_DATA_POINTS = "minimum_data_points"
    OVERLAPPING_RANGES = "overlapping_ranges"
    SCALE_ISSUES = "scale_issues"
    OUTLIER_DETECTION = "outlier_detection"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    rule: ValidationRule
    level: ValidationLevel
    message: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    affected_rows: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Results of data validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    quality_score: float
    recommendations: List[str]


class DataValidator:
    """
    Comprehensive data validation system for spectroscopy data.
    
    Provides validation rules for:
    - Data range validation
    - Structural integrity checks
    - Quality assessment
    - Spectroscopy-specific validations
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = logging.getLogger(__name__)
        
        # Default validation thresholds
        self.thresholds = {
            'wavenumber_min': 400.0,
            'wavenumber_max': 4000.0,
            'absorbance_min': -0.5,
            'absorbance_max': 5.0,
            'transmittance_min': 0.0,
            'transmittance_max': 100.0,
            'max_missing_percent': 5.0,
            'max_noise_level': 0.1,
            'min_resolution': 0.5,
            'max_baseline_drift': 0.2,
            'min_data_points': 100,
            'outlier_threshold': 3.0,  # Standard deviations
            'max_scale_factor': 1000.0
        }
        
        # Validation rules configuration
        self.enabled_rules = {
            ValidationRule.WAVENUMBER_RANGE: True,
            ValidationRule.WAVENUMBER_ORDER: True,
            ValidationRule.ABSORBANCE_RANGE: True,
            ValidationRule.TRANSMITTANCE_RANGE: True,
            ValidationRule.MISSING_VALUES: True,
            ValidationRule.DUPLICATE_VALUES: True,
            ValidationRule.DATA_CONTINUITY: True,
            ValidationRule.BASELINE_DRIFT: False,  # Advanced analysis
            ValidationRule.NOISE_LEVEL: False,     # Advanced analysis
            ValidationRule.SPECTRAL_RESOLUTION: False,  # Advanced analysis
            ValidationRule.MINIMUM_DATA_POINTS: True,
            ValidationRule.OVERLAPPING_RANGES: True,
            ValidationRule.SCALE_ISSUES: True,
            ValidationRule.OUTLIER_DETECTION: False  # Advanced analysis
        }
    
    def validate_data(self, df: pd.DataFrame, data_type: str = "spectral") -> ValidationResult:
        """
        Perform comprehensive validation of spectral data.
        
        Args:
            df: DataFrame containing spectral data
            data_type: Type of data being validated
            
        Returns:
            ValidationResult with all validation findings
        """
        self.logger.info(f"Starting validation of {data_type} data")
        
        issues = []
        statistics = {}
        
        try:
            # Basic structure validation
            structure_issues = self._validate_structure(df)
            issues.extend(structure_issues)
            
            # Column-specific validations
            if 'wavenumber' in df.columns:
                wavenumber_issues = self._validate_wavenumber_column(df['wavenumber'])
                issues.extend(wavenumber_issues)
            
            # Intensity column validations
            intensity_cols = [col for col in df.columns if 'intensity' in col.lower() or 'absorbance' in col.lower()]
            for col in intensity_cols:
                intensity_issues = self._validate_intensity_column(df[col], col)
                issues.extend(intensity_issues)
            
            # Data quality validations
            quality_issues = self._validate_data_quality(df)
            issues.extend(quality_issues)
            
            # Spectral-specific validations
            spectral_issues = self._validate_spectral_requirements(df)
            issues.extend(spectral_issues)
            
            # Calculate statistics
            statistics = self._calculate_statistics(df)
            
            # Determine overall validity and quality score
            is_valid = not any(issue.level == ValidationLevel.ERROR for issue in issues)
            quality_score = self._calculate_quality_score(issues, statistics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues)
            
            result = ValidationResult(
                is_valid=is_valid,
                issues=issues,
                statistics=statistics,
                quality_score=quality_score,
                recommendations=recommendations
            )
            
            self.logger.info(f"Validation completed: {len(issues)} issues found, quality score: {quality_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    rule=ValidationRule.MISSING_VALUES,
                    level=ValidationLevel.ERROR,
                    message=f"Validation process failed: {e}"
                )],
                statistics={},
                quality_score=0.0,
                recommendations=["Fix data format issues and retry validation"]
            )
    
    def _validate_structure(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate basic data structure."""
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append(ValidationIssue(
                rule=ValidationRule.MISSING_VALUES,
                level=ValidationLevel.ERROR,
                message="Dataset is empty",
                suggested_fix="Ensure data file contains valid spectral data"
            ))
            return issues
        
        # Check for required columns
        required_cols = ['wavenumber']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(ValidationIssue(
                rule=ValidationRule.MISSING_VALUES,
                level=ValidationLevel.ERROR,
                message=f"Missing required columns: {missing_cols}",
                suggested_fix="Ensure data contains wavenumber column"
            ))
        
        # Check for intensity columns
        intensity_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['intensity', 'absorbance', 'transmittance'])]
        if not intensity_cols:
            issues.append(ValidationIssue(
                rule=ValidationRule.MISSING_VALUES,
                level=ValidationLevel.WARNING,
                message="No intensity/absorbance columns detected",
                suggested_fix="Verify column naming conventions"
            ))
        
        return issues
    
    def _validate_wavenumber_column(self, wavenumber_series: pd.Series) -> List[ValidationIssue]:
        """Validate wavenumber column data."""
        issues = []
        
        if not self.enabled_rules[ValidationRule.WAVENUMBER_RANGE]:
            return issues
        
        # Check for missing values
        missing_count = wavenumber_series.isnull().sum()
        if missing_count > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.MISSING_VALUES,
                level=ValidationLevel.ERROR,
                message=f"Wavenumber column has {missing_count} missing values",
                suggested_fix="Remove or interpolate missing wavenumber values"
            ))
        
        # Validate numeric data
        numeric_data = pd.to_numeric(wavenumber_series, errors='coerce')
        non_numeric_count = numeric_data.isnull().sum() - missing_count
        if non_numeric_count > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.WAVENUMBER_RANGE,
                level=ValidationLevel.ERROR,
                message=f"Wavenumber column has {non_numeric_count} non-numeric values",
                suggested_fix="Ensure all wavenumber values are numeric"
            ))
            return issues
        
        # Check wavenumber range
        min_wn, max_wn = numeric_data.min(), numeric_data.max()
        if min_wn < self.thresholds['wavenumber_min'] or max_wn > self.thresholds['wavenumber_max']:
            issues.append(ValidationIssue(
                rule=ValidationRule.WAVENUMBER_RANGE,
                level=ValidationLevel.WARNING,
                message=f"Wavenumber range ({min_wn:.1f}-{max_wn:.1f}) outside typical IR range",
                suggested_fix="Verify wavenumber units and instrument calibration"
            ))
        
        # Check wavenumber order (should be descending for IR)
        if self.enabled_rules[ValidationRule.WAVENUMBER_ORDER]:
            if not numeric_data.is_monotonic_decreasing:
                issues.append(ValidationIssue(
                    rule=ValidationRule.WAVENUMBER_ORDER,
                    level=ValidationLevel.WARNING,
                    message="Wavenumber values are not in descending order",
                    suggested_fix="Sort data by wavenumber in descending order"
                ))
        
        # Check for duplicates
        if self.enabled_rules[ValidationRule.DUPLICATE_VALUES]:
            duplicate_count = numeric_data.duplicated().sum()
            if duplicate_count > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.DUPLICATE_VALUES,
                    level=ValidationLevel.WARNING,
                    message=f"Found {duplicate_count} duplicate wavenumber values",
                    suggested_fix="Remove or average duplicate measurements"
                ))
        
        return issues
    
    def _validate_intensity_column(self, intensity_series: pd.Series, column_name: str) -> List[ValidationIssue]:
        """Validate intensity/absorbance column data."""
        issues = []
        
        # Determine data type from column name
        is_absorbance = 'absorbance' in column_name.lower() or 'abs' in column_name.lower()
        is_transmittance = 'transmittance' in column_name.lower() or 'trans' in column_name.lower()
        
        # Check for missing values
        missing_count = intensity_series.isnull().sum()
        missing_percent = (missing_count / len(intensity_series)) * 100
        
        if missing_percent > self.thresholds['max_missing_percent']:
            issues.append(ValidationIssue(
                rule=ValidationRule.MISSING_VALUES,
                level=ValidationLevel.ERROR,
                message=f"Column '{column_name}' has {missing_percent:.1f}% missing values",
                suggested_fix="Interpolate or remove rows with missing intensity data"
            ))
        elif missing_count > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.MISSING_VALUES,
                level=ValidationLevel.WARNING,
                message=f"Column '{column_name}' has {missing_count} missing values",
                suggested_fix="Consider interpolating missing values"
            ))
        
        # Validate numeric data
        numeric_data = pd.to_numeric(intensity_series, errors='coerce')
        non_numeric_count = numeric_data.isnull().sum() - missing_count
        if non_numeric_count > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.ABSORBANCE_RANGE if is_absorbance else ValidationRule.TRANSMITTANCE_RANGE,
                level=ValidationLevel.ERROR,
                message=f"Column '{column_name}' has {non_numeric_count} non-numeric values",
                suggested_fix="Ensure all intensity values are numeric"
            ))
            return issues
        
        # Range validation
        if is_absorbance and self.enabled_rules[ValidationRule.ABSORBANCE_RANGE]:
            min_val, max_val = numeric_data.min(), numeric_data.max()
            if min_val < self.thresholds['absorbance_min'] or max_val > self.thresholds['absorbance_max']:
                issues.append(ValidationIssue(
                    rule=ValidationRule.ABSORBANCE_RANGE,
                    level=ValidationLevel.WARNING,
                    message=f"Absorbance range ({min_val:.3f}-{max_val:.3f}) outside typical range",
                    suggested_fix="Verify instrument calibration and sample preparation"
                ))
        
        elif is_transmittance and self.enabled_rules[ValidationRule.TRANSMITTANCE_RANGE]:
            min_val, max_val = numeric_data.min(), numeric_data.max()
            if min_val < self.thresholds['transmittance_min'] or max_val > self.thresholds['transmittance_max']:
                issues.append(ValidationIssue(
                    rule=ValidationRule.TRANSMITTANCE_RANGE,
                    level=ValidationLevel.WARNING,
                    message=f"Transmittance range ({min_val:.1f}-{max_val:.1f}) outside 0-100% range",
                    suggested_fix="Check transmittance units and calculation"
                ))
        
        return issues
    
    def _validate_data_quality(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate overall data quality."""
        issues = []
        
        if not self.enabled_rules[ValidationRule.DATA_CONTINUITY]:
            return issues
        
        # Check data continuity (gaps in wavenumber)
        if 'wavenumber' in df.columns:
            wavenumber = pd.to_numeric(df['wavenumber'], errors='coerce').dropna()
            if len(wavenumber) > 1:
                # Calculate typical step size
                steps = np.diff(wavenumber.sort_values())
                median_step = np.median(np.abs(steps))
                
                # Find large gaps (more than 3x typical step)
                large_gaps = np.abs(steps) > 3 * median_step
                if large_gaps.any():
                    gap_count = large_gaps.sum()
                    issues.append(ValidationIssue(
                        rule=ValidationRule.DATA_CONTINUITY,
                        level=ValidationLevel.WARNING,
                        message=f"Found {gap_count} large gaps in wavenumber sequence",
                        suggested_fix="Check for missing data points or instrument issues"
                    ))
        
        # Advanced quality checks (if enabled)
        if self.enabled_rules[ValidationRule.NOISE_LEVEL]:
            noise_issues = self._check_noise_level(df)
            issues.extend(noise_issues)
        
        if self.enabled_rules[ValidationRule.BASELINE_DRIFT]:
            baseline_issues = self._check_baseline_drift(df)
            issues.extend(baseline_issues)
        
        return issues
    
    def _validate_spectral_requirements(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate spectroscopy-specific requirements."""
        issues = []
        
        # Check minimum data points
        if self.enabled_rules[ValidationRule.MINIMUM_DATA_POINTS]:
            if len(df) < self.thresholds['min_data_points']:
                issues.append(ValidationIssue(
                    rule=ValidationRule.MINIMUM_DATA_POINTS,
                    level=ValidationLevel.WARNING,
                    message=f"Dataset has only {len(df)} data points, minimum recommended is {self.thresholds['min_data_points']}",
                    suggested_fix="Collect more data points for meaningful spectral analysis"
                ))
        
        # Check for scale issues
        if self.enabled_rules[ValidationRule.SCALE_ISSUES]:
            scale_issues = self._check_scale_issues(df)
            issues.extend(scale_issues)
        
        # Check for outliers
        if self.enabled_rules[ValidationRule.OUTLIER_DETECTION]:
            outlier_issues = self._check_outliers(df)
            issues.extend(outlier_issues)
        
        # Check spectral resolution
        if self.enabled_rules[ValidationRule.SPECTRAL_RESOLUTION]:
            resolution_issues = self._check_spectral_resolution(df)
            issues.extend(resolution_issues)
        
        return issues
    
    def _check_scale_issues(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for potential scale issues in the data."""
        issues = []
        
        if 'wavenumber' not in df.columns:
            return issues
        
        wavenumber = pd.to_numeric(df['wavenumber'], errors='coerce').dropna()
        if len(wavenumber) == 0:
            return issues
        
        wn_range = wavenumber.max() - wavenumber.min()
        
        # Check if wavenumbers might be scaled incorrectly
        if wn_range < 100:  # Very small range, might be divided by 1000
            issues.append(ValidationIssue(
                rule=ValidationRule.SCALE_ISSUES,
                level=ValidationLevel.WARNING,
                message=f"Wavenumber range ({wn_range:.1f}) is very small, data might be scaled incorrectly",
                suggested_fix="Check if wavenumbers need to be multiplied by 1000"
            ))
        elif wn_range > 10000:  # Very large range, might be in Hz instead of cm⁻¹
            issues.append(ValidationIssue(
                rule=ValidationRule.SCALE_ISSUES,
                level=ValidationLevel.WARNING,
                message=f"Wavenumber range ({wn_range:.1f}) is very large, might be in wrong units",
                suggested_fix="Check if wavenumbers are in Hz instead of cm⁻¹"
            ))
        
        # Check intensity scales
        intensity_cols = [col for col in df.columns if any(keyword in col.lower()
                         for keyword in ['intensity', 'absorbance', 'transmittance'])]
        
        for col in intensity_cols:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 0:
                max_val = data.max()
                if max_val > self.thresholds['max_scale_factor']:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.SCALE_ISSUES,
                        level=ValidationLevel.WARNING,
                        message=f"Column '{col}' has very large values (max: {max_val:.2e}), might be scaled incorrectly",
                        suggested_fix="Check if intensity values need scaling"
                    ))
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for statistical outliers in the data."""
        issues = []
        
        intensity_cols = [col for col in df.columns if any(keyword in col.lower()
                         for keyword in ['intensity', 'absorbance', 'transmittance'])]
        
        for col in intensity_cols:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 10:
                # Use IQR method for outlier detection
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_percent = (outlier_count / len(data)) * 100
                    level = ValidationLevel.WARNING if outlier_percent > 5 else ValidationLevel.INFO
                    
                    issues.append(ValidationIssue(
                        rule=ValidationRule.OUTLIER_DETECTION,
                        level=level,
                        message=f"Column '{col}' has {outlier_count} outliers ({outlier_percent:.1f}%)",
                        suggested_fix="Review outlier values and consider data cleaning",
                        metadata={'outlier_values': outliers.tolist()[:10]}  # First 10 outliers
                    ))
        
        return issues
    
    def _check_spectral_resolution(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check spectral resolution and uniformity."""
        issues = []
        
        if 'wavenumber' not in df.columns:
            return issues
        
        wavenumber = pd.to_numeric(df['wavenumber'], errors='coerce').dropna().sort_values()
        if len(wavenumber) < 2:
            return issues
        
        # Calculate step sizes
        steps = np.diff(wavenumber)
        avg_step = np.mean(np.abs(steps))
        step_std = np.std(steps)
        
        # Check if resolution is too low
        if avg_step > self.thresholds['min_resolution']:
            issues.append(ValidationIssue(
                rule=ValidationRule.SPECTRAL_RESOLUTION,
                level=ValidationLevel.INFO,
                message=f"Low spectral resolution detected (avg step: {avg_step:.2f} cm⁻¹)",
                suggested_fix="Consider using higher resolution measurements"
            ))
        
        # Check for non-uniform spacing
        if step_std / avg_step > 0.1:  # More than 10% variation
            issues.append(ValidationIssue(
                rule=ValidationRule.SPECTRAL_RESOLUTION,
                level=ValidationLevel.WARNING,
                message=f"Non-uniform wavenumber spacing detected (std/mean: {step_std/avg_step:.3f})",
                suggested_fix="Check instrument calibration or data interpolation"
            ))
        
        return issues
    
    def _check_noise_level(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for excessive noise in spectral data."""
        issues = []
        
        intensity_cols = [col for col in df.columns if 'intensity' in col.lower() or 'absorbance' in col.lower()]
        
        for col in intensity_cols:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 10:
                # Calculate noise level using standard deviation of differences
                noise_level = np.std(np.diff(data)) / np.mean(np.abs(data))
                
                if noise_level > self.thresholds['max_noise_level']:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.NOISE_LEVEL,
                        level=ValidationLevel.WARNING,
                        message=f"High noise level detected in '{col}' (noise ratio: {noise_level:.3f})",
                        suggested_fix="Consider applying smoothing or noise reduction"
                    ))
        
        return issues
    
    def _check_baseline_drift(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for baseline drift in spectral data."""
        issues = []
        
        intensity_cols = [col for col in df.columns if 'intensity' in col.lower() or 'absorbance' in col.lower()]
        
        for col in intensity_cols:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(data) > 50:
                # Simple baseline drift detection using linear trend
                x = np.arange(len(data))
                slope = np.polyfit(x, data, 1)[0]
                
                # Normalize slope by data range
                data_range = data.max() - data.min()
                if data_range > 0:
                    normalized_slope = abs(slope) / data_range
                    
                    if normalized_slope > self.thresholds['max_baseline_drift']:
                        issues.append(ValidationIssue(
                            rule=ValidationRule.BASELINE_DRIFT,
                            level=ValidationLevel.INFO,
                            message=f"Baseline drift detected in '{col}' (slope: {slope:.6f})",
                            suggested_fix="Consider baseline correction"
                        ))
        
        return issues
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data statistics."""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # Wavenumber statistics
        if 'wavenumber' in df.columns:
            wn_data = pd.to_numeric(df['wavenumber'], errors='coerce')
            stats['wavenumber_stats'] = {
                'min': float(wn_data.min()) if not wn_data.empty else None,
                'max': float(wn_data.max()) if not wn_data.empty else None,
                'count': int(wn_data.count()),
                'duplicates': int(wn_data.duplicated().sum())
            }
        
        # Intensity statistics
        intensity_cols = [col for col in df.columns if any(keyword in col.lower() 
                         for keyword in ['intensity', 'absorbance', 'transmittance'])]
        
        stats['intensity_stats'] = {}
        for col in intensity_cols:
            data = pd.to_numeric(df[col], errors='coerce')
            stats['intensity_stats'][col] = {
                'min': float(data.min()) if not data.empty else None,
                'max': float(data.max()) if not data.empty else None,
                'mean': float(data.mean()) if not data.empty else None,
                'std': float(data.std()) if not data.empty else None,
                'count': int(data.count())
            }
        
        return stats
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], statistics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        base_score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.level == ValidationLevel.ERROR:
                base_score -= 20
            elif issue.level == ValidationLevel.WARNING:
                base_score -= 10
            elif issue.level == ValidationLevel.INFO:
                base_score -= 2
        
        # Bonus for completeness
        if statistics.get('total_rows', 0) > 100:
            base_score += 5
        
        # Penalty for missing data
        total_missing = sum(statistics.get('missing_values', {}).values())
        total_cells = statistics.get('total_rows', 1) * statistics.get('total_columns', 1)
        missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        base_score -= missing_percent * 2
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on validation issues."""
        recommendations = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.rule not in issue_types:
                issue_types[issue.rule] = []
            issue_types[issue.rule].append(issue)
        
        # Generate specific recommendations
        if ValidationRule.MISSING_VALUES in issue_types:
            recommendations.append("Address missing values through interpolation or data cleaning")
        
        if ValidationRule.WAVENUMBER_RANGE in issue_types:
            recommendations.append("Verify wavenumber calibration and units")
        
        if ValidationRule.WAVENUMBER_ORDER in issue_types:
            recommendations.append("Sort data by wavenumber in descending order")
        
        if ValidationRule.DUPLICATE_VALUES in issue_types:
            recommendations.append("Remove or average duplicate measurements")
        
        if ValidationRule.DATA_CONTINUITY in issue_types:
            recommendations.append("Check for missing data points in spectral sequence")
        
        # General recommendations
        error_count = sum(1 for issue in issues if issue.level == ValidationLevel.ERROR)
        if error_count > 0:
            recommendations.append("Resolve all error-level issues before proceeding with analysis")
        
        warning_count = sum(1 for issue in issues if issue.level == ValidationLevel.WARNING)
        if warning_count > 3:
            recommendations.append("Review data collection and preprocessing procedures")
        
        return recommendations
    
    def validate_spectral_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate spectroscopy data format requirements.
        
        Args:
            df: DataFrame containing spectral data
            
        Returns:
            ValidationResult with spectroscopy-specific validation
        """
        return self.validate_data(df, data_type="spectral")
    
    def check_wavenumber_range(self, wavenumbers: pd.Series) -> ValidationIssue:
        """
        Validate wavenumber values and range.
        
        Args:
            wavenumbers: Series containing wavenumber data
            
        Returns:
            ValidationIssue if problems found, None otherwise
        """
        numeric_data = pd.to_numeric(wavenumbers, errors='coerce')
        
        if numeric_data.isnull().all():
            return ValidationIssue(
                rule=ValidationRule.WAVENUMBER_RANGE,
                level=ValidationLevel.ERROR,
                message="No valid numeric wavenumber data found",
                suggested_fix="Ensure wavenumber column contains numeric values"
            )
        
        min_wn, max_wn = numeric_data.min(), numeric_data.max()
        
        if min_wn < self.thresholds['wavenumber_min'] or max_wn > self.thresholds['wavenumber_max']:
            return ValidationIssue(
                rule=ValidationRule.WAVENUMBER_RANGE,
                level=ValidationLevel.WARNING,
                message=f"Wavenumber range ({min_wn:.1f}-{max_wn:.1f}) outside typical IR range",
                suggested_fix="Verify wavenumber units and instrument calibration"
            )
        
        return None
    
    def check_absorbance_values(self, absorbance: pd.Series) -> ValidationIssue:
        """
        Validate absorbance data quality.
        
        Args:
            absorbance: Series containing absorbance data
            
        Returns:
            ValidationIssue if problems found, None otherwise
        """
        numeric_data = pd.to_numeric(absorbance, errors='coerce')
        
        if numeric_data.isnull().all():
            return ValidationIssue(
                rule=ValidationRule.ABSORBANCE_RANGE,
                level=ValidationLevel.ERROR,
                message="No valid numeric absorbance data found",
                suggested_fix="Ensure absorbance column contains numeric values"
            )
        
        min_abs, max_abs = numeric_data.min(), numeric_data.max()
        
        if min_abs < self.thresholds['absorbance_min'] or max_abs > self.thresholds['absorbance_max']:
            return ValidationIssue(
                rule=ValidationRule.ABSORBANCE_RANGE,
                level=ValidationLevel.WARNING,
                message=f"Absorbance range ({min_abs:.3f}-{max_abs:.3f}) outside typical range",
                suggested_fix="Verify instrument calibration and sample preparation"
            )
        
        return None
    
    def validate_baseline_overlap(self, baseline_df: pd.DataFrame, sample_df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Check for overlapping ranges between baseline and sample data.
        
        Args:
            baseline_df: DataFrame containing baseline data
            sample_df: DataFrame containing sample data
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if 'wavenumber' not in baseline_df.columns or 'wavenumber' not in sample_df.columns:
            issues.append(ValidationIssue(
                rule=ValidationRule.OVERLAPPING_RANGES,
                level=ValidationLevel.ERROR,
                message="Missing wavenumber columns in baseline or sample data",
                suggested_fix="Ensure both datasets have wavenumber columns"
            ))
            return issues
        
        baseline_wn = pd.to_numeric(baseline_df['wavenumber'], errors='coerce').dropna()
        sample_wn = pd.to_numeric(sample_df['wavenumber'], errors='coerce').dropna()
        
        if len(baseline_wn) == 0 or len(sample_wn) == 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.OVERLAPPING_RANGES,
                level=ValidationLevel.ERROR,
                message="No valid wavenumber data in baseline or sample",
                suggested_fix="Check data quality and format"
            ))
            return issues
        
        baseline_range = (baseline_wn.min(), baseline_wn.max())
        sample_range = (sample_wn.min(), sample_wn.max())
        
        # Check for overlap
        overlap_start = max(baseline_range[0], sample_range[0])
        overlap_end = min(baseline_range[1], sample_range[1])
        
        if overlap_start >= overlap_end:
            issues.append(ValidationIssue(
                rule=ValidationRule.OVERLAPPING_RANGES,
                level=ValidationLevel.ERROR,
                message=f"No overlap between baseline ({baseline_range[0]:.1f}-{baseline_range[1]:.1f}) and sample ({sample_range[0]:.1f}-{sample_range[1]:.1f}) ranges",
                suggested_fix="Ensure baseline and sample measurements cover overlapping wavenumber ranges"
            ))
        else:
            overlap_size = overlap_end - overlap_start
            baseline_size = baseline_range[1] - baseline_range[0]
            sample_size = sample_range[1] - sample_range[0]
            
            min_coverage = min(overlap_size / baseline_size, overlap_size / sample_size)
            
            if min_coverage < 0.8:  # Less than 80% overlap
                issues.append(ValidationIssue(
                    rule=ValidationRule.OVERLAPPING_RANGES,
                    level=ValidationLevel.WARNING,
                    message=f"Limited overlap ({min_coverage*100:.1f}%) between baseline and sample ranges",
                    suggested_fix="Increase measurement range overlap for better baseline correction"
                ))
        
        return issues
    
    def set_threshold(self, parameter: str, value: float):
        """Set validation threshold parameter."""
        if parameter in self.thresholds:
            self.thresholds[parameter] = value
            self.logger.info(f"Updated threshold {parameter} to {value}")
        else:
            self.logger.warning(f"Unknown threshold parameter: {parameter}")
    
    def enable_rule(self, rule: ValidationRule, enabled: bool = True):
        """Enable or disable a validation rule."""
        self.enabled_rules[rule] = enabled
        self.logger.info(f"{'Enabled' if enabled else 'Disabled'} validation rule: {rule.value}")
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Generate a human-readable validation summary."""
        summary_lines = [
            f"Validation Summary:",
            f"  Status: {'PASSED' if result.is_valid else 'FAILED'}",
            f"  Quality Score: {result.quality_score:.1f}/100",
            f"  Issues Found: {len(result.issues)}"
        ]
        
        if result.issues:
            error_count = sum(1 for issue in result.issues if issue.level == ValidationLevel.ERROR)
            warning_count = sum(1 for issue in result.issues if issue.level == ValidationLevel.WARNING)
            info_count = sum(1 for issue in result.issues if issue.level == ValidationLevel.INFO)
            
            summary_lines.extend([
                f"    Errors: {error_count}",
                f"    Warnings: {warning_count}",
                f"    Info: {info_count}"
            ])
        
        if result.recommendations:
            summary_lines.append("  Recommendations:")
            for rec in result.recommendations[:3]:  # Show top 3
                summary_lines.append(f"    - {rec}")
        
        return "\n".join(summary_lines)