"""
AI-powered CSV normalization engine for spectroscopy data.

Uses OpenRouter API with advanced AI models to intelligently normalize
and map CSV columns to standard spectroscopy data formats.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime

from config.api_config import APIConfig
from utils.api_client import APIClient, OpenRouterClient
from utils.security import SecureKeyManager
from utils.cost_tracker import CostTracker
# Import CacheManager lazily to avoid circular import


class ConfidenceLevel(Enum):
    """AI confidence levels for normalization decisions."""
    HIGH = "high"      # >90% - Auto-apply
    MEDIUM = "medium"  # 70-90% - Show preview
    LOW = "low"        # <70% - Manual mapping required


@dataclass
class ColumnMapping:
    """Mapping between original and target columns."""
    original_name: str
    target_name: str
    data_type: str
    transformation: Optional[str] = None
    confidence: float = 0.0
    notes: Optional[str] = None


@dataclass
class TransformationStep:
    """Individual transformation step."""
    type: str
    parameters: Dict[str, Any]
    reason: str
    confidence: float = 1.0


@dataclass
class AIAnalysis:
    """AI response with analysis results."""
    can_normalize: bool
    confidence: float
    detected_format: Dict[str, Any]
    column_mapping: Dict[str, Optional[str]]
    transformations: List[TransformationStep]
    warnings: List[str]
    recommendations: List[str]
    analysis_notes: str


@dataclass
class NormalizationResult:
    """Complete normalization results."""
    success: bool
    normalized_data: Optional[pd.DataFrame] = None
    plan: Optional['NormalizationPlan'] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    processing_time: float = 0.0
    cache_hit: bool = False


@dataclass
class NormalizationPlan:
    """Complete normalization plan from AI analysis."""
    file_hash: str
    column_mappings: List[ColumnMapping]
    data_transformations: List[str]
    confidence_score: float
    confidence_level: ConfidenceLevel
    issues_detected: List[str]
    metadata: Dict[str, Any]
    ai_model: str
    timestamp: str
    transformation_steps: List[TransformationStep] = None


@dataclass
class NormalizationRequest:
    """Request structure for AI normalization."""
    csv_preview: str
    file_info: Dict[str, Any]
    expected_format: Dict[str, Any]
    context: Optional[str] = None


@dataclass
class UsageStats:
    """API usage tracking and cost monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None


class AINormalizer:
    """
    AI-powered normalization engine for spectroscopy CSV data.
    
    Features:
    - Intelligent column mapping using AI
    - Confidence-based decision making
    - Caching for cost optimization
    - Fallback strategies for reliability
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the AI normalizer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize components
        self.api_config = APIConfig()
        self.api_client = APIClient(config_manager)
        self.cache_manager = None  # Initialize lazily
        self.cost_tracker = CostTracker(config_manager=config_manager)
        
        # Initialize OpenRouter client
        self.openrouter_client = None
        self.security_manager = SecureKeyManager()
        self._init_openrouter_client()
        
        # Usage statistics
        self.usage_stats = UsageStats()
        
        # Expected spectroscopy format
        self.expected_format = {
            "columns": {
                "wavenumber": {
                    "description": "Wavenumber in cm⁻¹",
                    "range": [400, 4000],
                    "order": "descending",
                    "required": True
                },
                "absorbance": {
                    "description": "Absorbance values",
                    "range": [0.0, 5.0],
                    "required": False
                },
                "transmittance": {
                    "description": "Transmittance percentage",
                    "range": [0.0, 100.0],
                    "required": False
                },
                "intensity": {
                    "description": "Signal intensity",
                    "range": [0.0, None],
                    "required": False
                }
            },
            "format": "Two-column format with wavenumber and intensity/absorbance"
        }
    
    def _init_openrouter_client(self):
        """Initialize OpenRouter client with API key."""
        try:
            if self.config_manager:
                api_key = self.config_manager.get_api_key("openrouter")
                if api_key:
                    self.openrouter_client = OpenRouterClient(api_key, self.config_manager)
                    self.logger.info("OpenRouter client initialized")
                else:
                    self.logger.warning("No OpenRouter API key found")
            else:
                self.logger.warning("No config manager provided for OpenRouter client")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenRouter client: {e}")
    
    async def normalize_csv(self, csv_data: pd.DataFrame, file_path: str,
                           force_refresh: bool = False, use_ai: bool = True) -> NormalizationResult:
        """
        Normalize CSV data using AI analysis with comprehensive result tracking.
        
        Args:
            csv_data: DataFrame containing CSV data
            file_path: Path to the original CSV file
            force_refresh: Skip cache and force new AI analysis
            use_ai: If False, only use cached data and never call AI (for cost savings)
            
        Returns:
            NormalizationResult with complete processing information
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting AI normalization for {file_path}")
            
            # Generate file hash for caching using enhanced method
            if self.cache_manager is None:
                from utils.cache_manager import CacheManager
                self.cache_manager = CacheManager()
            file_hash = self.cache_manager.generate_file_structure_hash(csv_data, file_path)
            
            # Check cache first (unless forced refresh)
            cached_plan = None
            if not force_refresh:
                if self.cache_manager is None:
                    from utils.cache_manager import CacheManager
                    self.cache_manager = CacheManager()
                cached_plan = await self.cache_manager.get_normalization_plan(file_hash)
                if cached_plan:
                    self.logger.info("Using cached normalization plan")
                    self.usage_stats.cache_hits += 1
                    if self.openrouter_client:
                        self.openrouter_client.track_cache_hit()
                    
                    # Track cache hit for cost savings
                    avg_cost = self.cost_tracker.get_usage_statistics().average_cost_per_call
                    if avg_cost > 0:
                        self.cost_tracker.track_api_call(
                            model=cached_plan.ai_model,
                            provider="openrouter",
                            tokens_used=0,
                            cost=0.0,
                            response_time=0.0,
                            success=True,
                            cache_hit=True,
                            operation_type="normalization",
                            file_hash=file_hash
                        )
                    
                    # Apply cached plan
                    normalized_data = await self.apply_normalization_plan(csv_data, cached_plan)
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    return NormalizationResult(
                        success=True,
                        normalized_data=normalized_data,
                        plan=cached_plan,
                        processing_time=processing_time,
                        cache_hit=True
                    )
            
            # Check if AI is allowed
            if not use_ai:
                # Cache-only mode: fail if no cache available
                self.logger.warning(f"Cache miss and use_ai=False - cannot normalize without AI: {file_path}")
                processing_time = (datetime.now() - start_time).total_seconds()
                return NormalizationResult(
                    success=False,
                    error_message="No cached normalization plan available and AI is disabled (use_ai=False)",
                    processing_time=processing_time,
                    cache_hit=False
                )
            
            # Get AI normalization plan (only when use_ai=True)
            plan = await self._get_ai_normalization_plan(csv_data, file_path, file_hash)
            
            # Apply normalization plan
            normalized_data = await self.apply_normalization_plan(csv_data, plan)
            
            # Cache the plan
            if self.cache_manager is None:
                from utils.cache_manager import CacheManager
                self.cache_manager = CacheManager()
            await self.cache_manager.store_normalization_plan(file_hash, plan)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"AI normalization completed with {plan.confidence_level.value} confidence")
            
            return NormalizationResult(
                success=True,
                normalized_data=normalized_data,
                plan=plan,
                processing_time=processing_time,
                cache_hit=False
            )
            
        except Exception as e:
            self.logger.error(f"AI normalization failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Try fallback plan
            try:
                fallback_plan = self._create_fallback_plan(csv_data, file_path, str(e))
                normalized_data = await self.apply_normalization_plan(csv_data, fallback_plan)
                
                return NormalizationResult(
                    success=True,
                    normalized_data=normalized_data,
                    plan=fallback_plan,
                    error_message=f"AI failed, used fallback: {str(e)}",
                    warnings=["Used heuristic-based normalization due to AI failure"],
                    processing_time=processing_time,
                    cache_hit=False
                )
            except Exception as fallback_error:
                return NormalizationResult(
                    success=False,
                    error_message=f"Both AI and fallback normalization failed: {str(e)}, {str(fallback_error)}",
                    processing_time=processing_time,
                    cache_hit=False
                )
    
    async def normalize_csv_structure(self, csv_data: pd.DataFrame, structure) -> NormalizationResult:
        """
        Normalize CSV structure - compatibility method for integration tests.
        
        Args:
            csv_data: DataFrame containing CSV data
            structure: CSV structure information (ignored for now)
            
        Returns:
            NormalizationResult with normalization information
        """
        # Use the main normalize_csv method with a dummy file path
        return await self.normalize_csv(csv_data, "test_file.csv")
    
    async def _get_ai_normalization_plan(self, csv_data: pd.DataFrame, file_path: str, file_hash: str) -> NormalizationPlan:
        """Get AI normalization plan using OpenRouter client."""
        try:
            # Prepare CSV preview for AI analysis
            csv_preview = self._prepare_csv_preview(csv_data)
            
            # Create file info
            file_info = {
                "path": file_path,
                "rows": len(csv_data),
                "columns": len(csv_data.columns),
                "column_names": list(csv_data.columns),
                "dtypes": csv_data.dtypes.astype(str).to_dict(),
                "sample_values": {
                    col: csv_data[col].head(3).tolist() if not csv_data[col].empty else []
                    for col in csv_data.columns
                }
            }
            
            # Use OpenRouter client if available, otherwise fallback to general API client
            if self.openrouter_client:
                ai_response = await self.openrouter_client.analyze_csv_structure(csv_preview, file_info)
                
                # Track cost and usage
                response_time = 0.5  # Placeholder - would get from actual response
                tokens_used = 1500  # Placeholder - would get from actual response
                cost = self.api_config.estimate_cost(self.api_config.current_model, tokens_used)
                
                self.cost_tracker.track_api_call(
                    model=self.api_config.current_model,
                    provider="openrouter",
                    tokens_used=tokens_used,
                    cost=cost,
                    response_time=response_time,
                    success=True,
                    cache_hit=False,
                    operation_type="normalization",
                    file_hash=file_hash
                )
                
                # Update usage stats
                self.usage_stats.total_requests += 1
                self.usage_stats.successful_requests += 1
                self.usage_stats.last_request_time = datetime.now()
                
            else:
                # Fallback to general API client
                request = NormalizationRequest(
                    csv_preview=csv_preview,
                    file_info=file_info,
                    expected_format=self.expected_format,
                    context="Spectroscopy data normalization for MRG Labs"
                )
                ai_response = await self._request_ai_normalization(request)
            
            # Process AI response into normalization plan
            plan = self._process_ai_response(ai_response, file_hash)
            
            return plan
            
        except Exception as e:
            self.usage_stats.failed_requests += 1
            self.logger.error(f"Failed to get AI normalization plan: {e}")
            raise
    
    async def normalize_csv_legacy(self, csv_data: pd.DataFrame, file_path: str,
                           force_refresh: bool = False) -> NormalizationPlan:
        """
        Normalize CSV data using AI analysis.
        
        Args:
            csv_data: DataFrame containing CSV data
            file_path: Path to the original CSV file
            force_refresh: Skip cache and force new AI analysis
            
        Returns:
            NormalizationPlan with mapping and transformation instructions
        """
        try:
            self.logger.info(f"Starting AI normalization for {file_path}")
            
            # Generate file hash for caching
            file_hash = self._generate_file_hash(csv_data, file_path)
            
            # Check cache first (unless forced refresh)
            if not force_refresh:
                if self.cache_manager is None:
                    from utils.cache_manager import CacheManager
                    self.cache_manager = CacheManager()
                cached_plan = await self.cache_manager.get_normalization_plan(file_hash)
                if cached_plan:
                    self.logger.info("Using cached normalization plan")
                    return cached_plan
            
            # Prepare CSV preview for AI analysis
            csv_preview = self._prepare_csv_preview(csv_data)
            
            # Create normalization request
            request = NormalizationRequest(
                csv_preview=csv_preview,
                file_info={
                    "path": file_path,
                    "rows": len(csv_data),
                    "columns": len(csv_data.columns),
                    "column_names": list(csv_data.columns)
                },
                expected_format=self.expected_format,
                context="Spectroscopy data normalization for MRG Labs"
            )
            
            # Get AI normalization plan
            ai_response = await self._request_ai_normalization(request)
            
            # Process AI response into normalization plan
            plan = self._process_ai_response(ai_response, file_hash)
            
            # Cache the plan
            if self.cache_manager is None:
                from utils.cache_manager import CacheManager
                self.cache_manager = CacheManager()
            await self.cache_manager.store_normalization_plan(file_hash, plan)
            
            self.logger.info(f"AI normalization completed with {plan.confidence_level.value} confidence")
            return plan
            
        except Exception as e:
            self.logger.error(f"AI normalization failed: {e}")
            return self._create_fallback_plan(csv_data, file_path, str(e))
    
    async def apply_normalization_plan(self, csv_data: pd.DataFrame, 
                                     plan: NormalizationPlan) -> pd.DataFrame:
        """
        Apply normalization plan to CSV data.
        
        Args:
            csv_data: Original CSV data
            plan: Normalization plan to apply
            
        Returns:
            Normalized DataFrame
        """
        try:
            self.logger.info("Applying normalization plan")
            
            normalized_df = csv_data.copy()
            
            # Apply column mappings
            column_renames = {}
            for mapping in plan.column_mappings:
                if mapping.original_name in normalized_df.columns:
                    column_renames[mapping.original_name] = mapping.target_name
            
            if column_renames:
                normalized_df = normalized_df.rename(columns=column_renames)
            
            # Apply data transformations
            for transformation in plan.data_transformations:
                normalized_df = self._apply_transformation(normalized_df, transformation)
            
            # Validate normalized data
            validation_result = self._validate_normalized_data(normalized_df)
            if not validation_result['is_valid']:
                self.logger.warning(f"Normalized data validation issues: {validation_result['issues']}")
            
            self.logger.info("Normalization plan applied successfully")
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Failed to apply normalization plan: {e}")
            raise
    
    def _generate_file_hash(self, csv_data: pd.DataFrame, file_path: str) -> str:
        """Generate hash for CSV structure (for caching) - deprecated, use cache_manager method."""
        self.logger.warning("Using deprecated _generate_file_hash method. Use cache_manager.generate_file_structure_hash instead.")
        try:
            # Create structure signature
            structure_info = {
                'columns': list(csv_data.columns),
                'dtypes': csv_data.dtypes.astype(str).to_dict(),
                'shape': csv_data.shape,
                'sample_data': csv_data.head(5).to_dict() if not csv_data.empty else {}
            }
            
            # Generate hash
            structure_str = json.dumps(structure_info, sort_keys=True)
            return hashlib.sha256(structure_str.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate file hash: {e}")
            return hashlib.sha256(file_path.encode()).hexdigest()
    
    def _prepare_csv_preview(self, csv_data: pd.DataFrame, max_rows: int = 50) -> str:
        """Prepare CSV preview for AI analysis."""
        try:
            # Get preview data
            preview_df = csv_data.head(max_rows)
            
            # Convert to CSV string
            csv_preview = preview_df.to_csv(index=False)
            
            # Limit size to avoid token limits
            max_chars = 8000
            if len(csv_preview) > max_chars:
                # Truncate and add indicator
                csv_preview = csv_preview[:max_chars] + "\n... (truncated)"
            
            return csv_preview
            
        except Exception as e:
            self.logger.error(f"Failed to prepare CSV preview: {e}")
            return "Error preparing CSV preview"
    
    def _create_normalization_prompt(self) -> str:
        """Create the AI prompt template for normalization with grease analysis context."""
        return """
You are normalizing FTIR spectroscopy data for grease condition monitoring in machinery health assessment.

**Context:**
This CSV file contains spectral data from grease samples used in industrial machinery. The data comes from various laboratory instruments and may have inconsistent formatting.

**Expected Standard Format:**
- Two columns: Wavenumber (cm⁻¹) and Absorbance
- Wavenumber range: typically 400-4000 cm⁻¹ for FTIR, descending order
- Absorbance range: typically 0.0-5.0
- Numeric data only (except header row)

**CSV Preview:**
{csv_preview}

**File Information:**
{file_info}

**Your Task:**
Analyze this CSV structure and determine how to transform it into the standard format. Consider:
- Column identification (which is wavenumber, which is absorbance/transmittance)
- Metadata rows to skip
- Delimiter and decimal separator formats
- Data ordering (ascending vs descending wavenumbers)
- Unit conversions needed (e.g., transmittance % → absorbance)

Provide a structured JSON response with your normalization plan:

{{
    "column_mappings": [
        {{
            "original_name": "original_column_name",
            "target_name": "wavenumber|absorbance|transmittance|intensity|metadata",
            "data_type": "numeric|text|categorical",
            "transformation": "none|unit_conversion|scale_factor|other",
            "confidence": 0.0-1.0,
            "notes": "explanation of mapping decision"
        }}
    ],
    "data_transformations": [
        "List of required data transformations (e.g., 'sort_by_wavenumber_desc', 'convert_transmittance_to_absorbance')"
    ],
    "confidence_score": 0-100,
    "issues_detected": [
        "List of potential issues or concerns with the data"
    ],
    "analysis_notes": "Overall analysis and recommendations for grease condition monitoring"
}}

Focus on:
1. Identifying wavenumber/frequency columns (look for cm⁻¹, wavenumber, frequency patterns)
2. Identifying intensity/absorbance columns (look for absorbance, transmittance, intensity patterns)
3. Detecting data quality issues relevant to grease analysis
4. Recommending necessary transformations for machinery health assessment
5. Providing confidence scores for each mapping decision

Be conservative with confidence scores - only use high confidence (>0.9) when very certain.
"""
    
    async def _request_ai_normalization(self, request: NormalizationRequest) -> Dict[str, Any]:
        """Send normalization request to AI service."""
        try:
            # Format prompt with request data
            prompt = self._create_normalization_prompt().format(
                csv_preview=request.csv_preview,
                file_info=json.dumps(request.file_info, indent=2)
            )
            
            # Get recommended model for normalization task
            model_name = self.api_config.get_recommended_model("normalization")
            
            # Prepare API request
            api_request = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert spectroscopy data analyst. Provide accurate, well-reasoned normalization plans in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Low temperature for consistent results
                "max_tokens": 2000,
                "response_format": {"type": "json_object"}
            }
            
            # Send request with retry logic
            response = await self.api_client.chat_completion(api_request)
            
            # Parse JSON response
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                return json.loads(content)
            else:
                raise Exception("Invalid API response format")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response JSON: {e}")
            raise Exception("AI response was not valid JSON")
        except Exception as e:
            self.logger.error(f"AI normalization request failed: {e}")
            raise
    
    def _process_ai_response(self, ai_response: Dict[str, Any], file_hash: str) -> NormalizationPlan:
        """Process AI response into normalization plan."""
        try:
            # Extract column mappings
            mappings = []
            for mapping_data in ai_response.get('column_mappings', []):
                mapping = ColumnMapping(
                    original_name=mapping_data.get('original_name', ''),
                    target_name=mapping_data.get('target_name', ''),
                    data_type=mapping_data.get('data_type', 'unknown'),
                    transformation=mapping_data.get('transformation'),
                    confidence=float(mapping_data.get('confidence', 0.0)),
                    notes=mapping_data.get('notes')
                )
                mappings.append(mapping)
            
            # Determine confidence level
            confidence_score = float(ai_response.get('confidence_score', 0))
            if confidence_score >= 90:
                confidence_level = ConfidenceLevel.HIGH
            elif confidence_score >= 70:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW
            
            # Create normalization plan
            plan = NormalizationPlan(
                file_hash=file_hash,
                column_mappings=mappings,
                data_transformations=ai_response.get('data_transformations', []),
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                issues_detected=ai_response.get('issues_detected', []),
                metadata={
                    'analysis_notes': ai_response.get('analysis_notes', ''),
                    'model_used': self.api_config.current_model
                },
                ai_model=self.api_config.current_model,
                timestamp=pd.Timestamp.now().isoformat()
            )
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to process AI response: {e}")
            raise
    
    def _apply_transformation(self, df: pd.DataFrame, transformation: str) -> pd.DataFrame:
        """Apply a specific data transformation with enhanced capabilities."""
        try:
            if transformation == "sort_by_wavenumber_desc":
                if 'wavenumber' in df.columns:
                    df = df.sort_values('wavenumber', ascending=False)
            
            elif transformation == "sort_by_wavenumber_asc":
                if 'wavenumber' in df.columns:
                    df = df.sort_values('wavenumber', ascending=True)
            
            elif transformation == "convert_transmittance_to_absorbance":
                if 'transmittance' in df.columns:
                    # A = -log10(T/100), handle edge cases
                    transmittance = df['transmittance'].copy()
                    transmittance = np.clip(transmittance, 0.001, 100.0)  # Avoid log(0)
                    df['absorbance'] = -np.log10(transmittance / 100.0)
                    df = df.drop('transmittance', axis=1)
            
            elif transformation == "convert_absorbance_to_transmittance":
                if 'absorbance' in df.columns:
                    # T = 10^(-A) * 100
                    df['transmittance'] = (10 ** (-df['absorbance'])) * 100
                    df = df.drop('absorbance', axis=1)
            
            elif transformation == "remove_duplicate_wavenumbers":
                if 'wavenumber' in df.columns:
                    df = df.drop_duplicates(subset=['wavenumber'], keep='first')
            
            elif transformation == "interpolate_missing_values":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
            
            elif transformation == "remove_negative_values":
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if col != 'wavenumber':  # Don't modify wavenumber
                        df[col] = df[col].clip(lower=0)
            
            elif transformation == "normalize_intensity":
                intensity_cols = [col for col in df.columns if col in ['absorbance', 'transmittance', 'intensity']]
                for col in intensity_cols:
                    if col in df.columns:
                        max_val = df[col].max()
                        if max_val > 0:
                            df[col] = df[col] / max_val
            
            elif transformation.startswith("scale_factor_"):
                # Extract scale factor and column
                parts = transformation.split('_')
                if len(parts) >= 4:
                    try:
                        factor = float(parts[2])
                        column = '_'.join(parts[3:])
                        if column in df.columns:
                            df[column] = df[column] * factor
                    except ValueError:
                        self.logger.warning(f"Invalid scale factor in transformation: {transformation}")
            
            elif transformation.startswith("skip_rows_"):
                # Extract number of rows to skip from top
                parts = transformation.split('_')
                if len(parts) >= 3:
                    try:
                        skip_count = int(parts[2])
                        if skip_count > 0 and skip_count < len(df):
                            df = df.iloc[skip_count:].reset_index(drop=True)
                    except ValueError:
                        self.logger.warning(f"Invalid skip count in transformation: {transformation}")
            
            elif transformation.startswith("remove_columns_"):
                # Remove specified columns
                columns_to_remove = transformation.replace("remove_columns_", "").split(",")
                columns_to_remove = [col.strip() for col in columns_to_remove if col.strip() in df.columns]
                if columns_to_remove:
                    df = df.drop(columns=columns_to_remove)
            
            elif transformation == "reverse_order":
                df = df.iloc[::-1].reset_index(drop=True)
            
            elif transformation == "remove_outliers":
                # Remove outliers using IQR method for intensity columns
                intensity_cols = [col for col in df.columns if col in ['absorbance', 'transmittance', 'intensity']]
                for col in intensity_cols:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            else:
                self.logger.warning(f"Unknown transformation: {transformation}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to apply transformation '{transformation}': {e}")
            return df
    
    def apply_transformation_steps(self, df: pd.DataFrame, steps: List[TransformationStep]) -> pd.DataFrame:
        """Apply a list of transformation steps with detailed tracking."""
        try:
            result_df = df.copy()
            
            for step in steps:
                self.logger.debug(f"Applying transformation: {step.type} - {step.reason}")
                
                if step.type == "skip_rows":
                    count = step.parameters.get('count', 0)
                    if count > 0 and count < len(result_df):
                        result_df = result_df.iloc[count:].reset_index(drop=True)
                
                elif step.type == "rename_columns":
                    mapping = step.parameters.get('mapping', {})
                    result_df = result_df.rename(columns=mapping)
                
                elif step.type == "reverse_order":
                    result_df = result_df.iloc[::-1].reset_index(drop=True)
                
                elif step.type == "convert_units":
                    column = step.parameters.get('column')
                    factor = step.parameters.get('factor', 1.0)
                    if column and column in result_df.columns:
                        result_df[column] = result_df[column] * factor
                
                elif step.type == "scale_values":
                    column = step.parameters.get('column')
                    factor = step.parameters.get('factor', 1.0)
                    if column and column in result_df.columns:
                        result_df[column] = result_df[column] * factor
                
                elif step.type == "interpolate_missing":
                    method = step.parameters.get('method', 'linear')
                    numeric_columns = result_df.select_dtypes(include=[np.number]).columns
                    result_df[numeric_columns] = result_df[numeric_columns].interpolate(method=method)
                
                else:
                    # Try to apply as string transformation
                    result_df = self._apply_transformation(result_df, step.type)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to apply transformation steps: {e}")
            return df
    
    async def test_ai_connection(self) -> Dict[str, Any]:
        """Test AI service connection and return status."""
        try:
            if self.openrouter_client:
                response = await self.openrouter_client.test_connection()
                return {
                    'openrouter_available': response.success,
                    'openrouter_message': response.data.get('message', response.error) if response.data else response.error,
                    'response_time': response.response_time
                }
            else:
                return {
                    'openrouter_available': False,
                    'openrouter_message': 'OpenRouter client not initialized',
                    'response_time': None
                }
        except Exception as e:
            return {
                'openrouter_available': False,
                'openrouter_message': f'Connection test failed: {str(e)}',
                'response_time': None
            }
    
    def get_supported_transformations(self) -> List[Dict[str, Any]]:
        """Get list of supported transformations with descriptions."""
        return [
            {
                'name': 'sort_by_wavenumber_desc',
                'description': 'Sort data by wavenumber in descending order',
                'parameters': []
            },
            {
                'name': 'sort_by_wavenumber_asc',
                'description': 'Sort data by wavenumber in ascending order',
                'parameters': []
            },
            {
                'name': 'convert_transmittance_to_absorbance',
                'description': 'Convert transmittance values to absorbance',
                'parameters': []
            },
            {
                'name': 'convert_absorbance_to_transmittance',
                'description': 'Convert absorbance values to transmittance',
                'parameters': []
            },
            {
                'name': 'remove_duplicate_wavenumbers',
                'description': 'Remove duplicate wavenumber entries',
                'parameters': []
            },
            {
                'name': 'interpolate_missing_values',
                'description': 'Interpolate missing values using linear method',
                'parameters': []
            },
            {
                'name': 'remove_negative_values',
                'description': 'Remove or clip negative values in intensity columns',
                'parameters': []
            },
            {
                'name': 'normalize_intensity',
                'description': 'Normalize intensity values to 0-1 range',
                'parameters': []
            },
            {
                'name': 'reverse_order',
                'description': 'Reverse the order of data rows',
                'parameters': []
            },
            {
                'name': 'remove_outliers',
                'description': 'Remove statistical outliers using IQR method',
                'parameters': []
            },
            {
                'name': 'scale_factor_X_column',
                'description': 'Scale column values by factor X',
                'parameters': ['factor', 'column_name']
            },
            {
                'name': 'skip_rows_X',
                'description': 'Skip X rows from the beginning',
                'parameters': ['row_count']
            },
            {
                'name': 'remove_columns_X,Y,Z',
                'description': 'Remove specified columns',
                'parameters': ['column_names']
            }
        ]
    
    def _validate_normalized_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate normalized data structure."""
        validation = {
            'is_valid': True,
            'issues': []
        }
        
        # Check for required columns
        if 'wavenumber' not in df.columns:
            validation['is_valid'] = False
            validation['issues'].append("Missing wavenumber column")
        
        # Check for intensity columns
        intensity_cols = [col for col in df.columns if col in ['absorbance', 'transmittance', 'intensity']]
        if not intensity_cols:
            validation['issues'].append("No intensity/absorbance columns found")
        
        # Check data types
        if 'wavenumber' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['wavenumber']):
                validation['issues'].append("Wavenumber column is not numeric")
        
        return validation
    
    def _create_fallback_plan(self, csv_data: pd.DataFrame, file_path: str, error_msg: str) -> NormalizationPlan:
        """Create fallback normalization plan when AI fails."""
        self.logger.info("Creating fallback normalization plan")
        
        # Simple heuristic-based column mapping
        mappings = []
        
        for i, col_name in enumerate(csv_data.columns):
            col_lower = col_name.lower()
            
            # Try to identify wavenumber column
            if any(keyword in col_lower for keyword in ['wavenumber', 'wave', 'cm-1', 'frequency']):
                mappings.append(ColumnMapping(
                    original_name=col_name,
                    target_name='wavenumber',
                    data_type='numeric',
                    confidence=0.7,
                    notes='Heuristic identification based on column name'
                ))
            
            # Try to identify absorbance/intensity column
            elif any(keyword in col_lower for keyword in ['absorbance', 'abs', 'intensity', 'signal']):
                mappings.append(ColumnMapping(
                    original_name=col_name,
                    target_name='absorbance',
                    data_type='numeric',
                    confidence=0.6,
                    notes='Heuristic identification based on column name'
                ))
            
            else:
                mappings.append(ColumnMapping(
                    original_name=col_name,
                    target_name='metadata',
                    data_type='unknown',
                    confidence=0.3,
                    notes='Fallback classification'
                ))
        
        return NormalizationPlan(
            file_hash=self._generate_file_hash(csv_data, file_path),
            column_mappings=mappings,
            data_transformations=['sort_by_wavenumber_desc'],
            confidence_score=30.0,  # Low confidence for fallback
            confidence_level=ConfidenceLevel.LOW,
            issues_detected=[f"AI normalization failed: {error_msg}"],
            metadata={'fallback_reason': error_msg},
            ai_model='fallback_heuristic',
            timestamp=pd.Timestamp.now().isoformat()
        )
    
    def get_confidence_explanation(self, plan: NormalizationPlan) -> str:
        """Generate human-readable explanation of confidence level."""
        explanations = {
            ConfidenceLevel.HIGH: "High confidence - AI is very certain about the column mappings. Normalization can be applied automatically.",
            ConfidenceLevel.MEDIUM: "Medium confidence - AI has reasonable certainty but recommends user review before applying.",
            ConfidenceLevel.LOW: "Low confidence - AI is uncertain about mappings. Manual review and adjustment required."
        }
        
        base_explanation = explanations.get(plan.confidence_level, "Unknown confidence level")
        
        # Add specific details
        details = []
        if plan.issues_detected:
            details.append(f"Issues detected: {', '.join(plan.issues_detected[:3])}")
        
        high_confidence_mappings = [m for m in plan.column_mappings if m.confidence > 0.8]
        if high_confidence_mappings:
            details.append(f"High confidence mappings: {len(high_confidence_mappings)}/{len(plan.column_mappings)}")
        
        if details:
            return f"{base_explanation}\n\nDetails: {'; '.join(details)}"
        
        return base_explanation
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive AI service usage statistics including cost tracking."""
        try:
            # Get API client stats
            api_stats = await self.api_client.get_usage_stats()
            
            # Get cost tracker stats
            cost_stats = self.cost_tracker.get_usage_statistics()
            
            # Combine statistics
            combined_stats = {
                'api_client': api_stats,
                'cost_tracking': cost_stats.to_dict(),
                'cache_performance': {
                    'cache_hits': self.usage_stats.cache_hits,
                    'total_requests': self.usage_stats.total_requests,
                    'cache_hit_rate': (self.usage_stats.cache_hits / max(1, self.usage_stats.total_requests)) * 100
                }
            }
            
            return combined_stats
            
        except Exception as e:
            self.logger.error(f"Failed to get usage statistics: {e}")
            return {}
    
    async def get_cost_alerts(self) -> List[Dict[str, Any]]:
        """Get recent cost alerts."""
        try:
            alerts = self.cost_tracker.get_recent_alerts()
            return [alert.to_dict() for alert in alerts]
        except Exception as e:
            self.logger.error(f"Failed to get cost alerts: {e}")
            return []
    
    async def export_usage_report(self, filepath: str, format: str = 'csv') -> bool:
        """Export usage report including cost data."""
        try:
            return self.cost_tracker.export_usage_report(filepath, format)
        except Exception as e:
            self.logger.error(f"Failed to export usage report: {e}")
            return False