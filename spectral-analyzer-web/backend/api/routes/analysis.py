"""
Analysis API Routes - CSV Parsing and AI Normalization
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import pandas as pd
import json

from core.csv_parser import CSVParser
from core.ai_normalizer import AINormalizer
from core.data_validator import DataValidator
from core.color_analyzer import calculate_grease_color
from api.routes.files import file_manager
from config.env_config import config_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components with config manager
csv_parser = CSVParser()
ai_normalizer = AINormalizer(config_manager=config_manager)
data_validator = DataValidator()

class NormalizeRequest(BaseModel):
    """Request model for normalization"""
    file_id: str
    force_refresh: bool = False

class NormalizeResponse(BaseModel):
    """Response model for normalization"""
    success: bool
    normalized_data: Optional[dict] = None
    plan: Optional[dict] = None
    warnings: Optional[List[str]] = None
    cache_hit: bool = False
    processing_time: float = 0.0

@router.post("/parse")
async def parse_csv(file_id: str):
    """
    Parse CSV file and analyze structure
    
    Returns parsed data with structure analysis
    """
    try:
        file_path = file_manager.get_file_path(file_id)
        
        # Parse file
        result = csv_parser.parse_file(file_path)
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse CSV: {result.error}"
            )
        
        # Convert DataFrame to dict for JSON response
        data_preview = None
        if result.data is not None:
            data_preview = result.data.head(50).to_dict(orient='records')
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "preview": data_preview,
                "structure": {
                    "format_type": result.structure.format_type.value,
                    "row_count": result.structure.row_count,
                    "column_count": result.structure.column_count,
                    "columns": [
                        {
                            "index": col.index,
                            "name": col.name,
                            "type": col.data_type.value,
                            "confidence": col.confidence,
                            "has_missing": col.has_missing,
                            "numeric_range": col.numeric_range
                        }
                        for col in result.structure.columns
                    ],
                    "encoding": result.structure.encoding,
                    "delimiter": result.structure.delimiter,
                    "has_header": result.structure.has_header
                },
                "issues": result.issues,
                "warnings": result.warnings,
                "parsing_time": result.parsing_time
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parse error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/normalize")
async def normalize_csv(request: NormalizeRequest):
    """
    Normalize CSV using AI analysis
    
    Returns normalized data with mapping plan
    """
    try:
        file_path = file_manager.get_file_path(request.file_id)
        
        # Parse file first
        parse_result = csv_parser.parse_file(file_path)
        
        if not parse_result.success or parse_result.data is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse CSV file"
            )
        
        # Normalize using AI
        normalization_result = await ai_normalizer.normalize_csv(
            parse_result.data,
            str(file_path),
            force_refresh=request.force_refresh
        )
        
        if not normalization_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Normalization failed: {normalization_result.error_message}"
            )
        
        # Convert normalized data to dict
        normalized_data_dict = None
        if normalization_result.normalized_data is not None:
            normalized_data_dict = normalization_result.normalized_data.to_dict(orient='records')
        
        # Convert plan to dict
        plan_dict = None
        if normalization_result.plan:
            plan = normalization_result.plan
            plan_dict = {
                "file_hash": plan.file_hash,
                "column_mappings": [
                    {
                        "original_name": m.original_name,
                        "target_name": m.target_name,
                        "data_type": m.data_type,
                        "transformation": m.transformation,
                        "confidence": m.confidence,
                        "notes": m.notes
                    }
                    for m in plan.column_mappings
                ],
                "data_transformations": plan.data_transformations,
                "confidence_score": plan.confidence_score,
                "confidence_level": plan.confidence_level.value,
                "issues_detected": plan.issues_detected,
                "ai_model": plan.ai_model,
                "timestamp": plan.timestamp
            }
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "normalized_data": normalized_data_dict,
                "plan": plan_dict,
                "warnings": normalization_result.warnings or [],
                "cache_hit": normalization_result.cache_hit,
                "processing_time": normalization_result.processing_time
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Normalize error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_data(file_id: str):
    """
    Validate spectral data quality
    
    Returns validation results with issues and recommendations
    """
    try:
        file_path = file_manager.get_file_path(file_id)
        
        # Parse file
        parse_result = csv_parser.parse_file(file_path)
        
        if not parse_result.success or parse_result.data is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse CSV file"
            )
        
        # Validate data
        validation_result = data_validator.validate_data(parse_result.data)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "is_valid": validation_result.is_valid,
                "quality_score": validation_result.quality_score,
                "issues": [
                    {
                        "rule": issue.rule.value,
                        "level": issue.level.value,
                        "message": issue.message,
                        "suggested_fix": issue.suggested_fix
                    }
                    for issue in validation_result.issues
                ],
                "statistics": validation_result.statistics,
                "recommendations": validation_result.recommendations
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ai-status")
async def get_ai_status():
    """
    Check AI service connectivity and status
    
    Returns AI service availability and configuration
    """
    try:
        status = await ai_normalizer.test_ai_connection()
        
        return JSONResponse(content={
            "success": True,
            "data": status
        })
    
    except Exception as e:
        logger.error(f"AI status error: {e}")
        return JSONResponse(content={
            "success": False,
            "data": {
                "openrouter_available": False,
                "openrouter_message": str(e)
            }
        })

@router.get("/transformations")
async def get_supported_transformations():
    """
    Get list of supported data transformations
    
    Returns available transformations with descriptions
    """
    try:
        transformations = ai_normalizer.get_supported_transformations()
        
        return JSONResponse(content={
            "success": True,
            "data": transformations
        })
    
    except Exception as e:
        logger.error(f"Get transformations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interpret")
async def interpret_spectral_data(file_id: str):
    """
    Generate AI interpretation report for spectral data
    
    Uses OpenRouter to analyze normalized spectral data and provide insights
    about chemical composition, functional groups, quality, and anomalies.
    """
    try:
        file_path = file_manager.get_file_path(file_id)
        
        # Parse file
        parse_result = csv_parser.parse_file(file_path)
        
        if not parse_result.success or parse_result.data is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse CSV file"
            )
        
        # Normalize using AI first
        normalization_result = await ai_normalizer.normalize_csv(
            parse_result.data,
            str(file_path),
            force_refresh=False
        )
        
        if not normalization_result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Normalization failed: {normalization_result.error_message}"
            )
        
        # Prepare data summary for AI interpretation
        normalized_df = normalization_result.normalized_data
        
        # Get data statistics
        wavenumber_col = None
        intensity_col = None
        
        for col in normalized_df.columns:
            col_lower = str(col).lower()
            if 'wavenumber' in col_lower or 'wave' in col_lower:
                wavenumber_col = col
            elif 'absorbance' in col_lower or 'intensity' in col_lower or 'transmittance' in col_lower:
                intensity_col = col
        
        if wavenumber_col is None or intensity_col is None:
            # Try first two columns
            cols = list(normalized_df.columns)
            wavenumber_col = cols[0] if len(cols) > 0 else None
            intensity_col = cols[1] if len(cols) > 1 else None
        
        data_summary = {
            "total_points": len(normalized_df),
            "wavenumber_range": {
                "min": float(normalized_df[wavenumber_col].min()) if wavenumber_col else None,
                "max": float(normalized_df[wavenumber_col].max()) if wavenumber_col else None
            },
            "intensity_stats": {
                "min": float(normalized_df[intensity_col].min()) if intensity_col else None,
                "max": float(normalized_df[intensity_col].max()) if intensity_col else None,
                "mean": float(normalized_df[intensity_col].mean()) if intensity_col else None
            },
            "sample_data": normalized_df.head(20).to_dict(orient='records')
        }
        
        # Get CSV preview for the prompt
        wavenumber_min = data_summary['wavenumber_range']['min']
        wavenumber_max = data_summary['wavenumber_range']['max']
        absorbance_min = data_summary['intensity_stats']['min']
        absorbance_max = data_summary['intensity_stats']['max']
        data = normalized_df.to_dict(orient='records')
        csv_preview = normalized_df.head(10).to_string()
        
        # Create interpretation prompt with grease condition monitoring context
        interpretation_prompt = f"""You are analyzing FTIR spectroscopy data for grease condition monitoring in machinery health assessment.

**Context:**
This spectral data comes from grease samples extracted from machinery. The analysis is used by MRG Labs to:
- Determine grease condition and degradation
- Detect contamination levels (water, fuel, dirt)
- Assess equipment health
- Monitor lubricant performance

**Spectral Data Being Analyzed:**
Wavenumber range: {wavenumber_min:.1f} to {wavenumber_max:.1f} cm⁻¹
Number of data points: {len(data)}
Absorbance range: {absorbance_min:.3f} to {absorbance_max:.3f}

**CSV Preview (first 10 rows):**
{csv_preview}

**Please provide a comprehensive analysis including:**

1. **Grease Condition Assessment:**
   - Overall grease health (Good/Fair/Poor)
   - Signs of oxidation or thermal degradation
   - Base oil condition

2. **Contamination Detection:**
   - Water contamination (look for O-H stretch around 3300 cm⁻¹)
   - Fuel dilution (look for C-H peaks)
   - Particulate contamination

3. **Key Spectral Features:**
   - Major peaks and their significance for grease analysis
   - Baseline characteristics
   - Any anomalies or unusual patterns

4. **Functional Group Analysis:**
   - Identify key functional groups in the grease formulation
   - Note any degradation products
   - Assess additive package integrity

5. **Recommendations:**
   - Equipment health assessment
   - Recommended actions (e.g., "Continue monitoring", "Schedule grease change", "Investigate contamination source")
   - Urgency level (Low/Medium/High)

Provide your analysis in a clear, structured JSON format that lab technicians and maintenance engineers can act upon."""
        
        prompt = interpretation_prompt

        # Use OpenRouter API for interpretation
        from utils.api_client import OpenRouterClient
        
        api_key = config_manager.get_api_key('openrouter')
        
        # Don't use async with - OpenRouterClient doesn't support context manager protocol
        client = OpenRouterClient(api_key, config_manager)
        try:
            # Make request
            request = {
                "model": "google/gemini-2.0-flash-001",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert spectroscopist with deep knowledge of FTIR, Raman, UV-Vis, and other spectroscopic techniques. Provide detailed, accurate interpretations of spectral data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2500,
                "response_format": {"type": "json_object"}
            }
            
            response = await client._make_request(request)
            
            if not response.success:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI interpretation failed: {response.error}"
                )
            
            # Parse the interpretation
            interpretation = response.data
            
            # DEBUG: Enhanced logging for structure analysis
            logger.info(f"=== INTERPRETATION DEBUG ===")
            logger.info(f"Response success: {response.success}")
            logger.info(f"Response type: {type(response.data)}")
            logger.info(f"Response data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'Not a dict'}")
            
            # Log the full interpretation to see actual structure
            logger.info(f"Full interpretation content:")
            logger.info(json.dumps(interpretation, indent=2))
            
            # Check if 'analysis' key exists
            if isinstance(interpretation, dict):
                logger.info(f"Has 'analysis' key: {'analysis' in interpretation}")
                logger.info(f"Has 'interpretation' key: {'interpretation' in interpretation}")
                logger.info(f"Top-level keys: {list(interpretation.keys())}")
            
            logger.info(f"=== END DEBUG ===")
            
            # Log what we're actually returning to the frontend
            response_payload = {
                "success": True,
                "data": {
                    "interpretation": interpretation,
                    "data_summary": data_summary,
                    "processing_time": response.response_time,
                    "tokens_used": response.tokens_used,
                    "cost": response.cost
                }
            }
            
            logger.info(f"=== RESPONSE PAYLOAD STRUCTURE ===")
            logger.info(f"Outer keys: {list(response_payload.keys())}")
            logger.info(f"Data keys: {list(response_payload['data'].keys())}")
            logger.info(f"Interpretation is dict: {isinstance(response_payload['data']['interpretation'], dict)}")
            if isinstance(response_payload['data']['interpretation'], dict):
                logger.info(f"Interpretation top-level keys: {list(response_payload['data']['interpretation'].keys())}")
            logger.info(f"=== END RESPONSE PAYLOAD ===")
            
            return JSONResponse(content=response_payload)
        finally:
            # Cleanup
            await client.close()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpretation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/color")
async def analyze_grease_color(file_id: str):
    """
    Analyze grease color from FTIR spectral data
    
    Calculates color based on spectral features including:
    - C-H stretching (hydrocarbon content)
    - CH₂ rocking (chain length)
    - C=O stretching (oxidation level)
    - Overall absorbance (darkness)
    
    Returns RGB color, hex code, description, and analysis details
    """
    try:
        file_path = file_manager.get_file_path(file_id)
        
        # Parse file
        parse_result = csv_parser.parse_file(file_path)
        
        if not parse_result.success or parse_result.data is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse CSV file"
            )
        
        # Get spectral data columns
        df = parse_result.data
        
        # Identify wavenumber and absorbance columns
        wavenumber_col = None
        absorbance_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'wavenumber' in col_lower or 'wave' in col_lower:
                wavenumber_col = col
            elif 'absorbance' in col_lower or 'intensity' in col_lower or 'transmittance' in col_lower:
                absorbance_col = col
        
        # Fallback to first two columns if not found
        if wavenumber_col is None or absorbance_col is None:
            cols = list(df.columns)
            if len(cols) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="CSV file must have at least 2 columns (wavenumber and absorbance)"
                )
            wavenumber_col = cols[0]
            absorbance_col = cols[1]
        
        # Extract data as lists
        wavenumbers = df[wavenumber_col].dropna().tolist()
        absorbances = df[absorbance_col].dropna().tolist()
        
        # Ensure we have data
        if len(wavenumbers) == 0 or len(absorbances) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid spectral data found in file"
            )
        
        # Calculate color
        color_data = calculate_grease_color(wavenumbers, absorbances)
        
        return JSONResponse(content={
            "success": True,
            "data": color_data
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Color analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))