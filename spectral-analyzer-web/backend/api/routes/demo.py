"""
Demo API Routes - Interactive demo workflow with various CSV scenarios
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import time
from pathlib import Path
import pandas as pd
import base64
import math
import numpy as np

from core.csv_parser import CSVParser
from core.ai_normalizer import AINormalizer
from core.graph_generator import SpectralGraphGenerator, GraphConfig
from config.env_config import config_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
csv_parser = CSVParser()
ai_normalizer = AINormalizer(config_manager=config_manager)
graph_generator = SpectralGraphGenerator()

# Demo scenarios configuration
DEMO_SCENARIOS = {
    "german_headers": {
        "name": "Foreign Language Headers",
        "description": "German column headers with transmittance data requiring semantic understanding",
        "file": "demo_german_headers.csv",
        "baseline": "baseline_perfect.csv",
        "issues": [
            "German language headers (Wellenzahl, Durchlässigkeit)",
            "Transmittance % → Absorbance conversion needed",
            "Superscript notation (cm⁻¹)",
            "Irrelevant columns to remove",
            "AI must understand spectroscopy domain"
        ]
    },
    "units_in_cells": {
        "name": "Units in Cell Values",
        "description": "Unit notations mixed within cell values requiring intelligent extraction",
        "file": "demo_units_in_cells.csv",
        "baseline": "baseline_perfect.csv",
        "issues": [
            "Numbers embedded with units (e.g., '0.95 a.u.')",
            "Multiple unit formats (a.u., arb. units, AU)",
            "Wavenumber formats ('cm-1', '/cm', 'cm⁻¹')",
            "Text annotations to ignore",
            "AI must parse semantic context"
        ]
    },
    "inline_metadata": {
        "name": "Inline Metadata Comments",
        "description": "Metadata rows interspersed throughout data requiring context awareness",
        "file": "demo_inline_metadata.csv",
        "baseline": "baseline_perfect.csv",
        "issues": [
            "Metadata scattered in data rows",
            "Comment lines throughout file",
            "AI must distinguish data from metadata",
            "Context understanding required",
            "Requires semantic filtering"
        ]
    },
    "scientific_mixed": {
        "name": "Mixed Scientific Notation",
        "description": "Multiple scientific notation formats requiring intelligent normalization",
        "file": "demo_scientific_mixed.csv",
        "baseline": "baseline_perfect.csv",
        "issues": [
            "Multiple notations (4.0E+03, 3.999e3, 3.999*10^2)",
            "Cryptic column names (X=Wavenumber, Y=Absorbance)",
            "Extraneous metadata columns",
            "AI must recognize patterns",
            "Semantic column identification needed"
        ]
    },
    "cryptic_headers": {
        "name": "Cryptic Abbreviations",
        "description": "Domain-specific abbreviations requiring spectroscopy knowledge",
        "file": "demo_cryptic_headers.csv",
        "baseline": "baseline_perfect.csv",
        "issues": [
            "Cryptic headers (WN, Abs., Tx%, Refl, SNR, BG)",
            "N/A value handling",
            "Transmittance → Absorbance conversion",
            "Column selection (ignore SNR, BG, Comment)",
            "Requires spectroscopy domain expertise",
            "AI must apply scientific knowledge"
        ]
    }
}

# Path to test data - go up to project root then into spectral_analyzer
# From: spectral-analyzer-web/backend/api/routes/demo.py
# To: schneiderHackYCP/spectral_analyzer/tests/test_data
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
TEST_DATA_PATH = PROJECT_ROOT / "spectral_analyzer" / "tests" / "test_data"

def sanitize_for_json(obj):
    """
    Convert NaN/inf values to None for JSON serialization.
    Recursively processes dicts, lists, and numpy arrays.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (np.integer, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return obj

class RunDemoRequest(BaseModel):
    """Request model for running demo"""
    scenario: str

class DemoStepData(BaseModel):
    """Data for a single demo step"""
    step_number: int
    step_name: str
    description: str
    data: Optional[Dict] = None
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0

@router.get("/scenarios")
async def get_demo_scenarios():
    """
    Get list of available demo scenarios
    
    Returns available scenarios with descriptions and expected issues
    """
    try:
        scenarios = [
            {
                "id": scenario_id,
                "name": config["name"],
                "description": config["description"],
                "issues": config["issues"]
            }
            for scenario_id, config in DEMO_SCENARIOS.items()
        ]
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "scenarios": scenarios,
                "total": len(scenarios)
            }
        })
    
    except Exception as e:
        logger.error(f"Get scenarios error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run")
async def run_demo(request: RunDemoRequest):
    """
    Execute the demo workflow for a specific scenario
    
    Returns 4 steps of data:
    1. Raw preview - Original file preview before processing
    2. AI analysis - AI normalization plan and detected issues
    3. Normalization - Normalized data preview
    4. Graph comparison - Visual comparison with baseline
    
    Includes execution time and cost tracking
    """
    try:
        start_time = time.time()
        
        # Validate scenario
        if request.scenario not in DEMO_SCENARIOS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid scenario. Available: {list(DEMO_SCENARIOS.keys())}"
            )
        
        scenario_config = DEMO_SCENARIOS[request.scenario]
        sample_file = TEST_DATA_PATH / scenario_config["file"]
        baseline_file = TEST_DATA_PATH / scenario_config["baseline"]
        
        # Check if files exist
        if not sample_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Sample file not found: {scenario_config['file']}"
            )
        if not baseline_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Baseline file not found: {scenario_config['baseline']}"
            )
        
        logger.info(f"Running demo for scenario: {request.scenario}")
        
        # Initialize steps list
        steps = []
        total_cost = 0.0
        
        # STEP 1: Raw Preview - Show ACTUAL file contents (not parsed data)
        step1_start = time.time()
        try:
            # Read actual raw file contents
            raw_lines = []
            with open(sample_file, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i >= 15:  # Show first 15 lines
                        break
                    raw_lines.append(line.rstrip('\n\r'))
            
            # Also parse to get structure info
            parse_result = csv_parser.parse_file(sample_file, preview_rows=10)
            
            if parse_result.success:
                structure_info = {
                    "filename": scenario_config["file"],
                    "format_type": parse_result.structure.format_type.value,
                    "row_count": parse_result.structure.row_count,
                    "column_count": parse_result.structure.column_count,
                    "delimiter": parse_result.structure.delimiter,
                    "has_header": parse_result.structure.has_header,
                    "encoding": parse_result.structure.encoding
                }
            else:
                structure_info = None
            
            steps.append({
                "step_number": 1,
                "step_name": "Raw Preview",
                "description": "Original file before AI normalization",
                "data": {
                    "raw_text": raw_lines,  # Actual file contents
                    "file_info": structure_info,
                    "detected_issues": parse_result.issues if parse_result.success else [],
                    "warnings": parse_result.warnings if parse_result.success else []
                },
                "success": True,
                "error": None,
                "execution_time": time.time() - step1_start
            })
        except Exception as e:
            logger.error(f"Step 1 failed: {e}", exc_info=True)
            steps.append({
                "step_number": 1,
                "step_name": "Raw Preview",
                "description": "Original file before AI normalization",
                "data": None,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step1_start
            })
        
        # STEP 2: AI Analysis
        step2_start = time.time()
        try:
            # Parse full file for normalization
            parse_result = csv_parser.parse_file(sample_file)
            
            if not parse_result.success or parse_result.data is None:
                raise Exception("Failed to parse sample file for AI analysis")
            
            # Run AI normalization with force_refresh and use_ai
            normalization_result = await ai_normalizer.normalize_csv(
                parse_result.data,
                str(sample_file),
                force_refresh=True,
                use_ai=True
            )
            
            # Track cost
            if hasattr(normalization_result, 'cost') and normalization_result.cost:
                total_cost += normalization_result.cost
            
            # Build plan data
            plan_data = None
            if normalization_result.plan:
                plan = normalization_result.plan
                plan_data = {
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
                    "ai_model": plan.ai_model
                }
            
            steps.append({
                "step_number": 2,
                "step_name": "AI Analysis",
                "description": "AI-powered format detection and normalization plan",
                "data": {
                    "ai_model": plan_data.get("ai_model") if plan_data else None,
                    "confidence_score": plan_data.get("confidence_score", 0) if plan_data else 0,
                    "confidence_level": plan_data.get("confidence_level") if plan_data else None,
                    "column_mappings": plan_data.get("column_mappings", []) if plan_data else [],
                    "transformations_planned": plan_data.get("data_transformations", []) if plan_data else [],
                    "issues_detected": plan_data.get("issues_detected", []) if plan_data else [],
                    "analysis_notes": plan_data.get("metadata", {}).get("analysis_notes", "") if plan_data else "",
                    "raw_ai_response": normalization_result.raw_response if hasattr(normalization_result, 'raw_response') else None,
                    "cache_hit": normalization_result.cache_hit if normalization_result else False,
                    "warnings": normalization_result.warnings if normalization_result else []
                },
                "success": normalization_result.success,
                "error": normalization_result.error_message if not normalization_result.success else None,
                "execution_time": time.time() - step2_start
            })
            
            # Store for next step
            normalized_data = normalization_result.normalized_data
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}", exc_info=True)
            steps.append({
                "step_number": 2,
                "step_name": "AI Analysis",
                "description": "AI-powered format detection and normalization plan",
                "data": None,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step2_start
            })
            normalized_data = None
        
        # STEP 3: Normalized Data
        step3_start = time.time()
        try:
            if normalized_data is not None:
                # Fix duplicate column names if any exist
                if len(normalized_data.columns) != len(set(normalized_data.columns)):
                    cols = list(normalized_data.columns)
                    seen = {}
                    for i, col in enumerate(cols):
                        if col in seen:
                            seen[col] += 1
                            cols[i] = f"{col}_{seen[col]}"
                        else:
                            seen[col] = 0
                    normalized_data.columns = cols
                
                # Convert to dict and sanitize NaN values
                normalized_preview = normalized_data.head(10).to_dict(orient='records')
                normalized_preview = sanitize_for_json(normalized_preview)
                
                # Get statistics
                stats = {
                    "row_count": len(normalized_data),
                    "column_count": len(normalized_data.columns),
                    "columns": list(normalized_data.columns)
                }
                
                # Extract wavenumber and absorbance ranges if available
                if len(normalized_data.columns) >= 2:
                    wavenumber_col = normalized_data.columns[0]
                    intensity_col = normalized_data.columns[1]
                    
                    # Sanitize min/max values
                    wn_min = float(normalized_data[wavenumber_col].min())
                    wn_max = float(normalized_data[wavenumber_col].max())
                    int_min = float(normalized_data[intensity_col].min())
                    int_max = float(normalized_data[intensity_col].max())
                    int_mean = float(normalized_data[intensity_col].mean())
                    
                    stats["wavenumber_range"] = {
                        "min": None if math.isnan(wn_min) or math.isinf(wn_min) else wn_min,
                        "max": None if math.isnan(wn_max) or math.isinf(wn_max) else wn_max
                    }
                    stats["intensity_range"] = {
                        "min": None if math.isnan(int_min) or math.isinf(int_min) else int_min,
                        "max": None if math.isnan(int_max) or math.isinf(int_max) else int_max,
                        "mean": None if math.isnan(int_mean) or math.isinf(int_mean) else int_mean
                    }
                
                # Generate CSV for download (only wavenumber and intensity columns)
                import io
                csv_buffer = io.StringIO()
                # Export only the first two columns (wavenumber and intensity)
                if len(normalized_data.columns) >= 2:
                    export_df = normalized_data.iloc[:, :2].copy()
                    export_df.to_csv(csv_buffer, index=False)
                    csv_content = csv_buffer.getvalue()
                    csv_filename = f"{scenario_config['file']}_normalized.csv"
                else:
                    csv_content = None
                    csv_filename = None
                
                steps.append({
                    "step_number": 3,
                    "step_name": "Normalized Data",
                    "description": "Cleaned and standardized spectral data",
                    "data": {
                        "preview": normalized_preview,
                        "statistics": stats,
                        "normalized_csv": csv_content,
                        "normalized_csv_filename": csv_filename
                    },
                    "success": True,
                    "error": None,
                    "execution_time": time.time() - step3_start
                })
            else:
                raise Exception("No normalized data available from AI analysis")
                
        except Exception as e:
            logger.error(f"Step 3 failed: {e}", exc_info=True)
            steps.append({
                "step_number": 3,
                "step_name": "Normalized Data",
                "description": "Cleaned and standardized spectral data",
                "data": None,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step3_start
            })
        
        # STEP 4: Graph Comparison
        step4_start = time.time()
        try:
            # Parse baseline file
            baseline_result = csv_parser.parse_file(baseline_file)
            
            if not baseline_result.success or baseline_result.data is None:
                raise Exception("Failed to parse baseline file")
            
            if normalized_data is None:
                raise Exception("No normalized data available for graphing")
            
            # Generate comparison graph
            config = GraphConfig()
            fig = graph_generator.generate_comparison_graph(
                baseline_result.data,
                normalized_data,
                scenario_config["baseline"],
                scenario_config["file"],
                config
            )
            
            # Save graph to temporary location
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "spectral_demo_graphs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            graph_filename = f"demo_{request.scenario}_{int(time.time())}.png"
            graph_path = temp_dir / graph_filename
            
            success = graph_generator.save_graph(fig, graph_path, 'png')
            
            if success:
                # Read and encode as base64
                with open(graph_path, 'rb') as f:
                    graph_data = base64.b64encode(f.read()).decode('utf-8')
                
                steps.append({
                    "step_number": 4,
                    "step_name": "Graph Comparison",
                    "description": "Visual comparison of normalized sample vs baseline",
                    "data": {
                        "graph_data": f"data:image/png;base64,{graph_data}",
                        "baseline_name": scenario_config["baseline"],
                        "sample_name": scenario_config["file"]
                    },
                    "success": True,
                    "error": None,
                    "execution_time": time.time() - step4_start
                })
            else:
                raise Exception("Failed to save graph")
            
            # Clean up
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Step 4 failed: {e}", exc_info=True)
            steps.append({
                "step_number": 4,
                "step_name": "Graph Comparison",
                "description": "Visual comparison of normalized sample vs baseline",
                "data": None,
                "success": False,
                "error": str(e),
                "execution_time": time.time() - step4_start
            })
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Determine overall success
        all_success = all(step["success"] for step in steps)
        
        return JSONResponse(content={
            "success": all_success,
            "data": {
                "scenario": {
                    "id": request.scenario,
                    "name": scenario_config["name"],
                    "description": scenario_config["description"],
                    "issues": scenario_config["issues"]
                },
                "steps": steps,
                "total_execution_time": total_time,
                "total_cost": total_cost,
                "all_steps_successful": all_success
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demo run error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))