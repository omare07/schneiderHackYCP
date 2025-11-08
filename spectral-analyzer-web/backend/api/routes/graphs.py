"""
Graph Generation API Routes
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import tempfile
from pathlib import Path
import base64

from core.csv_parser import CSVParser
from core.graph_generator import SpectralGraphGenerator, GraphConfig
from api.routes.files import file_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
csv_parser = CSVParser()
graph_generator = SpectralGraphGenerator()

# Temporary storage for generated graphs
GRAPH_DIR = Path(tempfile.gettempdir()) / "spectral_analyzer_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

class GenerateGraphRequest(BaseModel):
    """Request model for graph generation"""
    baseline_file_id: str
    sample_file_id: str
    format: str = "png"
    title: Optional[str] = None

class BatchGenerateRequest(BaseModel):
    """Request model for batch graph generation"""
    baseline_file_id: str
    sample_file_ids: List[str]
    format: str = "png"

@router.post("/generate")
async def generate_comparison_graph(request: GenerateGraphRequest):
    """
    Generate a comparison graph for baseline and sample
    
    Returns graph image as base64 or file path
    """
    try:
        # Get file paths
        baseline_path = file_manager.get_file_path(request.baseline_file_id)
        sample_path = file_manager.get_file_path(request.sample_file_id)
        
        # Parse files
        baseline_result = csv_parser.parse_file(baseline_path)
        sample_result = csv_parser.parse_file(sample_path)
        
        if not baseline_result.success or not sample_result.success:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse one or more CSV files"
            )
        
        # Get filenames
        baseline_name = file_manager.uploaded_files[request.baseline_file_id]["filename"]
        sample_name = file_manager.uploaded_files[request.sample_file_id]["filename"]
        
        # Generate graph
        config = GraphConfig()
        fig = graph_generator.generate_comparison_graph(
            baseline_result.data,
            sample_result.data,
            baseline_name,
            sample_name,
            config
        )
        
        # Save graph
        output_filename = f"{sample_name}_vs_{baseline_name}.{request.format}"
        output_path = GRAPH_DIR / output_filename
        
        success = graph_generator.save_graph(fig, output_path, request.format)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save graph"
            )
        
        # Read graph and encode as base64
        with open(output_path, 'rb') as f:
            graph_data = base64.b64encode(f.read()).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "graph_id": output_filename,
                "graph_data": f"data:image/{request.format};base64,{graph_data}",
                "format": request.format,
                "baseline_name": baseline_name,
                "sample_name": sample_name
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate graph error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-batch")
async def generate_batch_graphs(request: BatchGenerateRequest):
    """
    Generate comparison graphs for multiple samples against baseline
    
    Returns list of generated graph IDs
    """
    try:
        # Get baseline file
        baseline_path = file_manager.get_file_path(request.baseline_file_id)
        baseline_result = csv_parser.parse_file(baseline_path)
        
        if not baseline_result.success:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse baseline file"
            )
        
        baseline_name = file_manager.uploaded_files[request.baseline_file_id]["filename"]
        
        # Prepare sample datasets
        sample_datasets = []
        for sample_id in request.sample_file_ids:
            sample_path = file_manager.get_file_path(sample_id)
            sample_result = csv_parser.parse_file(sample_path)
            
            if sample_result.success:
                sample_name = file_manager.uploaded_files[sample_id]["filename"]
                sample_datasets.append((sample_result.data, sample_name))
        
        # Generate batch graphs
        batch_result = graph_generator.generate_batch_graphs(
            baseline_result.data,
            sample_datasets,
            str(GRAPH_DIR),
            request.format,
            baseline_name
        )
        
        # Convert file paths to graph IDs
        graph_ids = [Path(path).name for path in batch_result.output_files]
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "total_graphs": batch_result.total_graphs,
                "successful": batch_result.successful,
                "failed": batch_result.failed,
                "graph_ids": graph_ids,
                "errors": batch_result.errors,
                "processing_time": batch_result.processing_time
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch generate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{graph_id}")
async def get_graph(graph_id: str):
    """
    Get a generated graph by ID
    
    Returns graph file
    """
    try:
        graph_path = GRAPH_DIR / graph_id
        
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail="Graph not found")
        
        return FileResponse(
            graph_path,
            media_type="image/png",
            filename=graph_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{graph_id}/base64")
async def get_graph_base64(graph_id: str):
    """
    Get a generated graph as base64 encoded string
    
    Returns graph as base64
    """
    try:
        graph_path = GRAPH_DIR / graph_id
        
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail="Graph not found")
        
        # Read and encode
        with open(graph_path, 'rb') as f:
            graph_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Determine format from extension
        format_ext = graph_path.suffix.lstrip('.')
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "graph_id": graph_id,
                "graph_data": f"data:image/{format_ext};base64,{graph_data}",
                "format": format_ext
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get graph base64 error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{graph_id}")
async def delete_graph(graph_id: str):
    """
    Delete a generated graph
    
    Returns success status
    """
    try:
        graph_path = GRAPH_DIR / graph_id
        
        if not graph_path.exists():
            raise HTTPException(status_code=404, detail="Graph not found")
        
        graph_path.unlink()
        
        return JSONResponse(content={
            "success": True,
            "message": "Graph deleted successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))