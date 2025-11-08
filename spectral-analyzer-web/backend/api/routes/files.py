"""
File Management API Routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import tempfile
import shutil
from pathlib import Path
import logging

from core.csv_parser import CSVParser, ParseResult

logger = logging.getLogger(__name__)

router = APIRouter()

# Temporary storage for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "spectral_analyzer_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class FileManager:
    """Manage uploaded files"""
    
    def __init__(self):
        self.uploaded_files = {}
        self.parser = CSVParser()
    
    async def save_upload(self, file: UploadFile) -> dict:
        """Save uploaded file and return info"""
        try:
            # Generate unique filename
            file_id = f"{file.filename}_{hash(file.filename)}"
            file_path = UPLOAD_DIR / file_id
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Store file info
            self.uploaded_files[file_id] = {
                "filename": file.filename,
                "path": str(file_path),
                "size": file_path.stat().st_size
            }
            
            return {
                "file_id": file_id,
                "filename": file.filename,
                "size": file_path.stat().st_size
            }
        except Exception as e:
            logger.error(f"Failed to save upload: {e}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    def get_file_path(self, file_id: str) -> Path:
        """Get path to uploaded file"""
        if file_id not in self.uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        return Path(self.uploaded_files[file_id]["path"])

# Global file manager instance
file_manager = FileManager()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file for analysis
    
    Returns file_id for subsequent operations
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )
        
        file_info = await file_manager.save_upload(file)
        
        return JSONResponse(content={
            "success": True,
            "data": file_info,
            "message": "File uploaded successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple CSV files for batch processing
    
    Returns list of file_ids
    """
    try:
        uploaded_files = []
        errors = []
        
        for file in files:
            try:
                if not file.filename.endswith('.csv'):
                    errors.append(f"{file.filename}: Only CSV files supported")
                    continue
                
                file_info = await file_manager.save_upload(file)
                uploaded_files.append(file_info)
            
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
        
        return JSONResponse(content={
            "success": len(errors) == 0,
            "data": {
                "uploaded": uploaded_files,
                "errors": errors
            },
            "message": f"Uploaded {len(uploaded_files)} of {len(files)} files"
        })
    
    except Exception as e:
        logger.error(f"Batch upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_files():
    """List all uploaded files"""
    try:
        files = [
            {
                "file_id": file_id,
                "filename": info["filename"],
                "size": info["size"]
            }
            for file_id, info in file_manager.uploaded_files.items()
        ]
        
        return JSONResponse(content={
            "success": True,
            "data": files,
            "message": f"Found {len(files)} files"
        })
    
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file"""
    try:
        file_path = file_manager.get_file_path(file_id)
        
        # Delete file
        if file_path.exists():
            file_path.unlink()
        
        # Remove from registry
        del file_manager.uploaded_files[file_id]
        
        return JSONResponse(content={
            "success": True,
            "message": "File deleted successfully"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}/info")
async def get_file_info(file_id: str):
    """Get detailed file information"""
    try:
        file_path = file_manager.get_file_path(file_id)
        
        # Parse file to get structure
        result = file_manager.parser.parse_file(file_path, preview_rows=10)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "file_id": file_id,
                "filename": file_manager.uploaded_files[file_id]["filename"],
                "size": file_manager.uploaded_files[file_id]["size"],
                "structure": {
                    "format_type": result.structure.format_type.value,
                    "row_count": result.structure.row_count,
                    "column_count": result.structure.column_count,
                    "columns": [
                        {
                            "name": col.name,
                            "type": col.data_type.value,
                            "confidence": col.confidence
                        }
                        for col in result.structure.columns
                    ],
                    "encoding": result.structure.encoding,
                    "delimiter": result.structure.delimiter
                },
                "issues": result.issues
            }
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))