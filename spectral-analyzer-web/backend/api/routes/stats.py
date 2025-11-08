"""
Statistics API Routes - Cache and Cost Tracking
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import logging

from utils.cache_manager import CacheManager
from utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
cache_manager = CacheManager()
cost_tracker = CostTracker()

@router.get("/cache")
async def get_cache_statistics():
    """
    Get cache performance statistics
    
    Returns cache hit rate, memory usage, etc.
    """
    try:
        stats = await cache_manager.get_cache_statistics()
        
        return JSONResponse(content={
            "success": True,
            "data": stats.to_dict()
        })
    
    except Exception as e:
        logger.error(f"Get cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/costs")
async def get_cost_statistics():
    """
    Get API usage cost statistics
    
    Returns total costs, usage by model, etc.
    """
    try:
        stats = cost_tracker.get_usage_statistics()
        
        return JSONResponse(content={
            "success": True,
            "data": stats.to_dict()
        })
    
    except Exception as e:
        logger.error(f"Get cost stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cache entries
    
    Returns success status
    """
    try:
        success = await cache_manager.clear_cache()
        
        return JSONResponse(content={
            "success": success,
            "message": "Cache cleared successfully" if success else "Failed to clear cache"
        })
    
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cache/expired")
async def cleanup_expired():
    """
    Remove expired cache entries
    
    Returns count of cleaned entries
    """
    try:
        count = await cache_manager.cleanup_expired_entries()
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "cleaned_count": count
            },
            "message": f"Cleaned {count} expired entries"
        })
    
    except Exception as e:
        logger.error(f"Cleanup expired error: {e}")
        raise HTTPException(status_code=500, detail=str(e))