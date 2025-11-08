"""
FastAPI Main Application
"""

import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Set
import logging
import asyncio

from api.routes import files, analysis, graphs, stats, demo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Spectral Analyzer API",
    description="Modern web-based spectral analysis application with AI-powered normalization",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and common dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager for real-time progress updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_progress(self, session_id: str, message: str, progress: float):
        """Send progress update to all connections for a session"""
        if session_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json({
                        "type": "progress",
                        "message": message,
                        "progress": progress
                    })
                except Exception as e:
                    logger.error(f"Error sending progress: {e}")
                    dead_connections.add(connection)
            
            # Clean up dead connections
            for conn in dead_connections:
                self.disconnect(conn, session_id)

manager = ConnectionManager()

# Include routers
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(graphs.router, prefix="/api/graphs", tags=["graphs"])
app.include_router(stats.router, prefix="/api/stats", tags=["statistics"])
app.include_router(demo.router, prefix="/api/demo", tags=["demo"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Spectral Analyzer API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "spectral-analyzer-api"
    }

@app.websocket("/api/ws/progress/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back to confirm connection
            await websocket.send_json({
                "type": "ping",
                "message": "connected"
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, session_id)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )