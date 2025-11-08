"""
Enhanced caching system for normalization plans and processed data.

Provides intelligent multi-tier caching with TTL, file-based and Redis storage,
comprehensive performance metrics, and cost optimization for AI API usage.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import sqlite3
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import sys
import gzip
import pandas as pd

if TYPE_CHECKING:
    from core.ai_normalizer import NormalizationPlan

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    FILE = "file"
    REDIS = "redis"


class CompressionType(Enum):
    """Compression types for cache entries."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


@dataclass
class CacheStatistics:
    """Comprehensive cache performance metrics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    memory_usage_mb: float
    disk_usage_mb: float
    avg_lookup_time_ms: float
    total_entries: int
    expired_entries: int
    compression_ratio: float
    cache_levels: Dict[str, int]
    most_accessed_keys: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int
    last_accessed: datetime
    data_type: str
    compressed: bool
    compression_type: CompressionType
    original_size: int
    compressed_size: int
    cache_level: CacheLevel
    file_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.now() >= self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    memory_limit_mb: int = 100
    disk_limit_mb: int = 1000
    default_ttl_hours: int = 24
    cleanup_interval_minutes: int = 60
    compression_threshold_kb: int = 10
    compression_type: CompressionType = CompressionType.LZ4 if LZ4_AVAILABLE else CompressionType.GZIP
    enable_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    max_memory_entries: int = 1000
    enable_background_cleanup: bool = True


class CacheManager:
    """
    Enhanced multi-tier caching system for spectral analysis data.
    
    Features:
    - Multi-level memory cache with LRU eviction
    - Compressed file-based cache for persistence
    - Optional Redis cache for distributed deployment
    - TTL-based expiration with background cleanup
    - Comprehensive performance metrics and optimization
    - Intelligent file structure hashing
    - Cost optimization tracking
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, config: Optional[CacheConfig] = None):
        """
        Initialize the enhanced cache manager.
        
        Args:
            cache_dir: Directory for file-based cache
            config: Cache configuration settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = config or CacheConfig()
        
        # Cache directory setup
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".spectral_analyzer" / "cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Multi-tier cache storage
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_access_order: List[str] = []  # LRU tracking
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'lookup_times': [],
            'memory_hits': 0,
            'file_hits': 0,
            'redis_hits': 0
        }
        
        # Database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Redis cache (optional)
        self.redis_client = None
        if self.config.enable_redis:
            self._init_redis()
        
        # Background cleanup
        self.cleanup_thread = None
        self.cleanup_stop_event = threading.Event()
        if self.config.enable_background_cleanup:
            self._start_background_cleanup()
        
        self.logger.debug("Enhanced cache manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for enhanced cache metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        file_path TEXT,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TEXT,
                        data_type TEXT,
                        file_size INTEGER,
                        original_size INTEGER,
                        compressed_size INTEGER,
                        compression_type TEXT,
                        cache_level TEXT,
                        file_hash TEXT,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_data_type ON cache_entries(data_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON cache_entries(file_hash)")
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize cache database: {e}")
    
    def _init_redis(self):
        """Initialize Redis client if available."""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                decode_responses=False  # We'll handle binary data
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis cache enabled")
            
        except ImportError:
            self.logger.warning("Redis not available, using file cache only")
            self.config.enable_redis = False
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.config.enable_redis = False
    
    def _start_background_cleanup(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self.cleanup_stop_event.wait(self.config.cleanup_interval_minutes * 60):
                try:
                    asyncio.run(self.cleanup_expired())
                except Exception as e:
                    self.logger.error(f"Background cleanup failed: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        self.logger.debug("Background cleanup thread started")
    
    def generate_file_structure_hash(self, csv_data: pd.DataFrame, file_path: str = None) -> str:
        """
        Generate intelligent hash for CSV file structure for caching.
        
        Args:
            csv_data: DataFrame containing CSV data
            file_path: Optional file path for additional context
            
        Returns:
            Hash string representing file structure
        """
        try:
            # Create comprehensive structure signature
            structure_info = {
                'columns': list(csv_data.columns),
                'dtypes': csv_data.dtypes.astype(str).to_dict(),
                'shape': csv_data.shape,
                'column_count': len(csv_data.columns),
                'row_count': len(csv_data)
            }
            
            # Add sample data from first few rows for better differentiation
            if not csv_data.empty:
                sample_size = min(10, len(csv_data))
                structure_info['sample_data'] = {}
                
                for col in csv_data.columns:
                    sample_values = csv_data[col].head(sample_size).tolist()
                    # Convert to strings and handle NaN values
                    sample_values = [str(v) if pd.notna(v) else 'NaN' for v in sample_values]
                    structure_info['sample_data'][col] = sample_values
            
            # Add column statistics for numeric columns
            numeric_cols = csv_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                structure_info['numeric_stats'] = {}
                for col in numeric_cols:
                    if not csv_data[col].empty:
                        stats = {
                            'min': float(csv_data[col].min()) if pd.notna(csv_data[col].min()) else None,
                            'max': float(csv_data[col].max()) if pd.notna(csv_data[col].max()) else None,
                            'mean': float(csv_data[col].mean()) if pd.notna(csv_data[col].mean()) else None
                        }
                        structure_info['numeric_stats'][col] = stats
            
            # Generate hash from structure
            structure_str = json.dumps(structure_info, sort_keys=True)
            file_hash = hashlib.sha256(structure_str.encode()).hexdigest()
            
            self.logger.debug(f"Generated file structure hash: {file_hash[:12]}...")
            return file_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to generate file structure hash: {e}")
            # Fallback to simple hash
            fallback_data = f"{csv_data.shape}_{list(csv_data.columns)}"
            return hashlib.sha256(fallback_data.encode()).hexdigest()
    
    def _compress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified compression type."""
        try:
            if compression_type == CompressionType.GZIP:
                return gzip.compress(data)
            elif compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
                return lz4.frame.compress(data)
            else:
                return data
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return data
    
    def _decompress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified compression type."""
        try:
            if compression_type == CompressionType.GZIP:
                return gzip.decompress(data)
            elif compression_type == CompressionType.LZ4 and LZ4_AVAILABLE:
                return lz4.frame.decompress(data)
            else:
                return data
        except Exception as e:
            self.logger.warning(f"Decompression failed: {e}")
            return data
    
    def _should_compress(self, data_size: int) -> bool:
        """Determine if data should be compressed based on size."""
        return data_size > (self.config.compression_threshold_kb * 1024)
    
    def _update_lru_access(self, key: str):
        """Update LRU access order for memory cache."""
        with self.cache_lock:
            if key in self.memory_cache_access_order:
                self.memory_cache_access_order.remove(key)
            self.memory_cache_access_order.append(key)
    
    def _evict_lru_entries(self):
        """Evict least recently used entries if memory limit exceeded."""
        with self.cache_lock:
            while len(self.memory_cache) > self.config.max_memory_entries:
                if not self.memory_cache_access_order:
                    break
                
                lru_key = self.memory_cache_access_order.pop(0)
                if lru_key in self.memory_cache:
                    del self.memory_cache[lru_key]
                    self.logger.debug(f"Evicted LRU entry: {lru_key}")
    
    async def store_normalization_plan(self, file_hash: str, plan: 'NormalizationPlan',
                                     ttl: Optional[timedelta] = None) -> bool:
        """
        Store normalization plan in enhanced multi-tier cache.
        
        Args:
            file_hash: File structure hash
            plan: Normalization plan to store
            ttl: Time to live (optional)
            
        Returns:
            True if stored successfully
        """
        start_time = time.time()
        
        try:
            cache_key = f"normalization:{file_hash}"
            
            if ttl is None:
                ttl = timedelta(hours=self.config.default_ttl_hours)
            
            expires_at = datetime.now() + ttl
            
            # Serialize plan
            plan_data = {
                'file_hash': plan.file_hash,
                'column_mappings': [
                    {
                        'original_name': m.original_name,
                        'target_name': m.target_name,
                        'data_type': m.data_type,
                        'transformation': m.transformation,
                        'confidence': m.confidence,
                        'notes': m.notes
                    }
                    for m in plan.column_mappings
                ],
                'data_transformations': plan.data_transformations,
                'confidence_score': plan.confidence_score,
                'confidence_level': plan.confidence_level.value,
                'issues_detected': plan.issues_detected,
                'metadata': plan.metadata,
                'ai_model': plan.ai_model,
                'timestamp': plan.timestamp
            }
            
            # Serialize to JSON
            json_data = json.dumps(plan_data, indent=None)
            original_size = len(json_data.encode())
            
            # Determine compression
            should_compress = self._should_compress(original_size)
            compression_type = self.config.compression_type if should_compress else CompressionType.NONE
            
            # Compress if needed
            if should_compress:
                compressed_data = self._compress_data(json_data.encode(), compression_type)
                compressed_size = len(compressed_data)
            else:
                compressed_data = json_data.encode()
                compressed_size = original_size
            
            # Create cache entry
            cache_entry = CacheEntry(
                key=cache_key,
                data=plan_data,
                created_at=datetime.now(),
                expires_at=expires_at,
                access_count=1,
                last_accessed=datetime.now(),
                data_type='normalization_plan',
                compressed=should_compress,
                compression_type=compression_type,
                original_size=original_size,
                compressed_size=compressed_size,
                cache_level=CacheLevel.MEMORY,
                file_hash=file_hash
            )
            
            # Store in memory cache
            with self.cache_lock:
                self.memory_cache[cache_key] = cache_entry
                self._update_lru_access(cache_key)
                self._evict_lru_entries()
            
            # Store in file cache
            file_path = self.cache_dir / f"{cache_key}.cache"
            try:
                with open(file_path, 'wb') as f:
                    # Write metadata header
                    metadata = {
                        'expires_at': expires_at.isoformat(),
                        'data_type': 'normalization_plan',
                        'compressed': should_compress,
                        'compression_type': compression_type.value,
                        'original_size': original_size,
                        'compressed_size': compressed_size
                    }
                    header = json.dumps(metadata).encode()
                    f.write(len(header).to_bytes(4, 'big'))
                    f.write(header)
                    f.write(compressed_data)
                
                cache_entry.cache_level = CacheLevel.FILE
            except Exception as e:
                self.logger.warning(f"Failed to store in file cache: {e}")
            
            # Store in Redis if enabled
            if self.redis_client:
                try:
                    redis_data = {
                        'data': json_data,
                        'metadata': {
                            'expires_at': expires_at.isoformat(),
                            'data_type': 'normalization_plan',
                            'original_size': original_size
                        }
                    }
                    ttl_seconds = int(ttl.total_seconds())
                    self.redis_client.setex(cache_key, ttl_seconds, json.dumps(redis_data))
                    cache_entry.cache_level = CacheLevel.REDIS
                except Exception as e:
                    self.logger.warning(f"Redis storage failed: {e}")
            
            # Update database metadata
            self._store_cache_metadata(cache_entry, file_path)
            
            storage_time = (time.time() - start_time) * 1000
            self.logger.debug(f"Stored normalization plan: {cache_key[:12]}... ({storage_time:.2f}ms)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store normalization plan: {e}")
            return False
    
    def set_normalization_plan(self, file_hash: str, plan_data: Dict[str, Any],
                              ttl: Optional[timedelta] = None) -> bool:
        """
        Store normalization plan in cache - synchronous compatibility method.
        
        Args:
            file_hash: File structure hash
            plan_data: Normalization plan data (dict format)
            ttl: Time to live (optional)
            
        Returns:
            True if stored successfully
        """
        # For compatibility with integration tests, create a NormalizationPlan from dict
        try:
            from core.ai_normalizer import NormalizationPlan, ColumnMapping, ConfidenceLevel
            
            # If plan_data is already a NormalizationPlan, use it directly
            if isinstance(plan_data, NormalizationPlan):
                plan = plan_data
            else:
                # Convert dict to NormalizationPlan
                column_mappings = []
                for mapping_data in plan_data.get('column_mapping', {}).items():
                    column_mappings.append(ColumnMapping(
                        original_name=mapping_data[0],
                        target_name=mapping_data[1],
                        data_type='numeric',
                        confidence=plan_data.get('confidence', 0.85)
                    ))
                
                plan = NormalizationPlan(
                    file_hash=file_hash,
                    column_mappings=column_mappings,
                    data_transformations=[],
                    confidence_score=plan_data.get('confidence', 0.85) * 100,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    issues_detected=[],
                    metadata=plan_data,
                    ai_model='test',
                    timestamp=datetime.now().isoformat()
                )
            
            # Use asyncio to run the async method
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.store_normalization_plan(file_hash, plan, ttl))
            loop.close()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to set normalization plan: {e}")
            return False
    
    def get_normalization_plan_sync(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get normalization plan from cache - synchronous compatibility method.
        
        Args:
            file_hash: File structure hash
            
        Returns:
            Normalization plan dict if found, None otherwise
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            plan = loop.run_until_complete(self.get_normalization_plan(file_hash))
            loop.close()
            
            if plan:
                # Return as dict for compatibility
                return {
                    'column_mapping': {m.original_name: m.target_name for m in plan.column_mappings},
                    'confidence': plan.confidence_score / 100,
                    'timestamp': plan.timestamp
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get normalization plan sync: {e}")
            return None
    
    async def get_normalization_plan(self, file_hash: str) -> Optional['NormalizationPlan']:
        """
        Retrieve normalization plan from enhanced multi-tier cache.
        
        Args:
            file_hash: File structure hash
            
        Returns:
            NormalizationPlan if found and valid, None otherwise
        """
        start_time = time.time()
        cache_key = f"normalization:{file_hash}"
        
        try:
            self.stats['total_requests'] += 1
            
            # Check memory cache first
            with self.cache_lock:
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    if not entry.is_expired:
                        entry.access_count += 1
                        entry.last_accessed = datetime.now()
                        self._update_lru_access(cache_key)
                        
                        self.stats['cache_hits'] += 1
                        self.stats['memory_hits'] += 1
                        
                        lookup_time = (time.time() - start_time) * 1000
                        self.stats['lookup_times'].append(lookup_time)
                        
                        self.logger.debug(f"Memory cache hit: {cache_key[:12]}... ({lookup_time:.2f}ms)")
                        return self._deserialize_normalization_plan(entry.data)
                    else:
                        # Expired, remove from memory cache
                        del self.memory_cache[cache_key]
                        if cache_key in self.memory_cache_access_order:
                            self.memory_cache_access_order.remove(cache_key)
            
            # Check Redis cache
            if self.redis_client:
                try:
                    redis_data = self.redis_client.get(cache_key)
                    if redis_data:
                        redis_entry = json.loads(redis_data)
                        plan_data = json.loads(redis_entry['data'])
                        
                        # Store in memory cache for faster access
                        cache_entry = CacheEntry(
                            key=cache_key,
                            data=plan_data,
                            created_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(hours=self.config.default_ttl_hours),
                            access_count=1,
                            last_accessed=datetime.now(),
                            data_type='normalization_plan',
                            compressed=False,
                            compression_type=CompressionType.NONE,
                            original_size=redis_entry['metadata']['original_size'],
                            compressed_size=redis_entry['metadata']['original_size'],
                            cache_level=CacheLevel.REDIS,
                            file_hash=file_hash
                        )
                        
                        with self.cache_lock:
                            self.memory_cache[cache_key] = cache_entry
                            self._update_lru_access(cache_key)
                            self._evict_lru_entries()
                        
                        self.stats['cache_hits'] += 1
                        self.stats['redis_hits'] += 1
                        
                        lookup_time = (time.time() - start_time) * 1000
                        self.stats['lookup_times'].append(lookup_time)
                        
                        self.logger.debug(f"Redis cache hit: {cache_key[:12]}... ({lookup_time:.2f}ms)")
                        return self._deserialize_normalization_plan(plan_data)
                        
                except Exception as e:
                    self.logger.warning(f"Redis retrieval failed: {e}")
            
            # Check file cache
            file_path = self.cache_dir / f"{cache_key}.cache"
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        # Read metadata header
                        header_size = int.from_bytes(f.read(4), 'big')
                        header_data = f.read(header_size)
                        metadata = json.loads(header_data.decode())
                        
                        # Check expiration
                        expires_at = datetime.fromisoformat(metadata['expires_at'])
                        if datetime.now() < expires_at:
                            # Read and decompress data
                            compressed_data = f.read()
                            
                            if metadata.get('compressed', False):
                                compression_type = CompressionType(metadata['compression_type'])
                                json_data = self._decompress_data(compressed_data, compression_type).decode()
                            else:
                                json_data = compressed_data.decode()
                            
                            plan_data = json.loads(json_data)
                            
                            # Store in memory cache for faster access
                            cache_entry = CacheEntry(
                                key=cache_key,
                                data=plan_data,
                                created_at=datetime.now(),
                                expires_at=expires_at,
                                access_count=1,
                                last_accessed=datetime.now(),
                                data_type='normalization_plan',
                                compressed=metadata.get('compressed', False),
                                compression_type=CompressionType(metadata.get('compression_type', 'none')),
                                original_size=metadata.get('original_size', 0),
                                compressed_size=metadata.get('compressed_size', 0),
                                cache_level=CacheLevel.FILE,
                                file_hash=file_hash
                            )
                            
                            with self.cache_lock:
                                self.memory_cache[cache_key] = cache_entry
                                self._update_lru_access(cache_key)
                                self._evict_lru_entries()
                            
                            # Update access metadata
                            self._update_access_metadata(cache_key)
                            
                            self.stats['cache_hits'] += 1
                            self.stats['file_hits'] += 1
                            
                            lookup_time = (time.time() - start_time) * 1000
                            self.stats['lookup_times'].append(lookup_time)
                            
                            self.logger.debug(f"File cache hit: {cache_key[:12]}... ({lookup_time:.2f}ms)")
                            return self._deserialize_normalization_plan(plan_data)
                        else:
                            # Expired, remove file
                            file_path.unlink()
                            self._remove_cache_metadata(cache_key)
                            
                except Exception as e:
                    self.logger.warning(f"File cache read failed: {e}")
            
            # Cache miss
            self.stats['cache_misses'] += 1
            lookup_time = (time.time() - start_time) * 1000
            self.stats['lookup_times'].append(lookup_time)
            
            self.logger.debug(f"Cache miss: {cache_key[:12]}... ({lookup_time:.2f}ms)")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve normalization plan: {e}")
            return None
    
    def _deserialize_normalization_plan(self, plan_data: Dict[str, Any]) -> 'NormalizationPlan':
        """Deserialize normalization plan from cache data."""
        from core.ai_normalizer import NormalizationPlan, ColumnMapping, ConfidenceLevel
        
        # Reconstruct column mappings
        column_mappings = []
        for mapping_data in plan_data['column_mappings']:
            mapping = ColumnMapping(
                original_name=mapping_data['original_name'],
                target_name=mapping_data['target_name'],
                data_type=mapping_data['data_type'],
                transformation=mapping_data.get('transformation'),
                confidence=mapping_data['confidence'],
                notes=mapping_data.get('notes')
            )
            column_mappings.append(mapping)
        
        # Reconstruct confidence level
        confidence_level = ConfidenceLevel(plan_data['confidence_level'])
        
        # Create normalization plan
        plan = NormalizationPlan(
            file_hash=plan_data['file_hash'],
            column_mappings=column_mappings,
            data_transformations=plan_data['data_transformations'],
            confidence_score=plan_data['confidence_score'],
            confidence_level=confidence_level,
            issues_detected=plan_data['issues_detected'],
            metadata=plan_data['metadata'],
            ai_model=plan_data['ai_model'],
            timestamp=plan_data['timestamp']
        )
        
        return plan
    
    def _store_cache_metadata(self, entry: CacheEntry, file_path: Path):
        """Store cache entry metadata in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, file_path, created_at, expires_at, access_count, last_accessed, 
                     data_type, file_size, original_size, compressed_size, compression_type,
                     cache_level, file_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    str(file_path) if file_path else None,
                    entry.created_at.isoformat(),
                    entry.expires_at.isoformat(),
                    entry.access_count,
                    entry.last_accessed.isoformat(),
                    entry.data_type,
                    file_path.stat().st_size if file_path and file_path.exists() else 0,
                    entry.original_size,
                    entry.compressed_size,
                    entry.compression_type.value,
                    entry.cache_level.value,
                    entry.file_hash,
                    json.dumps(entry.metadata) if entry.metadata else None
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Failed to store cache metadata: {e}")
    
    def _update_access_metadata(self, key: str):
        """Update access metadata for cache entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE key = ?
                """, (datetime.now().isoformat(), key))
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Failed to update access metadata: {e}")
    
    def _remove_cache_metadata(self, key: str):
        """Remove cache metadata from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Failed to remove cache metadata: {e}")
    
    async def get_cache_statistics(self) -> CacheStatistics:
        """
        Get comprehensive cache statistics.
        
        Returns:
            CacheStatistics object with detailed metrics
        """
        try:
            # Calculate hit rate
            total_requests = self.stats['total_requests']
            cache_hits = self.stats['cache_hits']
            hit_rate = (cache_hits / max(1, total_requests)) * 100
            
            # Calculate average lookup time
            lookup_times = self.stats['lookup_times']
            avg_lookup_time = sum(lookup_times) / max(1, len(lookup_times)) if lookup_times else 0
            
            # Memory usage
            memory_usage_mb = 0
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
            
            # Disk usage
            disk_usage_mb = 0
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    disk_usage_mb += file_path.stat().st_size
                disk_usage_mb /= (1024 * 1024)
            except Exception:
                pass
            
            # Cache levels distribution
            cache_levels = {
                'memory': len(self.memory_cache),
                'file': 0,
                'redis': 0
            }
            
            # Most accessed keys
            most_accessed = []
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT key, access_count, last_accessed, data_type
                        FROM cache_entries 
                        WHERE expires_at > ?
                        ORDER BY access_count DESC
                        LIMIT 10
                    """, (datetime.now().isoformat(),))
                    
                    for row in cursor.fetchall():
                        most_accessed.append({
                            'key': row[0],
                            'access_count': row[1],
                            'last_accessed': row[2],
                            'data_type': row[3]
                        })
                    
                    # Get file cache count
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM cache_entries 
                        WHERE expires_at > ? AND cache_level = 'file'
                    """, (datetime.now().isoformat(),))
                    cache_levels['file'] = cursor.fetchone()[0] or 0
                    
            except Exception as e:
                self.logger.warning(f"Failed to get database statistics: {e}")
            
            # Performance metrics
            performance_metrics = {
                'memory_hit_rate': (self.stats['memory_hits'] / max(1, total_requests)) * 100,
                'file_hit_rate': (self.stats['file_hits'] / max(1, total_requests)) * 100,
                'redis_hit_rate': (self.stats['redis_hits'] / max(1, total_requests)) * 100,
                'avg_lookup_time_ms': avg_lookup_time,
                'total_requests': total_requests
            }
            
            # Compression ratio (simplified calculation)
            compression_ratio = 0.7  # Placeholder - would need actual calculation
            
            return CacheStatistics(
                total_requests=total_requests,
                cache_hits=cache_hits,
                cache_misses=self.stats['cache_misses'],
                hit_rate=hit_rate,
                memory_usage_mb=memory_usage_mb,
                disk_usage_mb=disk_usage_mb,
                avg_lookup_time_ms=avg_lookup_time,
                total_entries=sum(cache_levels.values()),
                expired_entries=0,  # Would need separate tracking
                compression_ratio=compression_ratio,
                cache_levels=cache_levels,
                most_accessed_keys=most_accessed,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {e}")
            # Return empty statistics
            return CacheStatistics(
                total_requests=0, cache_hits=0, cache_misses=0, hit_rate=0.0,
                memory_usage_mb=0.0, disk_usage_mb=0.0, avg_lookup_time_ms=0.0,
                total_entries=0, expired_entries=0, compression_ratio=0.0,
                cache_levels={}, most_accessed_keys=[], performance_metrics={}
            )
    
    async def cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries across all tiers.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        
        try:
            now = datetime.now()
            
            # Clean memory cache
            with self.cache_lock:
                expired_keys = [
                    key for key, entry in self.memory_cache.items()
                    if entry.is_expired
                ]
                
                for key in expired_keys:
                    del self.memory_cache[key]
                    if key in self.memory_cache_access_order:
                        self.memory_cache_access_order.remove(key)
                    cleaned_count += 1
            
            # Clean file cache
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT key, file_path FROM cache_entries WHERE expires_at < ?",
                    (now.isoformat(),)
                )
                
                expired_entries = cursor.fetchall()
                
                for key, file_path in expired_entries:
                    try:
                        if file_path:
                            Path(file_path).unlink(missing_ok=True)
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove cache file {file_path}: {e}")
                
                # Remove metadata for expired entries
                conn.execute("DELETE FROM cache_entries WHERE expires_at < ?", (now.isoformat(),))
                conn.commit()
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
            return cleaned_count
    
    async def clear_cache(self, data_type: Optional[str] = None) -> bool:
        """
        Clear cache entries with optional type filter.
        
        Args:
            data_type: Optional data type filter
            
        Returns:
            True if cleared successfully
        """
        try:
            # Clear memory cache
            with self.cache_lock:
                if data_type:
                    keys_to_remove = [
                        key for key, entry in self.memory_cache.items()
                        if entry.data_type == data_type
                    ]
                else:
                    keys_to_remove = list(self.memory_cache.keys())
                
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.memory_cache_access_order:
                        self.memory_cache_access_order.remove(key)
            
            # Clear file cache
            with sqlite3.connect(self.db_path) as conn:
                if data_type:
                    cursor = conn.execute(
                        "SELECT file_path FROM cache_entries WHERE data_type = ?",
                        (data_type,)
                    )
                else:
                    cursor = conn.execute("SELECT file_path FROM cache_entries")
                
                file_paths = [row[0] for row in cursor.fetchall() if row[0]]
                
                for file_path in file_paths:
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove cache file {file_path}: {e}")
                
                # Clear metadata
                if data_type:
                    conn.execute("DELETE FROM cache_entries WHERE data_type = ?", (data_type,))
                else:
                    conn.execute("DELETE FROM cache_entries")
                
                conn.commit()
            
            # Clear Redis cache if enabled
            if self.redis_client:
                try:
                    if data_type:
                        # Scan and delete keys by pattern (complex operation)
                        pattern = f"*{data_type}*"
                        for key in self.redis_client.scan_iter(match=pattern):
                            self.redis_client.delete(key)
                    else:
                        self.redis_client.flushdb()
                except Exception as e:
                    self.logger.warning(f"Redis cache clear failed: {e}")
            
            # Reset statistics
            if not data_type:
                self.stats = {
                    'total_requests': 0,
                    'cache_hits': 0,
                    'cache_misses': 0,
                    'lookup_times': [],
                    'memory_hits': 0,
                    'file_hits': 0,
                    'redis_hits': 0
                }
            
            self.logger.info(f"Cache cleared{f' for {data_type}' if data_type else ''}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get cache size information across all tiers."""
        try:
            info = {
                'memory_entries': len(self.memory_cache),
                'memory_limit_mb': self.config.memory_limit_mb,
                'disk_limit_mb': self.config.disk_limit_mb,
                'disk_usage_mb': 0,
                'file_count': 0
            }
            
            # Calculate disk usage
            try:
                total_size = 0
                file_count = 0
                for file_path in self.cache_dir.glob("*.cache"):
                    total_size += file_path.stat().st_size
                    file_count += 1
                
                info['disk_usage_mb'] = total_size / (1024 * 1024)
                info['file_count'] = file_count
            except Exception:
                pass
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get cache size info: {e}")
            return {}
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            # Stop background cleanup
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_stop_event.set()
                self.cleanup_thread.join(timeout=5)
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
                
        except Exception:
            pass