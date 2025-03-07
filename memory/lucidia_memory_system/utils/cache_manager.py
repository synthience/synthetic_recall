"""
LUCID RECALL PROJECT
Cache Manager

Implements memory caching strategies for improved performance.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, List, Tuple, Union
from collections import OrderedDict

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for cached values

class CacheManager(Generic[T]):
    """
    Generic cache manager with multiple strategies.
    
    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time To Live) expiration
    - Size limiting
    - Cache statistics
    """
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600, 
                strategy: str = 'lru', name: str = 'cache'):
        """
        Initialize the cache manager.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Default time-to-live in seconds
            strategy: Caching strategy ('lru', 'fifo', 'lfu')
            name: Name of this cache (for stats and debugging)
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.strategy = strategy.lower()
        self.name = name
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Initialize cache
        if self.strategy == 'lru':
            # LRU cache using OrderedDict
            self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        else:
            # Regular cache for other strategies
            self.cache: Dict[str, Dict[str, Any]] = {}
            
        # Access counts for LFU strategy
        self.access_counts: Dict[str, int] = {}
        
        # Stats
        self.stats = {
            'hits': 0,
            'misses': 0,
            'inserts': 0,
            'evictions': 0,
            'expirations': 0
        }
        
        # Set up auto cleanup task if ttl is enabled
        if ttl > 0:
            cleanup_interval = min(ttl / 2, 300)  # Half of TTL or 5 minutes, whichever is less
            self._cleanup_task = asyncio.create_task(self._auto_cleanup(cleanup_interval))
        else:
            self._cleanup_task = None
            
        logger.info(f"Initialized {name} cache with strategy={strategy}, max_size={max_size}, ttl={ttl}")
    
    async def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        async with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return default
                
            # Get cache entry
            entry = self.cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                self.stats['misses'] += 1
                return default
                
            # Update for LRU strategy
            if self.strategy == 'lru':
                # Move to end of OrderedDict (most recently used)
                self.cache.move_to_end(key)
            
            # Update access count for LFU strategy
            if self.strategy == 'lfu':
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
            self.stats['hits'] += 1
            return entry['value']
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL in seconds
        """
        async with self._lock:
            # Check if at max size before adding
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Evict an item
                await self._evict_item()
            
            # Create cache entry
            entry = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl if ttl is not None else self.default_ttl
            }
            
            # Add or update in cache
            self.cache[key] = entry
            
            # Initialize or reset access count
            if self.strategy == 'lfu':
                self.access_counts[key] = 0
                
            self.stats['inserts'] += 1
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Whether the key was found and deleted
        """
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self.cache.clear()
            self.access_counts.clear()
            logger.info(f"Cleared {self.name} cache")
    
    async def keys(self) -> List[str]:
        """Get list of all keys in cache."""
        async with self._lock:
            return list(self.cache.keys())
    
    async def contains(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            Whether the key exists and is not expired
        """
        async with self._lock:
            if key not in self.cache:
                return False
                
            # Check if expired
            if self._is_expired(self.cache[key]):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                return False
                
            return True
    
    async def touch(self, key: str, ttl: Optional[float] = None) -> bool:
        """
        Update the access time for a key.
        
        Args:
            key: Cache key
            ttl: Optional new TTL
            
        Returns:
            Whether the key was found and touched
        """
        async with self._lock:
            if key not in self.cache:
                return False
                
            # Check if expired
            if self._is_expired(self.cache[key]):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                return False
                
            # Update timestamp
            self.cache[key]['timestamp'] = time.time()
            
            # Update TTL if provided
            if ttl is not None:
                self.cache[key]['ttl'] = ttl
                
            # Update for LRU strategy
            if self.strategy == 'lru':
                # Move to end of OrderedDict (most recently used)
                self.cache.move_to_end(key)
                
            return True
    
    async def get_with_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get an item with its metadata.
        
        Args:
            key: Cache key
            
        Returns:
            Dict with value and metadata or None if not found
        """
        async with self._lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
                
            # Get cache entry
            entry = self.cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                # Remove expired entry
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                self.stats['expirations'] += 1
                self.stats['misses'] += 1
                return None
                
            # Update for LRU strategy
            if self.strategy == 'lru':
                # Move to end of OrderedDict (most recently used)
                self.cache.move_to_end(key)
                
            # Update access count for LFU strategy
            if self.strategy == 'lfu':
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
            self.stats['hits'] += 1
            
            # Return entry with metadata
            current_time = time.time()
            age = current_time - entry['timestamp']
            ttl = entry['ttl']
            remaining = max(0, ttl - age) if ttl > 0 else None
            
            return {
                'value': entry['value'],
                'age': age,
                'ttl': ttl,
                'remaining': remaining,
                'timestamp': entry['timestamp']
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            # Calculate hit ratio
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_ratio = self.stats['hits'] / max(1, total_requests)
            
            stats = {
                'name': self.name,
                'strategy': self.strategy,
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_ratio': hit_ratio,
                'inserts': self.stats['inserts'],
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations']
            }
            
            return stats
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a cache entry is expired.
        
        Args:
            entry: Cache entry
            
        Returns:
            Whether the entry is expired
        """
        if entry['ttl'] <= 0:
            # TTL of 0 or negative means never expire
            return False
            
        # Check if elapsed time exceeds TTL
        current_time = time.time()
        age = current_time - entry['timestamp']
        return age > entry['ttl']
    
    async def _evict_item(self) -> None:
        """Evict an item based on the selected strategy."""
        if not self.cache:
            return
            
        if self.strategy == 'lru':
            # LRU - remove first item in OrderedDict (least recently used)
            self.cache.popitem(last=False)
            self.stats['evictions'] += 1
            
        elif self.strategy == 'fifo':
            # FIFO - remove oldest inserted item
            # Find oldest item by timestamp
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            if oldest_key in self.access_counts:
                del self.access_counts[oldest_key]
            self.stats['evictions'] += 1
            
        elif self.strategy == 'lfu':
            # LFU - remove least frequently used item
            # Find key with lowest access count
            least_used_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            del self.cache[least_used_key]
            del self.access_counts[least_used_key]
            self.stats['evictions'] += 1
            
        else:
            # Default - remove random item
            random_key = next(iter(self.cache))
            del self.cache[random_key]
            if random_key in self.access_counts:
                del self.access_counts[random_key]
            self.stats['evictions'] += 1
    
    async def _auto_cleanup(self, interval: float) -> None:
        """
        Periodically clean up expired entries.
        
        Args:
            interval: Cleanup interval in seconds
        """
        try:
            while True:
                # Wait for interval
                await asyncio.sleep(interval)
                
                # Clean up expired entries
                await self.cleanup_expired()
                
        except asyncio.CancelledError:
            # Task cancelled, exit gracefully
            logger.info(f"Cleanup task for {self.name} cache cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup task: {e}")
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = []
            
            # Find expired entries
            for key, entry in list(self.cache.items()):
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_counts:
                    del self.access_counts[key]
            
            # Update stats
            self.stats['expirations'] += len(expired_keys)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries from {self.name} cache")
                
            return len(expired_keys)
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass