# engram3/utils/caching.py

from typing import Dict, Generic, TypeVar, Optional, Any, List
from collections import OrderedDict

K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type

class CacheManager(Generic[K, V]):
    """
    Generic cache manager with size limits and statistics.
    
    Type Parameters:
        K: Key type
        V: Value type
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._cache: OrderedDict[K, V] = OrderedDict()
        
    def get(self, key: K) -> Optional[V]:
        """Get item from cache with hit tracking."""
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)  # Move to end (most recently used)
            return self._cache[key]
        self.misses += 1
        return None
        
    def set(self, key: K, value: V) -> None:
        """Add item to cache with size management."""
        if len(self._cache) >= self.max_size:
            # Remove 10% oldest entries
            remove_count = self.max_size // 10
            for _ in range(remove_count):
                self._cache.popitem(last=False)  # Remove from start (least recently used)
        self._cache[key] = value
        self._cache.move_to_end(key)
        
    def clear(self) -> None:
        """Clear cache and reset statistics."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def clear_by_pattern(self, pattern: str) -> None:
        """Clear all cache entries containing pattern in key."""
        keys_to_remove = [k for k in self._cache if pattern in str(k)]
        for key in keys_to_remove:
            self._cache.pop(key, None)

    def get_cache_key(self, *args: Any) -> str:
        """Generate unique cache key from arguments."""
        return "_".join(str(arg) for arg in args)
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)
        
    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def __del__(self) -> None:
        """Ensure cache is cleared on deletion."""
        self.clear()