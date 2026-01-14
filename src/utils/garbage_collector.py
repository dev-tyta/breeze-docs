import gc
import weakref
from typing import Dict, Set, Any
from contextlib import contextmanager
import time

class LLMGarbageCollector:
    """
    Garbage collector for managing LLM resources and cleaning up unused allocations.
    
    Attributes:
        _resource_registry: Registry of active resources
        _last_cleanup: Timestamp of last cleanup
        cleanup_interval: Minimum time between cleanups
        resource_ttl: Time-to-live for resources
    """
    
    def __init__(self, cleanup_interval: int = 300, resource_ttl: int = 600):
        self._resource_registry: Dict[int, weakref.ref] = {}
        self._active_resources: Set[int] = set()
        self._last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
        self.resource_ttl = resource_ttl
        
    def register_resource(self, resource: Any) -> None:
        """Register a resource for tracking"""
        resource_id = id(resource)
        self._resource_registry[resource_id] = weakref.ref(
            resource, 
            lambda ref: self._handle_resource_deletion(resource_id)
        )
        self._active_resources.add(resource_id)
        
    def _handle_resource_deletion(self, resource_id: int) -> None:
        """Handle cleanup when a resource is deleted"""
        self._active_resources.discard(resource_id)
        self._resource_registry.pop(resource_id, None)
        
    def force_cleanup(self) -> None:
        """Force immediate garbage collection"""
        # Run Python's garbage collector
        gc.collect()
        
        # Clean up our resource registry
        current_time = time.time()
        expired_resources = set()
        
        for resource_id in self._active_resources:
            ref = self._resource_registry.get(resource_id)
            if ref is None or ref() is None:
                expired_resources.add(resource_id)
                
        # Remove expired resources
        for resource_id in expired_resources:
            self._handle_resource_deletion(resource_id)
            
        self._last_cleanup = current_time
        
    @contextmanager
    def track_resource(self, resource: Any):
        """Context manager for tracking resource lifecycle"""
        try:
            self.register_resource(resource)
            yield resource
        finally:
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self.force_cleanup()
