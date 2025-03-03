from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EventEmitter:
    """
    Asynchronous event emitter implementation.
    Supports event subscription and emission with async handlers.
    """
    def __init__(self):
        self._events: Dict[str, List[Callable]] = defaultdict(list)
        self._once_events: Dict[str, List[Callable]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register an event handler.
        Can be used as a decorator or method call.
        """
        def decorator(func: Callable) -> Callable:
            self._events[event_name].append(func)
            return func
            
        if handler is None:
            return decorator
        decorator(handler)
        return handler
        
    def once(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register a one-time event handler.
        Handler will be removed after first execution.
        """
        def decorator(func: Callable) -> Callable:
            self._once_events[event_name].append(func)
            return func
            
        if handler is None:
            return decorator
        decorator(handler)
        return handler
        
    def off(self, event_name: str, handler: Callable) -> None:
        """Remove a specific event handler."""
        if event_name in self._events:
            self._events[event_name] = [h for h in self._events[event_name] if h != handler]
        if event_name in self._once_events:
            self._once_events[event_name] = [h for h in self._once_events[event_name] if h != handler]
            
    def remove_all_listeners(self, event_name: Optional[str] = None) -> None:
        """Remove all handlers for an event, or all events if no name given."""
        if event_name:
            self._events[event_name].clear()
            self._once_events[event_name].clear()
        else:
            self._events.clear()
            self._once_events.clear()
            
    async def emit(self, event_name: str, data: Any = None) -> None:
        """
        Emit an event with optional data.
        Executes all handlers asynchronously.
        """
        # Regular handlers
        handlers = self._events.get(event_name, [])
        once_handlers = self._once_events.get(event_name, [])
        
        # Execute handlers
        for handler in handlers + once_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_name}: {e}", exc_info=True)
                
        # Clear once handlers
        if event_name in self._once_events:
            self._once_events[event_name].clear()
