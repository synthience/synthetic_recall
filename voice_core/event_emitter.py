from typing import Any, Callable, Dict, List

class EventEmitter:
    """
    A simple event emitter class that allows registering event handlers,
    emitting events with arbitrary arguments, and removing handlers.
    
    Usage:
    
        emitter = EventEmitter()
        
        # Register a handler normally
        def on_event(data):
            print("Event received:", data)
        emitter.on("data", on_event)
        
        # Or register using decorator syntax
        @emitter.on("data")
        def handle_data(data):
            print("Decorator handler received:", data)
        
        # Emit an event:
        emitter.emit("data", {"key": "value"})
        
        # Remove a handler:
        emitter.off("data", on_event)
    """
    
    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event_name: str, handler: Optional[Callable] = None) -> Callable:
        """
        Register an event handler for the specified event.
        If used as a decorator (i.e. without providing a handler),
        the function will be automatically registered.
        
        Args:
            event_name: The name of the event.
            handler: Optional callable to handle the event.
            
        Returns:
            If used as a decorator, returns the wrapped function.
            Otherwise, returns the handler.
        """
        if handler is None:
            # Return a decorator if no handler is passed.
            def decorator(fn: Callable) -> Callable:
                self.on(event_name, fn)
                return fn
            return decorator
        
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)
        return handler

    def off(self, event_name: str, handler: Callable) -> None:
        """
        Remove a registered event handler.
        
        Args:
            event_name: The name of the event.
            handler: The handler to remove.
        """
        if event_name in self._handlers:
            try:
                self._handlers[event_name].remove(handler)
                if not self._handlers[event_name]:
                    del self._handlers[event_name]
            except ValueError:
                # Handler was not found
                pass

    def emit(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        """
        Emit an event to all registered handlers.
        
        Args:
            event_name: The name of the event.
            *args: Positional arguments passed to the handler.
            **kwargs: Keyword arguments passed to the handler.
        """
        if event_name in self._handlers:
            for handler in self._handlers[event_name]:
                handler(*args, **kwargs)
