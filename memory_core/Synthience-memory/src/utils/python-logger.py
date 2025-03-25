from datetime import datetime
from typing import Any, Dict, List, Optional

class Logger:
    def __init__(self):
        self.logs = []
    
    def log(self, component: str, message: str, data: Any = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "info",
            "component": component,
            "message": message,
            "data": data
        }
        
        self.logs.append(entry)
        print(f"[{entry['timestamp']}] [{component}] {message}", "" if data is None else data)
    
    def error(self, component: str, message: str, error: Any = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "error",
            "component": component,
            "message": message,
            "data": error
        }
        
        self.logs.append(entry)
        print(f"[{entry['timestamp']}] [{component}] ERROR: {message}", "" if error is None else error)
    
    def debug(self, component: str, message: str, data: Any = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "debug",
            "component": component,
            "message": message,
            "data": data
        }
        
        self.logs.append(entry)
        print(f"[{entry['timestamp']}] [{component}] DEBUG: {message}", "" if data is None else data)
    
    def warn(self, component: str, message: str, data: Any = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "warn",
            "component": component,
            "message": message,
            "data": data
        }
        
        self.logs.append(entry)
        print(f"[{entry['timestamp']}] [{component}] WARN: {message}", "" if data is None else data)
    
    async def cleanup(self):
        # Clear logs
        self.logs = []

# Create singleton logger instance
logger = Logger()
