#!/usr/bin/env python
"""
Launcher script for Synthians Trainer Server
"""

import os
import sys

# Add the parent directory to sys.path so we can import synthians_memory_core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import and run the HTTP server
    from synthians_memory_core.synthians_trainer_server.http_server import app
    import uvicorn
    
    # Configure host and port
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    
    print(f"Starting Synthians Trainer Server at {host}:{port}")
    
    # Run the app
    uvicorn.run(app, host=host, port=port)
