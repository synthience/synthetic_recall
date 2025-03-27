# synthians_memory_core/run_server.py

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Ensure we can import from the synthians_memory_core package
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    """Run the Synthians Memory Core API server"""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5010"))
    
    print(f"Starting Synthians Memory Core API server at {host}:{port}")
    
    # Use Uvicorn to run the FastAPI application
    uvicorn.run(
        "synthians_memory_core.api.server:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1      # Single worker for memory consistency
    )

if __name__ == "__main__":
    main()
