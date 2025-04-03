# Run test_stability_fixes.py with proper imports
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and run the test
from utils.test_stability_fixes import test_main
import asyncio

if __name__ == "__main__":
    asyncio.run(test_main())