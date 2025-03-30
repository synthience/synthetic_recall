#!/usr/bin/env python

'''
Test script to run the repair function and print the results.
'''

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path.cwd()))

# Import the memory core
from synthians_memory_core.synthians_memory_core import SynthiansMemoryCore

async def run_repair():
    core = SynthiansMemoryCore()
    
    # Run the repair function
    print("Running repair_index...")
    result = await core.repair_index('recreate_mapping')
    
    # Print the results
    print(f"\nRepair Results:")
    print(f"Success: {result['success']}")
    print(f"Is Consistent: {result['is_consistent']}")
    
    # Print the diagnostics
    print(f"\nDiagnostics Before:")
    for key, value in result['diagnostics_before'].items():
        print(f"  {key}: {value}")
    
    print(f"\nDiagnostics After:")
    for key, value in result['diagnostics_after'].items():
        print(f"  {key}: {value}")
    
    return result

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_repair())
