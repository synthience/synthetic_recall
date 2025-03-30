#!/usr/bin/env python

"""
Repair index utility for Synthians Memory Core.

This script repairs the vector index by recreating ID mappings from persistent storage.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to allow importing modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from synthians_memory_core.synthians_memory_core import SynthiansMemoryCore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("repair_index")


async def repair_index(repair_type: str = "recreate_mapping", config_path: str = None):
    """Repair the vector index.
    
    Args:
        repair_type: Type of repair to perform (recreate_mapping, rebuild)
        config_path: Path to custom config file
    """
    # Use default config
    config = None
    
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded custom config from {config_path}")
    
    # Initialize memory core
    logger.info(f"Initializing SynthiansMemoryCore with repair_type={repair_type}")
    memory_core = SynthiansMemoryCore(config)
    
    # Run repair
    logger.info(f"Running repair of type '{repair_type}'...")
    success = await memory_core.repair_index(repair_type)
    
    if success:
        logger.info(f"✅ Repair of type '{repair_type}' completed successfully")
        # Verify the index integrity
        is_consistent, diagnostics = memory_core.vector_index.verify_index_integrity()
        logger.info(f"Index consistency after repair: {is_consistent}")
        logger.info(f"Index diagnostics: {diagnostics}")
    else:
        logger.error(f"❌ Repair of type '{repair_type}' failed")
    
    return success


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Repair the Synthians Memory Core vector index")
    parser.add_argument(
        "--repair-type", 
        type=str, 
        default="auto", 
        choices=["auto", "recreate_mapping", "rebuild"],
        help="Type of repair to perform"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to custom config file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(repair_index(args.repair_type, args.config))
