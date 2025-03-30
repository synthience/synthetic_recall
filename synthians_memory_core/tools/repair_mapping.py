#!/usr/bin/env python

"""
Repair ID mapping utility for Synthians Memory Core.

This script specifically fixes the ID mapping inconsistency where FAISS count > 0 but Mapping count = 0.
"""

import os
import sys
import json
import logging
import hashlib
import numpy as np
from pathlib import Path

# Add parent directory to path to allow importing modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("repair_mapping")


def scan_memory_files(memory_dir):
    """Scan all memory files in the directory to rebuild ID mapping.
    
    Args:
        memory_dir: Directory containing memory files
        
    Returns:
        dict: Dictionary mapping memory IDs to their numeric IDs
    """
    id_mapping = {}
    memory_ids = []
    
    # Find all memory files
    for root, _, files in os.walk(memory_dir):
        for file in files:
            if file.endswith('.json') and file.startswith('mem_'):
                memory_id = file.split('.')[0]  # Remove .json extension
                memory_ids.append(memory_id)
    
    logger.info(f"Found {len(memory_ids)} memory files")
    
    # Generate numeric IDs for all memory IDs
    for memory_id in memory_ids:
        numeric_id = int(hashlib.md5(memory_id.encode()).hexdigest(), 16) % (2**63-1)
        id_mapping[memory_id] = numeric_id
    
    return id_mapping


def save_mapping(id_mapping, storage_path):
    """Save ID mapping to a JSON file.
    
    Args:
        id_mapping: Dictionary mapping memory IDs to their numeric IDs
        storage_path: Path to save the mapping file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        mapping_path = os.path.join(storage_path, 'faiss_index.bin.mapping.json')
        
        # Create a serializable copy of the mapping
        serializable_mapping = {}
        for k, v in id_mapping.items():
            # Convert any non-string keys to strings for JSON serializability
            key = str(k)
            # Convert any special numeric types to standard Python types
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                value = int(v)
            else:
                value = v
            serializable_mapping[key] = value
        
        # Write the mapping to a file
        with open(mapping_path, 'w') as f:
            json.dump(serializable_mapping, f, indent=2)
        
        logger.info(f"Saved {len(serializable_mapping)} ID mappings to {mapping_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving ID mapping: {str(e)}")
        return False


def find_storage_path():
    """Find the storage path by looking for common directory structures."""
    possible_config_paths = [
        os.path.join(parent_dir, 'synthians_memory_core', 'config', 'core_config.json'),
        os.path.join(parent_dir, 'config', 'core_config.json'),
        os.path.join(parent_dir, 'core_config.json')
    ]
    
    # Try to find a config file
    for config_path in possible_config_paths:
        if os.path.exists(config_path):
            logger.info(f"Found config file at {config_path}")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                storage_path = config.get('storage_path')
                if storage_path:
                    return storage_path
            except Exception as e:
                logger.warning(f"Error reading config file {config_path}: {str(e)}")
    
    # If no config file found, try common storage paths
    possible_storage_paths = [
        os.path.join(parent_dir, 'storage'),
        os.path.join(parent_dir, 'synthians_memory_core', 'storage'),
        os.path.join(parent_dir, 'data', 'storage'),
    ]
    
    for path in possible_storage_paths:
        if os.path.exists(path):
            logger.info(f"Found storage directory at {path}")
            return path
    
    # Last resort: Just use a path in the current directory
    default_path = os.path.join(parent_dir, 'storage')
    os.makedirs(default_path, exist_ok=True)
    logger.warning(f"No storage path found, using default: {default_path}")
    return default_path


def main():
    """Main function to repair ID mapping."""
    try:
        # Find storage path without relying on config
        storage_path = find_storage_path()
        logger.info(f"Using storage path: {storage_path}")
        
        # Look for memories directory
        memories_path = os.path.join(storage_path, 'memories')
        if not os.path.exists(memories_path):
            # Try to find the memories directory
            for root, dirs, _ in os.walk(storage_path):
                for dir_name in dirs:
                    if dir_name == 'memories':
                        memories_path = os.path.join(root, dir_name)
                        break
                if os.path.exists(memories_path):
                    break
        
        if not os.path.exists(memories_path):
            logger.warning("Could not find 'memories' directory, creating one")
            os.makedirs(memories_path, exist_ok=True)
        
        logger.info(f"Using memories path: {memories_path}")
        
        # Scan memory files to rebuild ID mapping
        id_mapping = scan_memory_files(memories_path)
        
        # Save mapping
        success = save_mapping(id_mapping, storage_path)
        
        if success:
            logger.info(f"Successfully rebuilt ID mapping with {len(id_mapping)} entries")
            
            # Print instructions for restarting the server
            logger.info("""\n===========================================================\n
\
ID mapping has been repaired. Please restart the server to load the fixed mapping.\n\
If problems persist, run the following in Python:\n\
    from synthians_memory_core.synthians_memory_core import SynthiansMemoryCore\n\
    import asyncio\n\
    asyncio.run(SynthiansMemoryCore().repair_index('recreate_mapping'))\n\
===========================================================\n""")
        else:
            logger.error("Failed to rebuild ID mapping")
    
    except Exception as e:
        logger.error(f"Error during ID mapping repair: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
