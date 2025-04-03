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
import argparse

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


def save_mapping(id_mapping, mapping_file_path):
    """Save ID mapping to a JSON file.
    
    Args:
        id_mapping: Dictionary mapping memory IDs to their numeric IDs
        mapping_file_path: Full path to save the mapping file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)
        
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
        with open(mapping_file_path, 'w') as f:
            json.dump(serializable_mapping, f, indent=2)
        
        logger.info(f"Saved {len(serializable_mapping)} ID mappings to {mapping_file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving ID mapping: {str(e)}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Repair the Synthians Memory Core ID mapping file")
    parser.add_argument(
        "--storage-path", 
        type=str, 
        required=True,
        help="Path to the storage directory containing the 'stored' folder"
    )
    parser.add_argument(
        "--corpus", 
        type=str, 
        default="synthians",
        help="Corpus name (default: synthians)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    storage_path = args.storage_path
    corpus = args.corpus
    
    # Construct paths based on arguments
    base_path = os.path.join(storage_path, "stored", corpus)
    memory_dir = os.path.join(base_path, "memories")
    mapping_file_path = os.path.join(base_path, "faiss_index.bin.mapping.json")

    if not os.path.isdir(storage_path):
        logger.error(f"Storage path does not exist or is not a directory: {storage_path}")
        sys.exit(1)
        
    if not os.path.isdir(memory_dir):
        logger.error(f"Memory directory does not exist within storage path: {memory_dir}")
        logger.error(f"Ensure storage path '{storage_path}' contains 'stored/{corpus}/memories/' structure.")
        sys.exit(1)

    logger.info(f"Scanning memory files in: {memory_dir}")
    id_mapping = scan_memory_files(memory_dir)

    if id_mapping:
        logger.info(f"Rebuilt ID mapping with {len(id_mapping)} entries.")
        success = save_mapping(id_mapping, mapping_file_path)
        if success:
            logger.info("✅ Successfully repaired and saved the ID mapping.")
        else:
            logger.error("❌ Failed to save the repaired ID mapping.")
            sys.exit(1)
    else:
        logger.warning("⚠️ No memory files found to build mapping. Mapping file not created/updated.")
