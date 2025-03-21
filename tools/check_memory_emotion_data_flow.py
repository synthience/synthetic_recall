#!/usr/bin/env python3
# check_memory_emotion_data_flow.py - Compare memory files with their original .npz files

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
import asyncio
import traceback
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('check_memory_emotion_data_flow.log')
    ]
)
logger = logging.getLogger('check_memory_emotion_flow')

def check_memories_missing_emotions() -> List[str]:
    """Check memory files and identify those without emotion data."""
    memory_storage_path = Path('memory/stored')
    memories_without_emotions = []
    total_memories = 0
    
    try:
        for memory_file in memory_storage_path.glob('*.json'):
            total_memories += 1
            try:
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                # Check if memory has emotion data
                metadata = memory_data.get('metadata', {})
                has_emotions = 'emotions' in metadata or 'dominant_emotion' in metadata
                
                if not has_emotions:
                    memories_without_emotions.append(str(memory_file))
                    logger.debug(f"Memory without emotions: {memory_file}")
            except Exception as e:
                logger.error(f"Error reading memory file {memory_file}: {e}")
        
        logger.info(f"Checked {total_memories} memory files")
        logger.info(f"Found {len(memories_without_emotions)} memories without emotion data")
        
        return memories_without_emotions
    except Exception as e:
        logger.error(f"Error checking memories: {e}")
        return []

def find_matching_npz_file(memory_file: str) -> Tuple[str, Dict[str, Any]]:
    """Find the matching .npz file for a memory file and extract its metadata."""
    memory_id = os.path.basename(memory_file).replace('.json', '')
    embedding_dir = Path('memory/indexed/embeddings')
    npz_metadata = {}
    found_file = None
    
    try:
        # First, check if there's a direct match based on ID
        potential_match = embedding_dir / f"{memory_id}.npz"
        if potential_match.exists():
            found_file = str(potential_match)
            npz_metadata = extract_npz_metadata(potential_match)
            return found_file, npz_metadata
        
        # Otherwise, look through all .npz files for content that might match
        for npz_file in embedding_dir.glob('**/*.npz'):
            try:
                data = np.load(npz_file, allow_pickle=True)
                if 'metadata' in data:
                    metadata_json = str(data['metadata'])
                    
                    # Handle potential binary string format
                    if metadata_json.startswith("b'") and metadata_json.endswith("'"):
                        metadata_json = metadata_json[2:-1].replace('\\', '\\\\')
                    
                    # Parse the metadata JSON
                    stored_metadata = json.loads(metadata_json)
                    
                    # Check if this .npz file has emotion data
                    if 'emotions' in stored_metadata or 'dominant_emotion' in stored_metadata:
                        with open(memory_file, 'r') as f:
                            memory_data = json.load(f)
                        
                        # Check if content matches between memory and npz
                        if 'text' in stored_metadata and memory_data.get('text') == stored_metadata['text']:
                            found_file = str(npz_file)
                            npz_metadata = stored_metadata
                            return found_file, npz_metadata
            except Exception as e:
                logger.debug(f"Error processing .npz file {npz_file}: {e}")
    
    except Exception as e:
        logger.error(f"Error finding matching .npz for {memory_file}: {e}")
    
    return found_file, npz_metadata

def extract_npz_metadata(npz_file: Path) -> Dict[str, Any]:
    """Extract metadata from an .npz file."""
    try:
        data = np.load(npz_file, allow_pickle=True)
        if 'metadata' in data:
            metadata_json = str(data['metadata'])
            
            # Handle potential binary string format
            if metadata_json.startswith("b'") and metadata_json.endswith("'"):
                metadata_json = metadata_json[2:-1].replace('\\', '\\\\')
            
            # Parse the metadata JSON
            return json.loads(metadata_json)
    except Exception as e:
        logger.error(f"Error extracting metadata from {npz_file}: {e}")
    
    return {}

def check_indexing_process():
    """Check the entire indexing process to find where emotion data might be lost."""
    # 1. Get memories without emotion data
    memories_without_emotions = check_memories_missing_emotions()
    
    # Sample up to 10 memories for detailed analysis
    sample_size = min(10, len(memories_without_emotions))
    sampled_memories = memories_without_emotions[:sample_size]
    
    logger.info(f"Analyzing {sample_size} memory files in detail")
    
    # 2. For each sampled memory, try to find its matching .npz file
    for memory_file in sampled_memories:
        logger.info(f"Analyzing memory file: {memory_file}")
        
        # Find matching .npz file
        matching_npz, npz_metadata = find_matching_npz_file(memory_file)
        
        if matching_npz:
            logger.info(f"Found matching .npz file: {matching_npz}")
            
            # Check if the .npz file has emotion data
            has_emotions = 'emotions' in npz_metadata or 'dominant_emotion' in npz_metadata
            
            if has_emotions:
                logger.info(f"The .npz file DOES have emotion data: {npz_metadata.get('emotions', {})}")
                logger.info(f"Dominant emotion: {npz_metadata.get('dominant_emotion', 'N/A')}")
                logger.error("IDENTIFIED ISSUE: Emotion data is present in .npz but not transferred to memory")
            else:
                logger.info("The .npz file does NOT have emotion data")
                logger.info("This may indicate the source problem is earlier in the pipeline")
        else:
            logger.warning(f"Could not find a matching .npz file for {memory_file}")

def main():
    logger.info("Starting memory emotion data flow check")
    check_indexing_process()
    logger.info("Check complete")

if __name__ == '__main__':
    main()
