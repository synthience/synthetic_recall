#!/usr/bin/env python3
# debug_metadata_serialization.py - Debug and fix issues with emotional metadata serialization

import os
import sys
import json
import time
import uuid
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
        logging.FileHandler('debug_metadata_serialization.log')
    ]
)
logger = logging.getLogger('debug_metadata')

# Import memory system after path setup
from server.memory_system import MemorySystem
from tools.index_embeddings import EmbeddingIndexer

async def test_metadata_serialization():
    """Test metadata serialization with emotion data"""
    # Sample metadata with emotions
    metadata = {
        'text': "This is a test memory with emotional data.",
        'source': 'debug_metadata_serialization.py',
        'timestamp': time.time(),
        'role': 'system',
        'emotions': {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.05,
            'fear': 0.03,
            'surprise': 0.02
        },
        'dominant_emotion': 'joy'
    }
    
    # Add potentially problematic data types
    metadata['problem_set'] = set(['item1', 'item2'])  # Set is not JSON serializable
    metadata['problem_tuple'] = (1, 2, 3)  # Tuple is not JSON serializable
    metadata['problem_complex'] = complex(1, 2)  # Complex number is not JSON serializable
    
    # Try to serialize
    try:
        json_str = json.dumps(metadata)
        logger.info("INITIAL SERIALIZATION FAILED AS EXPECTED - This should not print")
    except TypeError as e:
        logger.info(f"EXPECTED FAILURE: {e}")
        
        # Fix serialization issues
        fixed_metadata = metadata.copy()
        for key, value in fixed_metadata.items():
            if isinstance(value, (set, tuple)):
                fixed_metadata[key] = list(value)
            elif isinstance(value, complex):
                fixed_metadata[key] = str(value)
                
        # Try again
        try:
            json_str = json.dumps(fixed_metadata)
            logger.info("FIXED SERIALIZATION SUCCEEDED")
            logger.debug(f"Serialized JSON: {json_str[:200]}...")
            
            # Check if emotion data survived
            fixed_json = json.loads(json_str)
            if 'emotions' in fixed_json and 'dominant_emotion' in fixed_json:
                logger.info(f"SUCCESS: Emotion data survived serialization")
                logger.info(f"Emotions: {fixed_json['emotions']}")
                logger.info(f"Dominant emotion: {fixed_json['dominant_emotion']}")
            else:
                logger.error(f"FAILURE: Emotion data was lost during serialization")
                logger.debug(f"Fixed JSON: {fixed_json}")
        except Exception as e:
            logger.error(f"UNEXPECTED ERROR during fixed serialization: {e}")
    
    return True

async def inspect_memory_system_save():
    """Replicate the memory saving process to debug serialization issues"""
    # Create a test memory system
    memory_config = {
        'storage_path': 'memory/debug_output',
        'embedding_dim': 384
    }
    memory_system = MemorySystem(config=memory_config)
    
    # Sample embedding (384 dimensions)
    embedding = np.random.rand(384).astype(np.float32)
    embedding_tensor = torch.tensor(embedding)
    
    # Sample text
    text = "This is a test memory with emotional data. I feel very happy about this test."
    
    # Sample metadata with emotions
    metadata = {
        'source': 'debug_metadata_serialization.py',
        'timestamp': time.time(),
        'role': 'system',
        'emotions': {
            'joy': 0.8,
            'sadness': 0.1,
            'anger': 0.05,
            'fear': 0.03,
            'surprise': 0.02
        },
        'dominant_emotion': 'joy'
    }
    
    # Add the memory
    logger.info("Adding memory with emotion data...")
    memory = await memory_system.add_memory(
        text=text,
        embedding=embedding_tensor,
        quickrecal_score=0.8,
        metadata=metadata
    )
    
    # Check the saved memory file
    memory_id = memory.get('id')
    memory_path = Path(memory_system.storage_path) / f"{memory_id}.json"
    
    if memory_path.exists():
        logger.info(f"Memory saved successfully: {memory_path}")
        
        # Load and inspect the saved file
        with open(memory_path, 'r') as f:
            saved_memory = json.load(f)
        
        saved_metadata = saved_memory.get('metadata', {})
        
        if 'emotions' in saved_metadata and 'dominant_emotion' in saved_metadata:
            logger.info(f"SUCCESS: Emotion data was properly saved")
            logger.info(f"Emotions: {saved_metadata['emotions']}")
            logger.info(f"Dominant emotion: {saved_metadata['dominant_emotion']}")
            return True
        else:
            logger.error(f"FAILURE: Emotion data was lost during save")
            logger.debug(f"Saved metadata: {saved_metadata}")
            return False
    else:
        logger.error(f"Memory file not created: {memory_path}")
        return False

async def fix_broken_memory_files():
    """Fix existing memory files that are missing emotion data"""
    # Load NPZ files with emotion data
    npz_dir = Path('memory/npz_files')
    if not npz_dir.exists():
        logger.error(f"NPZ directory not found: {npz_dir}")
        return False
    
    # Load existing memory files
    memory_dir = Path('memory/stored')
    if not memory_dir.exists():
        logger.error(f"Memory directory not found: {memory_dir}")
        return False
    
    npz_files = list(npz_dir.glob('*.npz'))
    memory_files = list(memory_dir.glob('*.json'))
    
    logger.info(f"Found {len(npz_files)} NPZ files and {len(memory_files)} memory files")
    
    # Extract emotion data from NPZ files
    emotion_data = {}
    for npz_file in npz_files:
        try:
            npz_data = np.load(npz_file)
            if 'metadata' in npz_data:
                metadata_json = npz_data['metadata'].item()
                
                # Handle potential binary string format
                if isinstance(metadata_json, bytes):
                    metadata_json = metadata_json.decode('utf-8')
                if isinstance(metadata_json, str) and metadata_json.startswith("b'") and metadata_json.endswith("'"):
                    metadata_json = metadata_json[2:-1].replace('\\', '\\')  # Fix here
                
                # Parse the metadata JSON
                try:
                    stored_metadata = json.loads(metadata_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing metadata from {npz_file.name}: {e}")
                    continue
                
                # Extract text to use as key
                text = stored_metadata.get('text', '')
                if text and ('emotions' in stored_metadata or 'dominant_emotion' in stored_metadata):
                    # Store emotion data keyed by text
                    emotion_data[text] = {
                        'emotions': stored_metadata.get('emotions', {}),
                        'dominant_emotion': stored_metadata.get('dominant_emotion', None)
                    }
                    logger.info(f"Found emotion data for: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Error loading NPZ file {npz_file.name}: {e}")
    
    logger.info(f"Extracted emotion data for {len(emotion_data)} text entries")
    
    # Fix memory files
    fixed_count = 0
    for memory_file in memory_files:
        try:
            with open(memory_file, 'r') as f:
                memory_data = json.load(f)
            
            # Check if already has emotion data
            metadata = memory_data.get('metadata', {})
            if 'emotions' in metadata and 'dominant_emotion' in metadata:
                logger.debug(f"Memory {memory_file.name} already has emotion data")
                continue
            
            # Look for matching text
            text = memory_data.get('text', '')
            if text in emotion_data:
                # Add emotion data
                if 'metadata' not in memory_data:
                    memory_data['metadata'] = {}
                
                memory_data['metadata']['emotions'] = emotion_data[text]['emotions']
                memory_data['metadata']['dominant_emotion'] = emotion_data[text]['dominant_emotion']
                
                # Save updated file
                with open(memory_file, 'w') as f:
                    json.dump(memory_data, f)
                
                fixed_count += 1
                logger.info(f"Fixed memory file: {memory_file.name}")
            elif any(text in stored_text for stored_text in emotion_data.keys()):
                # Partial match
                matching_key = next((k for k in emotion_data.keys() if text in k or k in text), None)
                if matching_key:
                    # Add emotion data from partial match
                    if 'metadata' not in memory_data:
                        memory_data['metadata'] = {}
                    
                    memory_data['metadata']['emotions'] = emotion_data[matching_key]['emotions']
                    memory_data['metadata']['dominant_emotion'] = emotion_data[matching_key]['dominant_emotion']
                    memory_data['metadata']['emotion_source'] = 'partial_match'
                    
                    # Save updated file
                    with open(memory_file, 'w') as f:
                        json.dump(memory_data, f)
                    
                    fixed_count += 1
                    logger.info(f"Fixed memory file with partial match: {memory_file.name}")
        except Exception as e:
            logger.warning(f"Error processing memory file {memory_file.name}: {e}")
    
    logger.info(f"Fixed {fixed_count} memory files")
    return fixed_count > 0

async def patch_memory_system():
    """Create a patch for the memory system to properly handle emotion data"""
    # Read the original file
    memory_system_path = Path('server/memory_system.py')
    
    if not memory_system_path.exists():
        logger.error(f"Memory system file not found: {memory_system_path}")
        return False
    
    with open(memory_system_path, 'r') as f:
        memory_system_code = f.read()
    
    # Create backup
    backup_path = memory_system_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(memory_system_code)
    
    logger.info(f"Created backup of memory system: {backup_path}")
    
    # Analyze and fix the _save_memory method
    # Look for patterns that might cause emotion data loss
    if 'def _save_memory(self' in memory_system_code:
        # Find the method and check its handling of metadata
        start_idx = memory_system_code.find('def _save_memory(self')
        end_idx = memory_system_code.find('def ', start_idx + 1)
        if end_idx == -1:
            end_idx = len(memory_system_code)
        
        method_code = memory_system_code[start_idx:end_idx]
        logger.debug(f"_save_memory method code:\n{method_code}")
        
        # Check if the method properly handles JSON serialization
        if 'try:' in method_code and 'json.dumps(memory_copy)' in method_code:
            logger.info("Memory system already has JSON serialization checks")
            
            # Ensure the serialization correction is comprehensive
            if 'isinstance(value, (set, tuple))' in method_code:
                logger.info("Serialization correction already handles sets and tuples")
            else:
                logger.info("Expanding serialization correction to handle more types")
                
                # Add more type corrections
                patched_code = method_code.replace(
                    "if isinstance(value, (set, tuple)):",
                    "if isinstance(value, (set, tuple, complex)):"
                )
                memory_system_code = memory_system_code.replace(method_code, patched_code)
        else:
            logger.info("Adding JSON serialization checks")
            
            # Add serialization check before writing to file
            idx = method_code.find('with open(file_path, \'w\') as f:')
            if idx != -1:
                new_code = (
                    "            # Ensure metadata is JSON serializable\n"
                    "            try:\n"
                    "                # Test JSON serialization before saving\n"
                    "                json.dumps(memory_copy)\n"
                    "            except TypeError as e:\n"
                    "                logger.error(f\"Memory {memory_id} contains non-JSON serializable data: {e}\")\n"
                    "                # Try to fix the metadata by converting problematic types\n"
                    "                if 'metadata' in memory_copy:\n"
                    "                    for key, value in memory_copy['metadata'].items():\n"
                    "                        if isinstance(value, (set, tuple, complex)):\n"
                    "                            memory_copy['metadata'][key] = list(value) if isinstance(value, (set, tuple)) else str(value)\n"
                    "                        elif hasattr(value, '__dict__'):\n"
                    "                            memory_copy['metadata'][key] = str(value)\n"
                    "                logger.info(f\"Attempted to fix non-serializable data in memory {memory_id}\")\n\n"
                )
                
                patched_method = method_code[:idx] + new_code + method_code[idx:]
                memory_system_code = memory_system_code.replace(method_code, patched_method)
    
    # Write the patched file
    patched_path = memory_system_path.with_suffix('.py.patched')
    with open(patched_path, 'w') as f:
        f.write(memory_system_code)
    
    logger.info(f"Created patched memory system: {patched_path}")
    return True

async def main():
    logger.info("Starting metadata serialization debugging")
    
    # Test 1: Metadata serialization
    logger.info("\n=== TEST 1: METADATA SERIALIZATION ===\n")
    await test_metadata_serialization()
    
    # Test 2: Memory system save
    logger.info("\n=== TEST 2: MEMORY SYSTEM SAVE ===\n")
    await inspect_memory_system_save()
    
    # Patch memory system
    logger.info("\n=== PATCHING MEMORY SYSTEM ===\n")
    await patch_memory_system()
    
    # Fix broken memory files - make this non-interactive for automated testing
    logger.info("\n=== FIXING BROKEN MEMORY FILES ===\n")
    logger.info("Running in automated mode, skipping user prompt")
    # Skip the user prompt
    await fix_broken_memory_files()
    
    logger.info("\nDebugging complete\n")

if __name__ == '__main__':
    import torch  # Import here to avoid circular imports
    asyncio.run(main())
