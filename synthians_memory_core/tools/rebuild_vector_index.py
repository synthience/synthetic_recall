#!/usr/bin/env python

"""
Rebuild Vector Index Tool

This script completely rebuilds the FAISS vector index from scratch by:
1. Loading all existing memory entries from persistence
2. Creating a fresh vector index
3. Adding valid embeddings to the index
4. Saving the rebuilt index

Use this when the vector index is corrupted or shows inconsistencies that
cannot be resolved with simpler repair methods.

Usage:
    python -m synthians_memory_core.tools.rebuild_vector_index --storage-path /path/to/storage
"""

import os
import sys
import argparse
import json
import shutil
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import torch

# Add parent directory to path so we can import memory core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from synthians_memory_core.memory_persistence import MemoryPersistence
from synthians_memory_core.vector_index import MemoryVectorIndex
from synthians_memory_core.memory_structures import MemoryEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("rebuild_vector_index")

def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild vector index from memory persistence")
    parser.add_argument(
        "--storage-path", 
        type=str, 
        required=True,
        help="Path to the storage directory (should contain 'memories' folder)"
    )
    parser.add_argument(
        "--corpus", 
        type=str, 
        default="synthians",
        help="Corpus name (default: synthians)"
    )
    parser.add_argument(
        "--embedding-dim", 
        type=int, 
        default=768,
        help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--backup", 
        action="store_true",
        help="Create backup of existing index files before rebuilding"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--test-storage", 
        action="store_true",
        help="Use default test storage path: ./test_storage"
    )
    return parser.parse_args()

def backup_index_files(storage_path: str, corpus: str) -> bool:
    """Backup existing index files if they exist."""
    try:
        base_path = os.path.join(storage_path, "stored", corpus)
        index_file = os.path.join(base_path, "faiss_index.bin")
        mapping_file = os.path.join(base_path, "faiss_index.bin.mapping.json")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if files exist
        index_exists = os.path.exists(index_file)
        mapping_exists = os.path.exists(mapping_file)
        
        if not index_exists and not mapping_exists:
            log.info("No existing index files found to backup.")
            return True
        
        # Create backup directory
        backup_dir = os.path.join(base_path, f"index_backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup index file if it exists
        if index_exists:
            shutil.copy2(index_file, os.path.join(backup_dir, "faiss_index.bin"))
            log.info(f"Backed up index file to {backup_dir}/faiss_index.bin")
        
        # Backup mapping file if it exists
        if mapping_exists:
            shutil.copy2(mapping_file, os.path.join(backup_dir, "faiss_index.bin.mapping.json"))
            log.info(f"Backed up mapping file to {backup_dir}/faiss_index.bin.mapping.json")
        
        return True
    except Exception as e:
        log.error(f"Error backing up index files: {str(e)}")
        return False

def delete_existing_index(storage_path: str, corpus: str) -> bool:
    """Delete existing index files to start fresh."""
    try:
        base_path = os.path.join(storage_path, "stored", corpus)
        index_file = os.path.join(base_path, "faiss_index.bin")
        mapping_file = os.path.join(base_path, "faiss_index.bin.mapping.json")
        
        # Delete index file if it exists
        if os.path.exists(index_file):
            os.remove(index_file)
            log.info(f"Deleted existing index file: {index_file}")
        
        # Delete mapping file if it exists
        if os.path.exists(mapping_file):
            os.remove(mapping_file)
            log.info(f"Deleted existing mapping file: {mapping_file}")
        
        return True
    except Exception as e:
        log.error(f"Error deleting existing index files: {str(e)}")
        return False

def validate_embedding(embedding, expected_dim):
    """Validate that an embedding is well-formed and handle dimension mismatches.
    
    Args:
        embedding: The embedding to validate (numpy array, list, or tensor)
        expected_dim: The expected embedding dimension
        
    Returns:
        The validated (and potentially normalized) embedding, or None if invalid
    """
    if embedding is None:
        log.warning("Embedding is None, cannot validate")
        return None
    
    # Convert to numpy array for consistent handling
    try:
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        elif isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            log.warning("Embedding contains NaN or Inf values. Replacing with zeros.")
            embedding = np.zeros(expected_dim, dtype=np.float32)  # Use zeros instead of skipping
            return embedding
            
        # Handle dimension mismatch
        actual_dim = embedding.shape[0] if len(embedding.shape) > 0 else 0
        
        if actual_dim != expected_dim:
            # If embedding is larger than expected, truncate
            if actual_dim > expected_dim:
                log.warning(f"Truncating embedding from dimension {actual_dim} to {expected_dim}")
                embedding = embedding[:expected_dim]
            # If embedding is smaller than expected, pad with zeros
            elif actual_dim < expected_dim:
                log.warning(f"Padding embedding from dimension {actual_dim} to {expected_dim}")
                pad_size = expected_dim - actual_dim
                embedding = np.pad(embedding, (0, pad_size), 'constant', constant_values=0)
        
        # Normalize the embedding to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            log.warning("Embedding has zero norm. Using zero vector.")
            embedding = np.zeros(expected_dim, dtype=np.float32)
        
        return embedding
    except Exception as e:
        log.error(f"Error validating embedding: {str(e)}")
        return None

async def rebuild_index_async(storage_path: str, corpus: str, embedding_dim: int, geometry_manager, verbose: bool = False) -> Tuple[int, int]:
    """Async version of rebuild_index to properly handle coroutines."""
    try:
        log.info("Rebuilding vector index...")
        
        # Detect Docker environment
        docker_path = "/app/memory"
        
        if storage_path == docker_path:
            log.info(f"Detected Docker environment with storage path: {storage_path}")
            # In Docker, the correct path is /app/memory/stored
            persist_path = os.path.join(storage_path, "stored")
            log.info(f"Using persistence path: {persist_path}")
            
            # Initialize with Docker-specific paths
            persistence = MemoryPersistence({
                "storage_path": persist_path, 
                "corpus": corpus
            })
            
            vector_index = MemoryVectorIndex({
                'embedding_dim': embedding_dim,
                'storage_path': persist_path,
                'corpus': corpus,
                'index_type': 'Cosine',
                'use_gpu': False
            })
        else:
            # Default path detection logic for non-Docker environments
            stored_dir = os.path.join(storage_path, "stored", corpus)
            if not os.path.exists(stored_dir):
                stored_dir = os.path.join(storage_path, corpus)
                if not os.path.exists(stored_dir):
                    log.error(f"Neither stored/{corpus} nor {corpus} directory exists under {storage_path}")
                    return (0, 0)
            
            # Initialize persistence with the correct path structure
            if "stored" in stored_dir:
                # Path includes 'stored' directory, so use the parent
                persistence = MemoryPersistence({"storage_path": os.path.dirname(stored_dir), "corpus": corpus})
            else:
                # Path doesn't include 'stored', use directly
                persistence = MemoryPersistence({"storage_path": storage_path, "corpus": corpus})
            
            # Initialize vector index
            vector_index = MemoryVectorIndex({
                'embedding_dim': embedding_dim,
                'storage_path': os.path.dirname(stored_dir) if "stored" in stored_dir else storage_path,
                'corpus': corpus,
                'index_type': 'Cosine',
                'use_gpu': False
            })
        
        # Load memory index to get all memory IDs
        memory_index = persistence.memory_index
        memory_ids = []

        # Additional logging for debugging
        if storage_path == docker_path:
            # In Docker, manually find memory files as a fallback
            memory_dir = os.path.join(persist_path, corpus)
            log.info(f"Checking for memory files in: {memory_dir}")
            
            if os.path.exists(memory_dir):
                # Find all memory files (*.json files that might be memories) in the directory
                memory_files = [f for f in os.listdir(memory_dir) if f.endswith('.json')]
                log.info(f"Found {len(memory_files)} memory files in {memory_dir}")
                
                if not memory_ids and memory_files:
                    # Extract memory IDs from filenames as fallback, handling both mem_ID.json and ID.json formats
                    memory_ids = []
                    for f in memory_files:
                        if f.startswith('mem_') and f.endswith('.json'):
                            memory_ids.append(f[4:-5])  # Extract ID from mem_ID.json
                        elif f.endswith('.json') and not f.startswith('memory_index') and not f.startswith('assembly'):
                            # Avoid non-memory files like memory_index.json or assembly files
                            file_id = f[:-5]  # Extract ID from ID.json
                            if len(file_id) >= 8:  # Basic check for reasonable ID length
                                memory_ids.append(file_id)
                    
                    log.info(f"Using {len(memory_ids)} memory IDs from filenames")
            else:
                log.error(f"Memory directory {memory_dir} does not exist")
        
        # Get memory IDs from index if available, otherwise use our fallback
        if not memory_ids:
            try:
                memory_ids = memory_index.get_all_ids()
                log.info(f"Retrieved {len(memory_ids)} memory IDs from memory index")
            except Exception as e:
                log.error(f"Error retrieving memory IDs from index: {e}")
                memory_ids = []

        # If still no memory IDs, check if files exist manually
        if not memory_ids:
            log.warning("No memory IDs found in index, checking for files manually")
            memory_ids = find_memory_files(storage_path, corpus)
            log.info(f"Found {len(memory_ids)} memory files by scanning directory")
        
        total_memories = len(memory_ids)
        log.info(f"Found {total_memories} memories in persistence")
        
        # Track statistics
        added = 0
        skipped = 0
        errors = 0
        
        # Process each memory
        for memory_id in memory_ids:
            try:
                # Show progress periodically
                processed = added + skipped + errors + 1
                if verbose or processed % 100 == 0 or processed == total_memories:
                    log.info(f"Processing memory {processed}/{total_memories} (ID: {memory_id})")
                
                # Get the memory entry
                memory_entry = None
                try:
                    # Properly await the coroutine
                    memory_entry = await persistence.load_memory(memory_id)
                except Exception as load_error:
                    # If standard loading fails, try direct file loading
                    if storage_path == docker_path:
                        # Try loading directly from file as a fallback
                        try:
                            # Try different file naming patterns and locations
                            memory_file_candidates = [
                                os.path.join(persist_path, corpus, f"{memory_id}.json"),
                                os.path.join(persist_path, corpus, f"mem_{memory_id}.json"),
                                os.path.join(persist_path, f"{memory_id}.json"),
                                os.path.join(persist_path, f"mem_{memory_id}.json")
                            ]
                            
                            memory_file = None
                            for candidate in memory_file_candidates:
                                if os.path.exists(candidate):
                                    memory_file = candidate
                                    break
                            
                            if memory_file:
                                log.info(f"Trying direct file load from: {memory_file}")
                                with open(memory_file, 'r') as f:
                                    import json
                                    memory_data = json.load(f)
                                    # Create memory entry from data
                                    # Try different import paths to handle potential module location changes
                                    try:
                                        from synthians_memory_core.memory_structures import MemoryEntry
                                    except ImportError:
                                        try:
                                            from synthians_memory_core.memory_entry import MemoryEntry
                                        except ImportError:
                                            log.error("Unable to import MemoryEntry from either module")
                                            raise
                                    
                                    # Ensure the ID is set correctly
                                    if 'id' not in memory_data:
                                        memory_data['id'] = memory_id
                                        
                                    memory_entry = MemoryEntry.from_dict(memory_data) if hasattr(MemoryEntry, 'from_dict') else MemoryEntry(**memory_data)
                                    log.info(f"Successfully loaded memory directly from file: {memory_id}")
                            else:
                                log.warning(f"Memory file not found in any of the candidate locations")
                        except Exception as direct_load_error:
                            log.error(f"Error with direct file loading: {direct_load_error}")
                            raise load_error
                    else:
                        raise load_error
                
                if not memory_entry:
                    log.warning(f"Failed to load memory with ID: {memory_id}, skipping")
                    skipped += 1
                    continue
                
                # Validate embedding
                embedding = memory_entry.embedding
                validated_embedding = validate_embedding(embedding, embedding_dim)
                if validated_embedding is None:
                    log.warning(f"Memory {memory_id} has invalid embedding. Skipping.")
                    skipped += 1
                    continue

                # Add to vector index
                vector_index.add(memory_id, validated_embedding)
                added += 1
                
                if verbose:
                    log.info(f"Added memory {memory_id} to vector index")
            except Exception as e:
                log.error(f"Error processing memory {memory_id}: {str(e)}")
                errors += 1
        
        # Save index
        try:
            log.info("Saving vector index...")
            vector_index.save()
            log.info("Vector index saved successfully")
        except Exception as e:
            log.error(f"Error saving vector index: {str(e)}")
        
        log.info("Rebuild complete:")
        log.info(f"  - Total memories found: {total_memories}")
        log.info(f"  - Memories added to index: {added}")
        log.info(f"  - Memories skipped: {skipped}")
        log.info(f"  - Errors encountered: {errors}")
        
        # Verify index
        index_count = vector_index.count()
        mapping_count = len(vector_index.id_to_index)
        log.info(f"  - FAISS index count: {index_count}")
        log.info(f"  - ID mapping count: {mapping_count}")
        
        return (added, skipped)
    except Exception as e:
        log.error(f"Error rebuilding vector index: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        return (0, 0)

def rebuild_index(storage_path: str, corpus: str, embedding_dim: int, geometry_manager, verbose: bool = False) -> Tuple[int, int]:
    """Rebuild the vector index from memory persistence. This is a wrapper for the async version."""
    return asyncio.run(rebuild_index_async(storage_path, corpus, embedding_dim, geometry_manager, verbose))

def check_index_integrity(storage_path: str, corpus: str) -> bool:
    """Check if the vector index is in a consistent state."""
    try:
        # For Docker environment, use the right path structure
        docker_path = "/app/memory"
        if storage_path == docker_path:
            log.info(f"Detected Docker environment for integrity check with storage path: {storage_path}")
            # In Docker, the correct path is /app/memory/stored/synthians
            persist_path = os.path.join(storage_path, "stored")
            log.info(f"Using persistence path for integrity check: {persist_path}")
            
            # Initialize vector index with Docker-specific paths
            vector_index = MemoryVectorIndex({
                'embedding_dim': 768,  # Default dimension, not critical for checking
                'storage_path': persist_path,
                'corpus': corpus
            })
        else:
            # Default path detection logic for non-Docker environments
            stored_dir = os.path.join(storage_path, "stored", corpus)
            if not os.path.exists(stored_dir):
                stored_dir = os.path.join(storage_path, corpus)
                if not os.path.exists(stored_dir):
                    log.warning(f"Neither stored/{corpus} nor {corpus} directory exists under {storage_path}")
                    # Continue anyway since we're just checking existing index
            
            # Use consistent path structure
            vector_path = os.path.dirname(stored_dir) if "stored" in stored_dir and os.path.exists(stored_dir) else storage_path
            
            # Initialize vector index
            vector_index = MemoryVectorIndex({
                'embedding_dim': 768,  # Default dimension, not critical for checking
                'storage_path': vector_path,
                'corpus': corpus
            })
        
        # Get counts
        index_count = vector_index.count()
        mapping_count = len(vector_index.id_to_index)
        
        log.info(f"Index integrity check:")
        log.info(f"  - FAISS index count: {index_count}")
        log.info(f"  - ID mapping count: {mapping_count}")
        
        # Check if counts match
        if index_count != mapping_count:
            log.warning(f"Vector index inconsistency detected! FAISS count: {index_count}, Mapping count: {mapping_count}")
            return False
        
        log.info("Vector index is consistent!")
        return True
    
    except Exception as e:
        log.error(f"Error checking index integrity: {str(e)}")
        return False

def find_memory_files(storage_path: str, corpus: str) -> List[str]:
    """Find memory files in the directory with comprehensive path handling."""
    memory_ids = []
    
    # Check for Docker environment
    is_docker = storage_path == "/app/memory"
    
    if is_docker:
        # For Docker, check both direct and stored paths
        potential_dirs = [
            os.path.join(storage_path, "stored", corpus),
            os.path.join(storage_path, corpus),
            os.path.join(storage_path, "stored")
        ]
    else:
        # For non-Docker, check standard paths
        potential_dirs = [
            os.path.join(storage_path, "stored", corpus),
            os.path.join(storage_path, corpus)
        ]
    
    # Find memory files in all potential directories
    for memory_dir in potential_dirs:
        if os.path.exists(memory_dir):
            log.info(f"Scanning for memory files in {memory_dir}")
            
            try:
                # Find all files ending with .json that could be memories
                json_files = [f for f in os.listdir(memory_dir) if f.endswith('.json')]
                
                for f in json_files:
                    if f.startswith('mem_') and f.endswith('.json'):
                        # Extract ID from mem_ID.json format
                        memory_ids.append(f[4:-5])
                    elif not f.startswith('memory_index') and not f.startswith('assembly') and len(f) > 12:
                        # Make sure it's not a system file and has a reasonable ID length
                        memory_ids.append(f[:-5])  # Extract ID from ID.json format
                
                log.info(f"Found {len(memory_ids)} potential memory files in {memory_dir}")
                
                # Remove duplicates that might have been found in multiple directories
                memory_ids = list(set(memory_ids))
                
                if memory_ids:
                    # If we found memory files, return them without checking other directories
                    break
            except Exception as e:
                log.error(f"Error scanning directory {memory_dir}: {str(e)}")
    
    return memory_ids

def main():
    """Main function to rebuild the vector index."""
    args = parse_args()
    
    if args.verbose:
        log.setLevel(logging.DEBUG)
    
    log.info(f"Starting vector index rebuild")
    
    # Use test storage path if requested
    if args.test_storage:
        args.storage_path = os.path.join(os.getcwd(), "test_storage")
        log.info(f"Using test storage path: {args.storage_path}")
    
    log.info(f"Storage path: {args.storage_path}")
    log.info(f"Corpus: {args.corpus}")
    log.info(f"Embedding dimension: {args.embedding_dim}")
    
    # Check if storage path exists
    if not os.path.exists(args.storage_path):
        log.error(f"Storage path does not exist: {args.storage_path}")
        return 1
    
    # For Docker environment, use the right path structure
    docker_path = "/app/memory"
    if args.storage_path == docker_path:
        log.info(f"Detected Docker environment. Using specific Docker paths.")
        # No need to check for stored/synthians directory - we know it exists
    else:
        # For non-Docker environments, check if stored directory exists
        stored_dir = os.path.join(args.storage_path, "stored", args.corpus)
        if not os.path.exists(stored_dir):
            # Try alternate path directly under storage_path
            stored_dir = os.path.join(args.storage_path, args.corpus)
            if not os.path.exists(stored_dir):
                log.error(f"Neither stored/{args.corpus} nor {args.corpus} directory exists under {args.storage_path}")
                return 1
            log.info(f"Using directory structure: {stored_dir}")
    
    # Initial integrity check
    log.info("Performing initial integrity check...")
    initial_integrity = check_index_integrity(args.storage_path, args.corpus)
    
    # Backup existing index files if requested
    if args.backup:
        log.info("Backing up existing index files...")
        if not backup_index_files(args.storage_path, args.corpus):
            log.error("Failed to backup index files. Aborting.")
            return 1
    
    # Delete existing index files
    log.info("Deleting existing index files...")
    if not delete_existing_index(args.storage_path, args.corpus):
        log.error("Failed to delete existing index files. Aborting.")
        return 1
    
    # Rebuild the index
    log.info("Rebuilding vector index...")
    added, skipped = rebuild_index(args.storage_path, args.corpus, args.embedding_dim, geometry_manager=None, verbose=args.verbose)
    
    # Final integrity check
    log.info("Performing final integrity check...")
    final_integrity = check_index_integrity(args.storage_path, args.corpus)
    
    if not final_integrity:
        log.error("Final integrity check failed. The rebuild may not have been completely successful.")
        return 1
    
    log.info(f"Vector index rebuild completed successfully!")
    log.info(f"  - Added {added} memories to index")
    log.info(f"  - Skipped {skipped} memories due to validation failures")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
