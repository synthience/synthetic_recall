#!/usr/bin/env python

"""
Direct utility script to rebuild the vector index without going through the API.
This is useful for fixing vector index inconsistencies directly, especially when the API is failing.
"""

import os
import sys
import asyncio
import traceback
import numpy as np

# Configure basic logging to console
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("VectorIndexRepair")

# Adjust path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Loading necessary modules...")

try:
    from synthians_memory_core.vector_index import MemoryVectorIndex
    from synthians_memory_core.memory_persistence import MemoryPersistence
    from synthians_memory_core.memory_structures import MemoryEntry, MemoryAssembly
    print("Successfully imported required modules")
except Exception as e:
    print(f"Failed to import modules: {e}")
    traceback.print_exc()
    sys.exit(1)

async def rebuild_vector_index_directly():
    """
    Directly rebuild the vector index using core components instead of going through SynthiansMemoryCore.
    This eliminates dependencies on other components that might be failing.
    """
    try:
        # Configuration
        storage_path = "/app/memory/stored/synthians"
        # Detect Windows environment and adjust path
        if os.name == 'nt' and not os.path.exists(storage_path):
            print("Running on Windows, using relative storage path")
            storage_path = os.path.join(os.getcwd(), "memory_storage")
            if not os.path.exists(storage_path):
                os.makedirs(storage_path, exist_ok=True)
        
        embedding_dim = 768
        index_type = "Cosine"
        
        print(f"Using storage path: {storage_path}")
        
        # 1. Initialize persistence to load all memory items
        print("Initializing MemoryPersistence...")
        persistence = MemoryPersistence({'storage_path': storage_path})
        await persistence.initialize()
        print(f"Found {len(persistence.memory_index)} memory entries in index")
        
        # 2. Create a new vector index
        print("Creating new MemoryVectorIndex...")
        new_index = MemoryVectorIndex({
            'embedding_dim': embedding_dim,
            'storage_path': storage_path,
            'index_type': index_type,
            'use_gpu': False
        })
        print("Initializing new index...")
        await new_index.initialize(True)  # force_new=True to create a fresh index
        
        # 3. Load all memory items and assemblies
        print("Loading all memory items from persistence...")
        items = await persistence.load_all()
        memory_entries = [item for item in items if isinstance(item, MemoryEntry)]
        assemblies = [item for item in items if isinstance(item, MemoryAssembly)]
        print(f"Loaded {len(memory_entries)} memory entries and {len(assemblies)} assemblies")
        
        # 4. Add each memory embedding to the new index
        print("Adding memory entries to new index...")
        added_memories = 0
        added_assemblies = 0
        failed_items = 0
        
        # Process memory entries first
        for memory in memory_entries:
            try:
                if not hasattr(memory, 'id') or not hasattr(memory, 'embedding'):
                    print(f"Skipping invalid memory entry without id or embedding")
                    failed_items += 1
                    continue
                
                item_id = memory.id
                embedding = memory.embedding
                
                if embedding is None:
                    print(f"Skipping memory {item_id} with None embedding")
                    failed_items += 1
                    continue
                    
                # Validate embedding
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                # Check for NaN/Inf values
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    print(f"Warning: Memory {item_id} has NaN/Inf values in embedding, replacing with zeros")
                    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure proper shape and dtype
                if len(embedding.shape) != 1:
                    print(f"Warning: Memory {item_id} has wrong embedding shape {embedding.shape}, reshaping")
                    embedding = embedding.flatten()
                
                if embedding.shape[0] != embedding_dim:
                    print(f"Warning: Memory {item_id} has wrong embedding dimension {embedding.shape[0]}, adjusting")
                    if embedding.shape[0] < embedding_dim:
                        # Pad with zeros
                        new_embedding = np.zeros(embedding_dim, dtype=np.float32)
                        new_embedding[:embedding.shape[0]] = embedding
                        embedding = new_embedding
                    else:
                        # Truncate
                        embedding = embedding[:embedding_dim]
                
                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                
                # Add to index
                print(f"Adding memory {item_id} to index...")
                success = await new_index.add(item_id, embedding)
                
                if success:
                    added_memories += 1
                    print(f"Successfully added memory {item_id} to index")
                else:
                    print(f"Failed to add memory {item_id} to index (add returned False)")
                    failed_items += 1
            except Exception as e:
                print(f"Error adding memory {memory.id if hasattr(memory, 'id') else 'unknown'} to index: {e}")
                traceback.print_exc()
                failed_items += 1
        
        # Process assemblies
        print("Adding assemblies to new index...")
        for assembly in assemblies:
            try:
                if not hasattr(assembly, 'id') or not hasattr(assembly, 'composite_embedding'):
                    print(f"Skipping invalid assembly without id or composite_embedding")
                    failed_items += 1
                    continue
                
                item_id = f"asm:{assembly.id}"
                embedding = assembly.composite_embedding
                
                if embedding is None:
                    print(f"Skipping assembly {assembly.id} with None composite_embedding")
                    failed_items += 1
                    continue
                
                # Validate embedding (same validation logic as above)
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                # Check for NaN/Inf values
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    print(f"Warning: Assembly {assembly.id} has NaN/Inf values in embedding, replacing with zeros")
                    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure proper shape and dtype
                if len(embedding.shape) != 1:
                    print(f"Warning: Assembly {assembly.id} has wrong embedding shape {embedding.shape}, reshaping")
                    embedding = embedding.flatten()
                
                if embedding.shape[0] != embedding_dim:
                    print(f"Warning: Assembly {assembly.id} has wrong embedding dimension {embedding.shape[0]}, adjusting")
                    if embedding.shape[0] < embedding_dim:
                        # Pad with zeros
                        new_embedding = np.zeros(embedding_dim, dtype=np.float32)
                        new_embedding[:embedding.shape[0]] = embedding
                        embedding = new_embedding
                    else:
                        # Truncate
                        embedding = embedding[:embedding_dim]
                
                if embedding.dtype != np.float32:
                    embedding = embedding.astype(np.float32)
                
                # Add to index
                print(f"Adding assembly {assembly.id} to index (as {item_id})...")
                success = await new_index.add(item_id, embedding)
                
                if success:
                    added_assemblies += 1
                    print(f"Successfully added assembly {assembly.id} to index")
                else:
                    print(f"Failed to add assembly {assembly.id} to index (add returned False)")
                    failed_items += 1
            except Exception as e:
                print(f"Error adding assembly {assembly.id if hasattr(assembly, 'id') else 'unknown'} to index: {e}")
                traceback.print_exc()
                failed_items += 1
        
        # 5. Save the new index
        print("Saving new index...")
        save_success = new_index.save()
        if save_success:
            print("Successfully saved new index!")
        else:
            print("Failed to save new index!")
            return False
        
        # 6. Verify the new index
        print("Verifying new index...")
        vector_count = await new_index.count_async()
        print(f"New index contains {vector_count} vectors")
        
        map_size = len(await new_index.get_mapping())
        print(f"Index mapping contains {map_size} entries")
        
        print(f"Summary: Added {added_memories} memories, {added_assemblies} assemblies. Failed/Skipped {failed_items} items.")
        
        if vector_count > 0 and map_size == vector_count:
            print("Index rebuild SUCCESS!")
            return True
        else:
            print("Index appears inconsistent after rebuild!")
            return False
    
    except Exception as e:
        print(f"Unexpected error during index rebuild: {e}")
        traceback.print_exc()
        return False

def main():
    print("======== VECTOR INDEX DIRECT REPAIR UTILITY ========")
    try:
        success = asyncio.run(rebuild_vector_index_directly())
        
        if success:
            print("\nVECTOR INDEX REBUILD COMPLETED SUCCESSFULLY")
            return 0
        else:
            print("\nVECTOR INDEX REBUILD FAILED")
            return 1
    except Exception as e:
        print(f"Fatal error in main: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
