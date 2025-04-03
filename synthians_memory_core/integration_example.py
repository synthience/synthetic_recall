"""
Integration Example for Synthians Memory Core Stability Improvements.

This module demonstrates how to use the embedding_validators and vector_index_repair
modules to enhance stability in the memory core system.

Usage:
1. Import these improved functions where needed in your codebase
2. Add pre-retrieval integrity checks to catch issues early
3. Ensure all embedding operations use validated embeddings
4. Enhance assembly handling with proper validation
"""

import asyncio
import logging
import time
import uuid
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Import the utility modules
from synthians_memory_core.utils.embedding_validators import (
    validate_embedding,
    align_vector_dimensions,
    align_vectors_for_comparison
)
from synthians_memory_core.utils.vector_index_repair import (
    diagnose_vector_index,
    repair_vector_index,
    validate_vector_index_integrity,
    verify_vector_dimensions,
    correct_id_mapping_discrepancies
)

logger = logging.getLogger(__name__)

# Example 1: Enhanced Memory Processing
async def enhanced_process_new_memory(memory_core, content, embedding, metadata=None):
    """Enhanced memory processing with improved validation."""
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[ENHANCED][{trace_id}] Processing new memory")
    
    # Validate the embedding
    embedding_dim = memory_core.config.get('embedding_dim', 768)
    index_type = memory_core.config.get('index_type', 'L2')
    validated_embedding = validate_embedding(
        embedding, 
        target_dim=embedding_dim,
        normalize=True, 
        index_type=index_type
    )
    
    if validated_embedding is None:
        logger.error(f"[ENHANCED][{trace_id}] Invalid embedding provided, cannot process memory")
        return None, 0.0
    
    # Process memory using validated embedding
    result = await memory_core.process_new_memory(
        content,
        embedding=validated_embedding,
        metadata=metadata or {}
    )
    
    return result

# Example 2: Enhanced Assembly Update
async def enhanced_update_assemblies(memory_core, memory):
    """Enhanced assembly update with robust embedding validation."""
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[ENHANCED][{trace_id}] Updating assemblies for memory {memory.id}")
    
    # Skip if no embedding
    if memory.embedding is None:
        logger.warning(f"[ENHANCED][{trace_id}] Memory {memory.id} has no embedding, skipping assembly update")
        return
    
    # Validate memory embedding
    embedding_dim = memory_core.config.get('embedding_dim', 768)
    index_type = memory_core.config.get('index_type', 'L2')
    validated_embedding = validate_embedding(
        memory.embedding, 
        target_dim=embedding_dim,
        normalize=True, 
        index_type=index_type
    )
    
    if validated_embedding is None:
        logger.error(f"[ENHANCED][{trace_id}] Invalid embedding for memory {memory.id}, skipping assembly update")
        return
    
    # Find assembly candidates
    query_emb = validated_embedding
    threshold = memory_core.config.get('assembly_threshold', 0.75)
    assembly_vector_k = memory_core.config.get('assembly_vector_search_threshold', 50)
    
    # Search vector index with validated embedding
    logger.info(f"[ENHANCED][{trace_id}] Searching for similar memories for assembly formation")
    try:
        # Ensure vector index integrity before search
        is_valid, diag = await validate_vector_index_integrity(
            memory_core.vector_index, 
            memory_core.vector_index.id_to_index
        )
        if not is_valid:
            logger.warning(f"[ENHANCED][{trace_id}] Vector index inconsistency detected before assembly search: {diag}")
            # Continue anyway - the search might still work
        
        similar_assemblies = await memory_core.vector_index.search(
            query_emb, assembly_vector_k
        )
        
        # Process similar assemblies safely
        for asm_id, similarity in similar_assemblies:
            if not asm_id.startswith("asm:"):
                continue
                
            # Extract actual assembly ID
            asm_id = asm_id[4:]  # Remove "asm:" prefix
            if similarity < threshold:
                continue
                
            # Add memory to assembly
            if asm_id in memory_core.assemblies:
                asm = memory_core.assemblies[asm_id]
                
                # Add memory to assembly with validated embedding
                added = asm.add_memory(memory, validated_embedding)
                
                if added:
                    # Update assembly in vector index
                    if asm.composite_embedding is not None:
                        validated_composite = validate_embedding(
                            asm.composite_embedding,
                            target_dim=embedding_dim,
                            normalize=True, 
                            index_type=index_type
                        )
                        
                        if validated_composite is not None:
                            # Update assembly vector in index
                            await memory_core.vector_index.update_entry(
                                f"asm:{asm_id}", 
                                validated_composite
                            )
                    
                    # Update memory to assembly mapping
                    async with memory_core._lock:
                        if memory.id in memory_core.memory_to_assemblies:
                            memory_core.memory_to_assemblies[memory.id].add(asm_id)
                        else:
                            memory_core.memory_to_assemblies[memory.id] = {asm_id}
        
    except Exception as e:
        logger.error(f"[ENHANCED][{trace_id}] Error searching for similar assemblies: {e}")

# Example 3: Enhanced Memory Retrieval with Pre-Check
async def enhanced_retrieve_memories(memory_core, query, top_k=5, threshold=None):
    """Enhanced memory retrieval with pre-retrieval integrity checks."""
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[ENHANCED][{trace_id}] Starting enhanced memory retrieval")
    
    # Verify vector index integrity before retrieval
    logger.info(f"[ENHANCED][{trace_id}] Performing pre-retrieval integrity check")
    is_valid, diagnostics = await validate_vector_index_integrity(
        memory_core.vector_index, 
        memory_core.vector_index.id_to_index
    )
    
    if not is_valid:
        logger.warning(f"[ENHANCED][{trace_id}] Vector index integrity check failed: {diagnostics}")
        
        # Check if serious issue that requires repair
        if diagnostics.get("issue") in ["empty_index_with_mappings", "large_count_mismatch"]:
            logger.warning(f"[ENHANCED][{trace_id}] Critical index inconsistency detected, attempting repair")
            
            # Attempt repair
            async def fetch_embeddings_callback(ids):
                # Implementation depends on your storage mechanism
                # This is a placeholder
                result = {}
                for mem_id in ids:
                    if mem_id.startswith("asm:"):
                        asm_id = mem_id[4:]
                        if asm_id in memory_core.assemblies:
                            result[mem_id] = memory_core.assemblies[asm_id].composite_embedding
                    else:
                        memory = await memory_core.get_memory(mem_id)
                        if memory and memory.embedding is not None:
                            result[mem_id] = memory.embedding
                return result
            
            # Try to repair the index
            embedding_dim = memory_core.config.get('embedding_dim', 768)
            success, _, new_index, new_mapping = await repair_vector_index(
                memory_core.vector_index,
                memory_core.vector_index.id_to_index,
                embedding_dim,
                repair_mode="auto",
                fetch_embeddings_callback=fetch_embeddings_callback
            )
            
            if success and new_index is not None:
                logger.info(f"[ENHANCED][{trace_id}] Vector index repair successful")
                # In a real implementation, you would update the memory_core's vector_index
                # memory_core.vector_index = new_index
                # memory_core.vector_index.id_to_index = new_mapping
            else:
                logger.error(f"[ENHANCED][{trace_id}] Vector index repair failed")
    
    # Generate and validate query embedding
    embedding_dim = memory_core.config.get('embedding_dim', 768)
    index_type = memory_core.config.get('index_type', 'L2')
    query_embedding = None
    
    if query:
        query_embedding = await memory_core.generate_embedding(query)
        query_embedding = validate_embedding(
            query_embedding, 
            target_dim=embedding_dim,
            normalize=True, 
            index_type=index_type
        )
        
        if query_embedding is None:
            logger.error(f"[ENHANCED][{trace_id}] Invalid query embedding generated")
            return {"success": False, "memories": [], "error": "Invalid query embedding"}
    
    # Proceed with retrieval using normal flow
    return await memory_core.retrieve_memories(
        query=query,
        top_k=top_k,
        threshold=threshold
    )

# Example 4: Enhanced Assembly Activation
async def enhanced_activate_assemblies(memory_core, query_embedding):
    """Enhanced assembly activation with improved validation and debugging."""
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[ENHANCED][{trace_id}] Activating assemblies")
    
    # Validate query embedding
    embedding_dim = memory_core.config.get('embedding_dim', 768)
    index_type = memory_core.config.get('index_type', 'L2')
    validated_query = validate_embedding(
        query_embedding, 
        target_dim=embedding_dim,
        normalize=True, 
        index_type=index_type
    )
    
    if validated_query is None:
        logger.error(f"[ENHANCED][{trace_id}] Invalid query embedding for assembly activation")
        return []
    
    # Get assembly search configuration
    activation_k = memory_core.config.get('max_assembly_activation', 10)
    activation_threshold = memory_core.config.get('assembly_activation_threshold', 0.6)
    
    # Search for assembly candidates
    try:
        # Use an asm: prefix filter to only search assemblies
        assembly_search_results = await memory_core.vector_index.search(
            validated_query, k=activation_k
        )
        
        # Filter and extract results
        activated_assemblies = []
        activated_ids = set()
        
        for asm_id, similarity in assembly_search_results:
            # Skip non-assembly results and low similarity
            if not asm_id.startswith("asm:") or similarity < activation_threshold:
                continue
                
            # Extract actual assembly ID
            asm_id_clean = asm_id[4:]  # Remove "asm:" prefix
            
            # Skip if already activated
            if asm_id_clean in activated_ids:
                continue
                
            # Get assembly
            if asm_id_clean in memory_core.assemblies:
                asm = memory_core.assemblies[asm_id_clean]
                
                # Verify assembly embedding
                if asm.composite_embedding is not None:
                    verified_embedding = validate_embedding(
                        asm.composite_embedding,
                        target_dim=embedding_dim,
                        normalize=True, 
                        index_type=index_type
                    )
                    
                    if verified_embedding is None:
                        logger.warning(f"[ENHANCED][{trace_id}] Assembly {asm_id_clean} has invalid embedding")
                        continue
                    
                    # Calculate actual similarity for verification
                    vec1, vec2 = align_vectors_for_comparison(validated_query, verified_embedding)
                    actual_similarity = np.dot(vec1, vec2)
                    
                    # Log discrepancy if significant
                    if abs(actual_similarity - similarity) > 0.1:
                        logger.warning(f"[ENHANCED][{trace_id}] Similarity discrepancy for {asm_id_clean}: "
                                      f"search={similarity:.4f}, calculated={actual_similarity:.4f}")
                    
                    # Add to activated assemblies
                    activated_assemblies.append((asm, actual_similarity))
                    activated_ids.add(asm_id_clean)
                    
                    # Update activation stats
                    asm.activation_count += 1
                    asm.last_activation = time.time()
                
        return activated_assemblies
                
    except Exception as e:
        logger.error(f"[ENHANCED][{trace_id}] Error activating assemblies: {e}", exc_info=True)
        return []

# Example 5: Debug Test Failures
async def debug_retrieval_boosting(memory_core, query_embedding, memory_ids):
    """Debug assembly boosting to identify issues in test_02_retrieval_boosting."""
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[DEBUG][{trace_id}] Diagnosing retrieval boosting issues")
    
    # Validate query embedding
    embedding_dim = memory_core.config.get('embedding_dim', 768)
    index_type = memory_core.config.get('index_type', 'L2')
    validated_query = validate_embedding(
        query_embedding, 
        target_dim=embedding_dim,
        normalize=True, 
        index_type=index_type
    )
    
    if validated_query is None:
        logger.error(f"[DEBUG][{trace_id}] Invalid query embedding for debugging")
        return {"error": "Invalid query embedding"}
    
    # 1. Check vector index integrity
    is_valid, diag = await validate_vector_index_integrity(
        memory_core.vector_index, 
        memory_core.vector_index.id_to_index
    )
    
    diagnostics = {
        "vector_index_valid": is_valid,
        "vector_index_diagnostics": diag,
        "memory_ids": memory_ids,
        "activated_assemblies": [],
        "memory_embeddings": {},
        "similarity_scores": {},
        "boosted_scores": {}
    }
    
    # 2. Check assemblies
    try:
        # Activate assemblies
        activated = await enhanced_activate_assemblies(memory_core, validated_query)
        
        for asm, sim in activated:
            diagnostics["activated_assemblies"].append({
                "id": asm.id,
                "similarity": sim,
                "member_count": len(asm.memories) if hasattr(asm, "memories") else 0,
                "last_activation": asm.last_activation if hasattr(asm, "last_activation") else None,
                "activation_count": asm.activation_count if hasattr(asm, "activation_count") else 0
            })
            
        # 3. Check each memory individually
        for mem_id in memory_ids:
            memory = await memory_core.get_memory(mem_id)
            
            if not memory or not memory.embedding is not None:
                diagnostics["memory_embeddings"][mem_id] = "missing"
                continue
                
            # Validate embedding
            valid_emb = validate_embedding(
                memory.embedding,
                target_dim=embedding_dim,
                normalize=True, 
                index_type=index_type
            )
            
            if valid_emb is None:
                diagnostics["memory_embeddings"][mem_id] = "invalid"
                continue
                
            diagnostics["memory_embeddings"][mem_id] = "valid"
            
            # Calculate similarity
            vec1, vec2 = align_vectors_for_comparison(validated_query, valid_emb)
            similarity = np.dot(vec1, vec2)
            diagnostics["similarity_scores"][mem_id] = similarity
            
            # Check assembly membership
            assemblies = memory_core.memory_to_assemblies.get(mem_id, set())
            diag_member_of = []
            
            for asm_id in assemblies:
                # Check if this assembly is activated
                is_activated = any(a[0].id == asm_id for a in activated)
                diag_member_of.append({
                    "assembly_id": asm_id,
                    "activated": is_activated
                })
                
            diagnostics["assembly_membership"] = diag_member_of
            
        return diagnostics
                
    except Exception as e:
        logger.error(f"[DEBUG][{trace_id}] Error in debugging: {e}", exc_info=True)
        diagnostics["error"] = str(e)
        return diagnostics

# Integration Example - How to use these enhanced functions
async def example_usage():
    """Example of how to use the enhanced functions."""
    from synthians_memory_core import SynthiansMemoryCore
    
    # Initialize memory core
    config = {
        'embedding_dim': 768,
        'index_type': 'L2',
        'storage_path': './faiss_index'
    }
    memory_core = SynthiansMemoryCore(config)
    
    # 1. Process a new memory with validation
    content = "This is a test memory with improved validation."
    embedding = np.random.rand(768).astype(np.float32)  # Random embedding for testing
    
    # Add some NaN values to test validation
    embedding[5:10] = np.nan
    
    memory_id, score = await enhanced_process_new_memory(memory_core, content, embedding)
    print(f"Processed memory with ID {memory_id} and score {score}")
    
    # 2. Retrieve memories with pre-checks
    query = "Test query with pre-retrieval checks"
    results = await enhanced_retrieve_memories(memory_core, query, top_k=5)
    
    # 3. Debug assembly issues
    sample_ids = [memory_id]
    diagnostics = await debug_retrieval_boosting(
        memory_core, 
        np.random.rand(768).astype(np.float32),  # Random query embedding
        sample_ids
    )
    print(f"Debug diagnostics: {diagnostics}")
    
    # 4. Verify vector dimensions
    async def sample_embeddings_callback(ids):
        result = {}
        for mem_id in ids:
            # Mock embedding
            result[mem_id] = np.random.rand(768).astype(np.float32)
        return result
    
    dimensions = await verify_vector_dimensions(
        memory_core.vector_index,
        sample_ids,
        sample_embeddings_callback
    )
    print(f"Dimension verification: {dimensions}")

# Run the example
if __name__ == "__main__":
    asyncio.run(example_usage())