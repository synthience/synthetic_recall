"""
Vector Index Repair Utilities for Synthians Memory Core.

This module provides specialized repair functions for addressing
common vector index inconsistencies between FAISS and ID mappings.
"""

import os
import logging
import asyncio
import json
import time
import uuid
import numpy as np
import faiss
from typing import Dict, List, Tuple, Any, Optional, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

async def diagnose_vector_index(index, id_to_index: Dict[str, int]) -> Dict[str, Any]:
    """Diagnose vector index issues without attempting repairs.
    
    Args:
        index: FAISS index object
        id_to_index: Dictionary mapping memory IDs to FAISS index positions
        
    Returns:
        Diagnostics information dictionary
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Diagnosing vector index")
    
    diagnostics = {}
    
    # Check if index is initialized
    if index is None:
        diagnostics["status"] = "INVALID"
        diagnostics["error"] = "Index not initialized"
        diagnostics["faiss_count"] = 0
        diagnostics["id_mapping_count"] = len(id_to_index)
        diagnostics["is_consistent"] = False
        return diagnostics
    
    # Get FAISS and mapping counts
    faiss_count = index.ntotal
    mapping_count = len(id_to_index) if id_to_index else 0
    
    diagnostics["faiss_count"] = faiss_count
    diagnostics["id_mapping_count"] = mapping_count
    diagnostics["is_index_id_map"] = hasattr(index, 'id_map')
    
    # Check consistency
    is_consistent = faiss_count == mapping_count
    diagnostics["is_consistent"] = is_consistent
    
    # Identify specific issues
    if faiss_count == 0 and mapping_count > 0:
        diagnostics["issue"] = "empty_index_with_mappings"
        diagnostics["recommended_repair"] = "rebuild_from_persistence"
    elif faiss_count > 0 and mapping_count == 0:
        diagnostics["issue"] = "index_without_mappings"
        diagnostics["recommended_repair"] = "recreate_mapping"
    elif faiss_count != mapping_count:
        diff = abs(faiss_count - mapping_count)
        percent_diff = diff / max(faiss_count, mapping_count) * 100
        
        if percent_diff > 20 or diff > 10:
            diagnostics["issue"] = "large_count_mismatch"
            diagnostics["recommended_repair"] = "rebuild_from_persistence"
        else:
            diagnostics["issue"] = "minor_count_mismatch"
            diagnostics["recommended_repair"] = "recreate_mapping"
    
    logger.info(f"[REPAIR][{trace_id}] Diagnostics complete: {diagnostics}")
    return diagnostics

async def rebuild_id_mapping(
    index,
    fetch_embeddings_callback: Optional[Callable[[List[str]], Awaitable[Dict[str, np.ndarray]]]] = None
) -> Dict[str, int]:
    """Recreate ID mapping dictionary from the index.
    
    Args:
        index: FAISS index object
        fetch_embeddings_callback: Optional callback to fetch embeddings for verification
        
    Returns:
        Reconstructed ID mapping dictionary
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Rebuilding ID mapping from index")
    
    # Check if index supports ID retrieval
    if not hasattr(index, 'id_map'):
        logger.error(f"[REPAIR][{trace_id}] Index does not support ID retrieval, cannot rebuild mapping")
        return {}
    
    # Extract IDs directly from the index
    try:
        ntotal = index.ntotal
        if ntotal == 0:
            logger.warning(f"[REPAIR][{trace_id}] Index is empty, nothing to rebuild")
            return {}
        
        logger.info(f"[REPAIR][{trace_id}] Extracting {ntotal} IDs from index")
        
        # Get all numeric IDs
        numeric_ids = []
        for i in range(ntotal):
            try:
                idx = index.id_map.at(i)
                numeric_ids.append(int(idx))
            except Exception as e:
                logger.error(f"[REPAIR][{trace_id}] Error extracting ID at position {i}: {e}")
        
        logger.info(f"[REPAIR][{trace_id}] Extracted {len(numeric_ids)} numeric IDs")
        
        # Use callback to fetch original string IDs if provided
        if fetch_embeddings_callback:
            logger.info(f"[REPAIR][{trace_id}] Using callback to fetch original memory IDs")
            
            # Since we don't have the original string IDs, we'd need to search
            # through all memories and match embeddings to our index
            # This is complex and would be implemented if needed
            pass
        
        # Fallback: recreate mapping with synthetic IDs
        new_mapping = {}
        for i, numeric_id in enumerate(numeric_ids):
            # Generate a synthetic ID
            synthetic_id = f"recovered_mem_{numeric_id}_{i}"
            new_mapping[synthetic_id] = numeric_id
        
        logger.info(f"[REPAIR][{trace_id}] Created {len(new_mapping)} synthetic ID mappings")
        return new_mapping
        
    except Exception as e:
        logger.error(f"[REPAIR][{trace_id}] Failed to rebuild ID mapping: {e}", exc_info=True)
        return {}

async def rebuild_index_from_mappings(
    id_to_index: Dict[str, int],
    embedding_dim: int,
    fetch_embeddings_callback: Callable[[List[str]], Awaitable[Dict[str, np.ndarray]]]
) -> Tuple[Optional[Any], Dict[str, int]]:
    """Rebuild FAISS index from ID mappings and embeddings.
    
    Args:
        id_to_index: Dictionary mapping memory IDs to FAISS index positions
        embedding_dim: Dimension of embeddings
        fetch_embeddings_callback: Callback to fetch embeddings for memories
        
    Returns:
        Tuple of (new_index, new_id_to_index)
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Rebuilding index from mappings")
    
    if not id_to_index:
        logger.error(f"[REPAIR][{trace_id}] No mappings provided for rebuild")
        return None, {}
    
    # Create a new FAISS index
    try:
        logger.info(f"[REPAIR][{trace_id}] Creating new IndexIDMap with dimension {embedding_dim}")
        base_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        new_index = faiss.IndexIDMap(base_index)
        
        # Extract memory IDs
        memory_ids = list(id_to_index.keys())
        logger.info(f"[REPAIR][{trace_id}] Fetching embeddings for {len(memory_ids)} memories")
        
        # Fetch embeddings
        if fetch_embeddings_callback:
            embeddings_dict = await fetch_embeddings_callback(memory_ids)
            logger.info(f"[REPAIR][{trace_id}] Fetched {len(embeddings_dict)} embeddings")
            
            # Process embeddings in batches
            batch_size = 100
            new_id_to_index = {}
            added_count = 0
            
            for i in range(0, len(memory_ids), batch_size):
                batch = memory_ids[i:i+batch_size]
                
                # Collect batch embeddings
                batch_embeddings = []
                batch_ids = []
                batch_numeric_ids = []
                
                for mem_id in batch:
                    if mem_id in embeddings_dict:
                        embedding = embeddings_dict[mem_id]
                        if embedding is not None:
                            # Convert to proper shape
                            embedding = np.reshape(embedding, (1, -1)).astype(np.float32)
                            
                            # Get numeric ID (use old one if available)
                            numeric_id = id_to_index.get(mem_id, hash(mem_id) % (2**31 - 1))
                            
                            batch_embeddings.append(embedding)
                            batch_ids.append(mem_id)
                            batch_numeric_ids.append(numeric_id)
                
                if batch_embeddings:
                    # Stack embeddings
                    stacked_embeddings = np.vstack(batch_embeddings)
                    ids_array = np.array(batch_numeric_ids, dtype=np.int64)
                    
                    # Add to index
                    new_index.add_with_ids(stacked_embeddings, ids_array)
                    
                    # Update mappings
                    for mem_id, numeric_id in zip(batch_ids, batch_numeric_ids):
                        new_id_to_index[mem_id] = numeric_id
                        added_count += 1
            
            logger.info(f"[REPAIR][{trace_id}] Successfully added {added_count} vectors to rebuilt index")
            return new_index, new_id_to_index
        else:
            logger.error(f"[REPAIR][{trace_id}] No embedding fetch callback provided, cannot rebuild index")
            return None, {}
            
    except Exception as e:
        logger.error(f"[REPAIR][{trace_id}] Failed to rebuild index: {e}", exc_info=True)
        return None, {}

async def repair_vector_index(
    index, 
    id_to_index: Dict[str, int],
    embedding_dim: int,
    repair_mode: str = "auto",
    fetch_embeddings_callback: Optional[Callable[[List[str]], Awaitable[Dict[str, np.ndarray]]]] = None
) -> Tuple[bool, Dict[str, Any], Optional[Any], Dict[str, int]]:
    """Repair vector index based on diagnosed issues.
    
    Args:
        index: FAISS index object
        id_to_index: Dictionary mapping memory IDs to FAISS index positions
        embedding_dim: Dimension of embeddings
        repair_mode: Repair strategy to use
        fetch_embeddings_callback: Callback to fetch embeddings for memories
        
    Returns:
        Tuple of (success, diagnostics, new_index, new_id_to_index)
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Starting vector index repair with mode: {repair_mode}")
    
    # First diagnose the index
    diagnostics = await diagnose_vector_index(index, id_to_index)
    
    # If already consistent and not forced, just return
    if diagnostics.get("is_consistent", False) and repair_mode != "force":
        logger.info(f"[REPAIR][{trace_id}] Index is already consistent, no repair needed")
        return True, diagnostics, index, id_to_index
    
    # If auto mode, use recommended repair
    if repair_mode == "auto":
        repair_mode = diagnostics.get("recommended_repair", "recreate_mapping")
        logger.info(f"[REPAIR][{trace_id}] Auto repair mode selected: {repair_mode}")
    
    # Apply repair strategy
    if repair_mode == "recreate_mapping":
        logger.info(f"[REPAIR][{trace_id}] Recreating ID mapping from index")
        new_id_to_index = await rebuild_id_mapping(index, fetch_embeddings_callback)
        
        if new_id_to_index:
            logger.info(f"[REPAIR][{trace_id}] Mapping recreation successful: {len(new_id_to_index)} entries")
            return True, diagnostics, index, new_id_to_index
        else:
            logger.error(f"[REPAIR][{trace_id}] Mapping recreation failed")
            return False, diagnostics, index, id_to_index
            
    elif repair_mode == "rebuild_from_persistence":
        logger.info(f"[REPAIR][{trace_id}] Rebuilding index from persistence")
        
        if not fetch_embeddings_callback:
            logger.error(f"[REPAIR][{trace_id}] Cannot rebuild without fetch_embeddings_callback")
            return False, diagnostics, index, id_to_index
        
        new_index, new_id_to_index = await rebuild_index_from_mappings(
            id_to_index, embedding_dim, fetch_embeddings_callback
        )
        
        if new_index is not None:
            logger.info(f"[REPAIR][{trace_id}] Index rebuild successful: {new_index.ntotal} vectors")
            return True, diagnostics, new_index, new_id_to_index
        else:
            logger.error(f"[REPAIR][{trace_id}] Index rebuild failed")
            return False, diagnostics, index, id_to_index
    
    else:
        logger.error(f"[REPAIR][{trace_id}] Unknown repair mode: {repair_mode}")
        return False, diagnostics, index, id_to_index

async def validate_vector_index_integrity(index, id_to_index: Dict[str, int]) -> Tuple[bool, Dict[str, Any]]:
    """Validate vector index integrity with more sophisticated checks.
    
    Args:
        index: FAISS index object
        id_to_index: Dictionary mapping memory IDs to FAISS index positions
        
    Returns:
        Tuple of (is_valid, diagnostics_dict)
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Validating vector index integrity")
    
    # Get basic diagnostics first
    diagnostics = await diagnose_vector_index(index, id_to_index)
    
    # Default to invalid if basic checks failed
    is_valid = diagnostics.get("is_consistent", False)
    
    # Additional checks for index functionality
    if index is not None:
        try:
            # Create a test vector
            test_vector = np.random.rand(1, index.d).astype(np.float32)
            
            # Try to search with the test vector
            ntotal_before = index.ntotal
            search_success = False
            
            try:
                # Search with a small k
                _, _ = index.search(test_vector, min(5, max(1, ntotal_before)))
                search_success = True
                diagnostics["search_test"] = "passed"
            except Exception as search_e:
                logger.error(f"[REPAIR][{trace_id}] Search test failed: {search_e}")
                diagnostics["search_test"] = "failed"
                diagnostics["search_error"] = str(search_e)
                is_valid = False
            
            # Test ID retrieval if it's an IDMap
            if hasattr(index, 'id_map') and ntotal_before > 0:
                try:
                    # Try to get an ID at position 0
                    _ = index.id_map.at(0)
                    diagnostics["id_retrieval_test"] = "passed"
                except Exception as id_e:
                    logger.error(f"[REPAIR][{trace_id}] ID retrieval test failed: {id_e}")
                    diagnostics["id_retrieval_test"] = "failed"
                    diagnostics["id_retrieval_error"] = str(id_e)
                    is_valid = False
            
            # Only test writing if other tests pass
            if search_success and is_valid and hasattr(index, 'add_with_ids'):
                try:
                    # Generate a test ID outside of normal range
                    test_id = 999999999
                    test_id_array = np.array([test_id], dtype=np.int64)
                    
                    # Add the test vector
                    index.add_with_ids(test_vector, test_id_array)
                    ntotal_after_add = index.ntotal
                    
                    # Verify count increased
                    if ntotal_after_add != ntotal_before + 1:
                        logger.error(f"[REPAIR][{trace_id}] Add test failed: count did not increase properly")
                        diagnostics["add_test"] = "failed"
                        diagnostics["add_error"] = "Count did not increase properly"
                        is_valid = False
                    else:
                        # Try to remove the test vector for cleanup if the index supports it
                        try:
                            if hasattr(index, 'remove_ids'):
                                index.remove_ids(test_id_array)
                                diagnostics["add_test"] = "passed_with_cleanup"
                            else:
                                diagnostics["add_test"] = "passed_no_cleanup"
                        except Exception as remove_e:
                            logger.warning(f"[REPAIR][{trace_id}] Could not clean up test vector: {remove_e}")
                            diagnostics["add_test"] = "passed_cleanup_failed"
                            
                except Exception as add_e:
                    logger.error(f"[REPAIR][{trace_id}] Add test failed: {add_e}")
                    diagnostics["add_test"] = "failed"
                    diagnostics["add_error"] = str(add_e)
                    is_valid = False
        
        except Exception as e:
            logger.error(f"[REPAIR][{trace_id}] Index functional tests failed with error: {e}", exc_info=True)
            diagnostics["functional_tests"] = "error"
            diagnostics["functional_error"] = str(e)
            is_valid = False
    
    if is_valid:
        logger.info(f"[REPAIR][{trace_id}] Vector index integrity validated successfully")
    else:
        logger.warning(f"[REPAIR][{trace_id}] Vector index integrity validation failed")
    
    return is_valid, diagnostics


async def verify_vector_dimensions(index, sample_ids: List[str], fetch_embeddings_callback: Callable) -> Dict[str, Any]:
    """Verify that vector dimensions are consistent in the index.
    
    Args:
        index: FAISS index object
        sample_ids: List of memory IDs to sample for dimension verification
        fetch_embeddings_callback: Callback to fetch embeddings for memories
        
    Returns:
        Dictionary with verification results
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Verifying vector dimensions for {len(sample_ids)} samples")
    
    results = {
        "index_dimension": index.d if index is not None else 0,
        "samples_checked": len(sample_ids),
        "dimension_mismatches": 0,
        "nan_inf_values": 0,
        "items_verified": 0
    }
    
    if not sample_ids or index is None:
        logger.warning(f"[REPAIR][{trace_id}] Cannot verify dimensions: {'no sample IDs' if not sample_ids else 'index is None'}") 
        return results
    
    try:
        # Fetch embeddings for sample IDs
        embeddings_dict = await fetch_embeddings_callback(sample_ids)
        
        for mem_id, embedding in embeddings_dict.items():
            if embedding is None:
                continue
                
            results["items_verified"] += 1
            
            # Check dimensions
            if len(embedding.shape) == 1:
                dim = embedding.shape[0]
            elif len(embedding.shape) == 2:
                dim = embedding.shape[1]
            else:
                # Skip invalid shapes
                continue
                
            if dim != index.d:
                results["dimension_mismatches"] += 1
            
            # Check for NaN/Inf values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                results["nan_inf_values"] += 1
        
        # Calculate percentages
        if results["items_verified"] > 0:
            results["dimension_mismatch_percent"] = (results["dimension_mismatches"] / results["items_verified"]) * 100
            results["nan_inf_percent"] = (results["nan_inf_values"] / results["items_verified"]) * 100
        
        logger.info(f"[REPAIR][{trace_id}] Dimension verification complete: {results}")
        return results
            
    except Exception as e:
        logger.error(f"[REPAIR][{trace_id}] Error verifying vector dimensions: {e}", exc_info=True)
        results["error"] = str(e)
        return results


async def correct_id_mapping_discrepancies(index, id_to_index: Dict[str, int]) -> Dict[str, int]:
    """Correct discrepancies between FAISS index and ID mapping.
    
    Args:
        index: FAISS index object
        id_to_index: Dictionary mapping memory IDs to FAISS index positions
        
    Returns:
        Corrected ID mapping dictionary
    """
    trace_id = str(uuid.uuid4())[:8]
    logger.info(f"[REPAIR][{trace_id}] Correcting ID mapping discrepancies")
    
    if index is None:
        logger.error(f"[REPAIR][{trace_id}] Cannot correct mapping for None index")
        return id_to_index.copy() if id_to_index else {}
    
    try:
        # Build a reverse mapping for validation
        index_to_id = {v: k for k, v in id_to_index.items()} if id_to_index else {}
        
        # Get FAISS index size
        ntotal = index.ntotal
        
        # Create a new mapping
        new_mapping = {}
        orphaned_ids = set()
        
        # Process only up to the index size
        if hasattr(index, 'id_map'):
            # For IDMap indices, extract actual IDs
            for i in range(ntotal):
                try:
                    idx = int(index.id_map.at(i))
                    
                    # Find the memory ID for this index
                    mem_id = None
                    for k, v in id_to_index.items():
                        if v == idx:
                            mem_id = k
                            break
                    
                    if mem_id:
                        new_mapping[mem_id] = idx
                    else:
                        # No memory ID found for this index
                        orphaned_ids.add(idx)
                except Exception as e:
                    logger.warning(f"[REPAIR][{trace_id}] Could not get ID at position {i}: {e}")
        else:
            # For non-IDMap indices, sequential IDs
            for mem_id, idx in id_to_index.items():
                if 0 <= idx < ntotal:
                    new_mapping[mem_id] = idx
                else:
                    # Index out of range
                    logger.warning(f"[REPAIR][{trace_id}] Index out of range: {idx} for {mem_id}, ntotal={ntotal}")
        
        # Report changes
        added = {k: v for k, v in new_mapping.items() if k not in id_to_index}
        removed = {k: v for k, v in id_to_index.items() if k not in new_mapping}
        
        logger.info(f"[REPAIR][{trace_id}] Mapping correction: {len(new_mapping)} entries kept, "
                   f"{len(added)} added, {len(removed)} removed, {len(orphaned_ids)} orphaned")
        
        return new_mapping
        
    except Exception as e:
        logger.error(f"[REPAIR][{trace_id}] Error correcting ID mapping: {e}", exc_info=True)
        return id_to_index.copy() if id_to_index else {}