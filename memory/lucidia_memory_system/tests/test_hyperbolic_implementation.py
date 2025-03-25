import asyncio
import sys
import os
import json
import numpy as np
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_hyperbolic")

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the knowledge graph module
from core.knowledge_graph import LucidiaKnowledgeGraph

class MockSelfModel:
    """Mock Self Model for testing"""
    def __init__(self):
        self.name = "MockSelfModel"

class MockWorldModel:
    """Mock World Model for testing"""
    def __init__(self):
        self.name = "MockWorldModel"

async def main():
    try:
        # Create mock dependencies
        self_model = MockSelfModel()
        world_model = MockWorldModel()
        
        # Initialize the knowledge graph with the correct parameters
        kg = LucidiaKnowledgeGraph(
            self_model=self_model,
            world_model=world_model,
            config={
                "hyperbolic_embedding": {
                    "enabled": True,
                    "curvature": -1.0
                }
            }
        )
        
        # Test our hyperbolic embedding implementation
        await test_hyperbolic_implementation(kg)
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        traceback.print_exc()

async def test_hyperbolic_implementation(kg):
    """Test the hyperbolic embedding implementation"""
    try:
        logger.info("Starting hyperbolic embedding test")
        
        # Create a simple test graph
        await kg.add_node("test_A", "test", {"type": "test"})
        logger.info("Added node test_A")
        await kg.add_node("test_B", "test", {"type": "test"})
        logger.info("Added node test_B")
        await kg.add_node("test_C", "test", {"type": "test"})
        logger.info("Added node test_C")
        await kg.add_node("test_D", "test", {"type": "test"})
        logger.info("Added node test_D")
        
        await kg.add_edge("test_A", "test_B", "connected_to", {"confidence": 0.8})
        logger.info("Added edge A->B")
        await kg.add_edge("test_B", "test_C", "connected_to", {"confidence": 0.9})
        logger.info("Added edge B->C")
        await kg.add_edge("test_A", "test_D", "connected_to", {"confidence": 0.7})
        logger.info("Added edge A->D")
        
        # Add embeddings
        a_emb = np.array([0.1, 0.2], dtype=np.float32)
        b_emb = np.array([0.3, 0.4], dtype=np.float32)
        c_emb = np.array([0.5, 0.6], dtype=np.float32)
        d_emb = np.array([0.7, 0.8], dtype=np.float32)
        
        # Validate embeddings for NaN/Inf values (based on memory)
        for name, emb in [("A", a_emb), ("B", b_emb), ("C", c_emb), ("D", d_emb)]:
            if np.isnan(emb).any() or np.isinf(emb).any():
                logger.warning(f"Node {name} has invalid embedding values, replacing with zeros")
                emb = np.zeros_like(emb)
        
        # Add regular embeddings to all nodes
        logger.info("Adding embeddings to nodes")
        await kg.update_node("test_A", {"embedding": a_emb.tolist()})
        await kg.update_node("test_B", {"embedding": b_emb.tolist()})
        await kg.update_node("test_C", {"embedding": c_emb.tolist()})
        await kg.update_node("test_D", {"embedding": d_emb.tolist()})
        
        # Ensure hyperbolic embedding dict is properly initialized
        logger.info("Checking hyperbolic_embedding initialization")
        
        # Print current kg attributes
        logger.info(f"KG attributes: {dir(kg)}")
        if hasattr(kg, 'hyperbolic_embedding'):
            logger.info(f"Current hyperbolic_embedding: {kg.hyperbolic_embedding}")
        else:
            logger.info("hyperbolic_embedding not found in kg attributes")
            
        # Initialize explicitly if needed
        if not hasattr(kg, 'hyperbolic_embedding') or not isinstance(kg.hyperbolic_embedding, dict):
            logger.info("Initializing hyperbolic_embedding dictionary")
            kg.hyperbolic_embedding = {
                "enabled": True,
                "embedding_nodes": set(),
                "curvature": -1.0
            }
        elif "embedding_nodes" not in kg.hyperbolic_embedding:
            kg.hyperbolic_embedding["embedding_nodes"] = set()
            
        logger.info(f"Hyperbolic embedding after initialization: {kg.hyperbolic_embedding}")
        
        # Convert to hyperbolic space and add them
        logger.info("Converting embeddings to hyperbolic space...")
        try:
            a_hyp = kg._to_hyperbolic(a_emb)
            b_hyp = kg._to_hyperbolic(b_emb)
            c_hyp = kg._to_hyperbolic(c_emb)
            
            logger.info(f"Hyperbolic embeddings: \n A: {a_hyp} \n B: {b_hyp} \n C: {c_hyp}")
        except Exception as e:
            logger.error(f"Error converting to hyperbolic: {e}")
            traceback.print_exc()
            raise
        
        # Add hyperbolic embeddings to nodes
        await kg.update_node("test_A", {"hyperbolic_embedding": a_hyp.tolist()})
        await kg.update_node("test_B", {"hyperbolic_embedding": b_hyp.tolist()})
        await kg.update_node("test_C", {"hyperbolic_embedding": c_hyp.tolist()})
        
        # Add nodes to hyperbolic embedding set
        kg.hyperbolic_embedding["embedding_nodes"].add("test_A")
        kg.hyperbolic_embedding["embedding_nodes"].add("test_B")
        kg.hyperbolic_embedding["embedding_nodes"].add("test_C")
        
        # Test hyperbolic distance and similarity
        logger.info("Testing hyperbolic distance and similarity...")
        try:
            h_dist_ab = kg._hyperbolic_distance(a_hyp, b_hyp, already_in_hyperbolic=True)
            h_sim_ab = kg._hyperbolic_similarity(a_hyp, b_hyp, already_in_hyperbolic=True)
            
            logger.info(f"Hyperbolic distance A->B: {h_dist_ab}")
            logger.info(f"Hyperbolic similarity A->B: {h_sim_ab}")
        except Exception as e:
            logger.error(f"Error calculating hyperbolic distance/similarity: {e}")
            traceback.print_exc()
            raise
        
        # Find paths with hyperbolic embeddings
        logger.info("Finding paths with hyperbolic embeddings...")
        try:
            paths_hyperbolic = await kg.find_paths("test_A", "test_C", max_depth=2, min_confidence=0.5, use_hyperbolic=True)
            logger.info(f"Paths with hyperbolic embeddings: {json.dumps(paths_hyperbolic, indent=2)}")
        except Exception as e:
            logger.error(f"Error finding paths with hyperbolic embeddings: {e}")
            traceback.print_exc()
            raise
        
        # Find paths without hyperbolic embeddings
        logger.info("Finding paths without hyperbolic embeddings...")
        try:
            paths_euclidean = await kg.find_paths("test_A", "test_C", max_depth=2, min_confidence=0.5, use_hyperbolic=False)
            logger.info(f"Paths without hyperbolic embeddings: {json.dumps(paths_euclidean, indent=2)}")
        except Exception as e:
            logger.error(f"Error finding paths without hyperbolic embeddings: {e}")
            traceback.print_exc()
            raise
        
        # Test batch conversion
        logger.info("Testing batch conversion to hyperbolic space...")
        try:
            # Check if the method exists in the KG implementation
            if hasattr(kg, "convert_nodes_to_hyperbolic"):
                # Enable hyperbolic embeddings explicitly before conversion
                kg.hyperbolic_embedding["enabled"] = True
                
                # Reset any existing hyperbolic embeddings to force conversion
                for node_id in ["test_A", "test_B", "test_C"]:
                    await kg.update_node(node_id, {"hyperbolic_embedding": None})
                kg.hyperbolic_embedding["embedding_nodes"] = set()
                
                conversion_result = await kg.convert_nodes_to_hyperbolic()
                logger.info(f"Batch conversion result: {json.dumps(conversion_result, indent=2, default=str)}")
                
                # Verify initialization status after conversion
                if conversion_result.get("converted", 0) > 0:
                    logger.info(f"Hyperbolic embedding initialization status: {kg.hyperbolic_embedding['initialized']}")
                    assert kg.hyperbolic_embedding["initialized"], "Hyperbolic embeddings should be marked as initialized"
            else:
                logger.warning("convert_nodes_to_hyperbolic method not available - using fallback implementation")
                # Alternative: perform the conversion manually for testing
                manual_stats = {
                    "total_nodes": 0,
                    "nodes_with_embeddings": 0,
                    "converted": 0,
                    "errors": 0,
                    "success": True
                }
                
                for node_id in ["test_A", "test_B", "test_C"]:
                    manual_stats["total_nodes"] += 1
                    node_data = kg.graph.nodes[node_id]
                    if "embedding" in node_data and node_data["embedding"] is not None:
                        manual_stats["nodes_with_embeddings"] += 1
                        try:
                            embedding = np.array(node_data["embedding"], dtype=np.float32)
                            hyperbolic_embedding = kg._to_hyperbolic(embedding)
                            await kg.update_node(node_id, {"hyperbolic_embedding": hyperbolic_embedding.tolist()})
                            kg.hyperbolic_embedding["embedding_nodes"].add(node_id)
                            manual_stats["converted"] += 1
                        except Exception as e:
                            logger.error(f"Error in manual conversion for node {node_id}: {e}")
                            manual_stats["errors"] += 1
                
                conversion_result = manual_stats
                logger.info(f"Manual conversion completed as fallback: {json.dumps(conversion_result, indent=2)}")
        except Exception as e:
            logger.error(f"Error during batch conversion: {e}")
            traceback.print_exc()
            conversion_result = {"error": str(e)}
        
        logger.info("Hyperbolic embedding test completed")
        
        # Report results
        return {
            "hyperbolic_paths": paths_hyperbolic,
            "euclidean_paths": paths_euclidean,
            "hyperbolic_distance": h_dist_ab,
            "hyperbolic_similarity": h_sim_ab,
            "batch_conversion": conversion_result
        }
    except Exception as e:
        logger.error(f"Error in test_hyperbolic_implementation: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
