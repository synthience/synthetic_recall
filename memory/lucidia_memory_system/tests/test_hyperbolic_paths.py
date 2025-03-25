import asyncio
import json
import numpy as np
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.knowledge_graph import LucidiaKnowledgeGraph

async def test_find_paths():
    """Test the hyperbolic path finding functionality"""
    # Initialize a knowledge graph for testing
    kg = LucidiaKnowledgeGraph(
        graph_name="test_hyperbolic",
        config={
            "hyperbolic_embedding": {
                "enabled": True,
                "curvature": -1.0
            }
        },
        module_registry=None,  # Required parameter
        event_bus=None,       # Required parameter
        logger=logging.getLogger("test_hyperbolic")
    )
    
    print("Starting test_find_paths")
    
    # Create a simple graph
    await kg.add_node("A", {"type": "test"})
    await kg.add_node("B", {"type": "test"})
    await kg.add_node("C", {"type": "test"})
    await kg.add_node("D", {"type": "test"})
    
    await kg.add_edge("A", "B", {"confidence": 0.8})
    await kg.add_edge("B", "C", {"confidence": 0.9})
    await kg.add_edge("A", "D", {"confidence": 0.7})
    
    # Add embeddings
    a_emb = np.array([0.1, 0.2], dtype=np.float32)
    b_emb = np.array([0.3, 0.4], dtype=np.float32)
    c_emb = np.array([0.5, 0.6], dtype=np.float32)
    d_emb = np.array([0.7, 0.8], dtype=np.float32)
    
    # Add regular embeddings to all nodes
    await kg.set_node_attributes("A", {"embedding": a_emb.tolist()})
    await kg.set_node_attributes("B", {"embedding": b_emb.tolist()})
    await kg.set_node_attributes("C", {"embedding": c_emb.tolist()})
    await kg.set_node_attributes("D", {"embedding": d_emb.tolist()})
    
    # Convert to hyperbolic and add them
    a_hyp = kg._to_hyperbolic(a_emb)
    b_hyp = kg._to_hyperbolic(b_emb)
    c_hyp = kg._to_hyperbolic(c_emb)
    
    await kg.set_node_attributes("A", {"hyperbolic_embedding": a_hyp.tolist()})
    await kg.set_node_attributes("B", {"hyperbolic_embedding": b_hyp.tolist()})
    await kg.set_node_attributes("C", {"hyperbolic_embedding": c_hyp.tolist()})
    
    # Add nodes to hyperbolic embedding set
    kg.hyperbolic_embedding["embedding_nodes"].add("A")
    kg.hyperbolic_embedding["embedding_nodes"].add("B")
    kg.hyperbolic_embedding["embedding_nodes"].add("C")
    
    print("Graph setup complete. Testing find_paths with and without hyperbolic embeddings...")
    
    # Find paths with hyperbolic embeddings
    paths_hyperbolic = await kg.find_paths("A", "C", max_depth=2, min_confidence=0.5, use_hyperbolic=True)
    print(f"\nPaths with hyperbolic embeddings: {json.dumps(paths_hyperbolic, indent=2)}")
    
    # Find paths without hyperbolic embeddings
    paths_euclidean = await kg.find_paths("A", "C", max_depth=2, min_confidence=0.5, use_hyperbolic=False)
    print(f"\nPaths without hyperbolic embeddings: {json.dumps(paths_euclidean, indent=2)}")
    
    print("\nTest completed")
    return {"hyperbolic": paths_hyperbolic, "euclidean": paths_euclidean}

# Run the test
async def main():
    try:
        result = await test_find_paths()
        print("\nSummary:")
        print(f"Hyperbolic paths: {len(result['hyperbolic'].get('paths', []))}")
        print(f"Euclidean paths: {len(result['euclidean'].get('paths', []))}")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
