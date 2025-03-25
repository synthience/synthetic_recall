import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

async def convert_nodes_to_hyperbolic(self, node_types: Optional[List[str]] = None, 
                                      domains: Optional[List[str]] = None,
                                      batch_size: int = 100) -> Dict[str, Any]:
    """
    Batch convert regular embeddings to hyperbolic embeddings for all nodes in the graph.
    
    Args:
        node_types: Optional list of node types to convert, all if None
        domains: Optional list of domains to convert, all if None
        batch_size: Number of nodes to process in each batch
        
    Returns:
        Dict with statistics about the conversion process
    """
    if not self.hyperbolic_embedding["enabled"]:
        self.logger.warning("Hyperbolic embeddings not enabled, skipping conversion")
        return {"error": "Hyperbolic embeddings not enabled", "converted": 0}

    self.logger.info(f"Starting batch conversion of embeddings to hyperbolic space")

    stats = {
        "total_nodes": 0,
        "nodes_with_embeddings": 0,
        "converted": 0,
        "errors": 0,
        "start_time": datetime.now().isoformat()
    }

    try:
        # Get all nodes that match the criteria
        nodes_to_process = []
        for node_id, node_data in self.graph.nodes(data=True):
            stats["total_nodes"] += 1
            
            # Skip if node already has hyperbolic embedding
            if "hyperbolic_embedding" in node_data and node_data["hyperbolic_embedding"] is not None:
                continue
                
            # Skip if node doesn't have a regular embedding
            if "embedding" not in node_data or node_data["embedding"] is None:
                continue
                
            # Filter by node type if specified
            if node_types and ("type" not in node_data or node_data["type"] not in node_types):
                continue
                
            # Filter by domain if specified
            if domains and ("domain" not in node_data or node_data["domain"] not in domains):
                continue
                
            stats["nodes_with_embeddings"] += 1
            nodes_to_process.append((node_id, node_data))
        
        # Process nodes in batches
        batch_count = 0
        for i in range(0, len(nodes_to_process), batch_size):
            batch = nodes_to_process[i:i+batch_size]
            batch_count += 1
            
            # Process each node in the batch
            for node_id, node_data in batch:
                try:
                    # Get regular embedding
                    embedding = node_data["embedding"]
                    if isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)
                        
                    # Convert to hyperbolic space
                    hyperbolic_embedding = self._to_hyperbolic(embedding)
                    
                    # Update node with hyperbolic embedding
                    await self.update_node(node_id, {"hyperbolic_embedding": hyperbolic_embedding.tolist()})
                    
                    # Add to set of nodes with hyperbolic embeddings
                    self.hyperbolic_embedding["embedding_nodes"].add(node_id)
                    
                    stats["converted"] += 1
                except Exception as e:
                    self.logger.error(f"Error converting embedding for node {node_id}: {e}")
                    stats["errors"] += 1
            
            self.logger.info(f"Processed batch {batch_count}: {len(batch)} nodes")
        
        stats["end_time"] = datetime.now().isoformat()
        stats["success"] = True
        
        self.logger.info(f"Converted {stats['converted']} nodes to hyperbolic space with {stats['errors']} errors")
        return stats
        
    except Exception as e:
        self.logger.error(f"Error during batch conversion to hyperbolic space: {e}")
        stats["error"] = str(e)
        stats["success"] = False
        stats["end_time"] = datetime.now().isoformat()
        return stats
