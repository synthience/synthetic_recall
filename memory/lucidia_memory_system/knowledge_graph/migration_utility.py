"""Migration Utility for Lucidia Knowledge Graph

This script helps migrate from the original monolithic knowledge graph
implementation to the new modular architecture. It handles data mapping
and ensures compatibility with embedding dimension handling.
"""

import logging
import json
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

# Set up path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import both old and new implementations
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph as LegacyKnowledgeGraph
from memory.lucidia_memory_system.knowledge_graph.core import LucidiaKnowledgeGraph as ModularKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MigrationUtility")

class KnowledgeGraphMigrator:
    """
    Migration utility for transferring data from the legacy knowledge graph
    to the new modular implementation, handling embedding dimension variations.
    """
    
    def __init__(self, legacy_graph=None, config=None):
        """
        Initialize the migrator.
        
        Args:
            legacy_graph: Optional existing legacy graph instance
            config: Configuration for the new graph
        """
        self.logger = logging.getLogger("KnowledgeGraphMigrator")
        self.legacy_graph = legacy_graph
        
        # Default configuration
        self.config = config or {
            "embedding": {
                "enable_hyperbolic": True,
                "hyperbolic_curvature": 1.0,
                "embedding_dimension": 768
            },
            "visualization": {
                "max_nodes": 200
            }
        }
        
        self.modular_graph = None
        self.logger.info("Knowledge Graph Migrator initialized")
    
    def align_embeddings(self, embedding, target_dim=768):
        """
        Align embeddings to the target dimension.
        
        Args:
            embedding: Source embedding vector
            target_dim: Target dimension (default 768)
            
        Returns:
            numpy.ndarray: Aligned embedding
        """
        if embedding is None:
            return np.zeros(target_dim, dtype=np.float32)
            
        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Error converting embedding to numpy array: {e}")
                return np.zeros(target_dim, dtype=np.float32)
        
        # Validate no NaN/Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            self.logger.warning("Embedding contains NaN or Inf values, replacing with zeros")
            return np.zeros(target_dim, dtype=np.float32)
        
        # Get current dimension
        current_dim = embedding.shape[0]
        
        # If dimensions match, return as is
        if current_dim == target_dim:
            return embedding
        
        # If current dimension is smaller, pad with zeros
        if current_dim < target_dim:
            padding = np.zeros(target_dim - current_dim, dtype=np.float32)
            return np.concatenate([embedding, padding])
        
        # If current dimension is larger, truncate
        return embedding[:target_dim]
    
    def initialize_modular_graph(self):
        """
        Initialize the new modular knowledge graph.
        
        Returns:
            ModularKnowledgeGraph: New modular graph instance
        """
        self.logger.info("Initializing modular knowledge graph")
        self.modular_graph = ModularKnowledgeGraph(config=self.config)
        return self.modular_graph
    
    async def migrate_nodes(self):
        """
        Migrate nodes from legacy graph to modular graph.
        
        Returns:
            int: Number of nodes migrated
        """
        if not self.legacy_graph or not self.modular_graph:
            self.logger.error("Both legacy and modular graphs must be initialized")
            return 0
        
        self.logger.info("Starting node migration")
        migrated_count = 0
        target_dim = self.config.get("embedding", {}).get("embedding_dimension", 768)
        
        # Get all nodes from legacy graph
        all_nodes = self.legacy_graph.get_all_nodes()
        
        for node_id, node_data in all_nodes.items():
            try:
                # Extract node attributes
                node_type = node_data.get("type", "general")
                attributes = {k: v for k, v in node_data.items() if k not in ["type", "embedding", "hyperbolic_embedding"]}
                domain = node_data.get("domain", "general_knowledge")
                
                # Handle embeddings
                if "embedding" in node_data:
                    # Align embedding to target dimension
                    aligned_embedding = self.align_embeddings(node_data["embedding"], target_dim)
                    attributes["embedding"] = aligned_embedding.tolist()
                
                # Add node to modular graph
                await self.modular_graph.add_node(node_id, node_type, attributes, domain)
                migrated_count += 1
                
                if migrated_count % 100 == 0:
                    self.logger.info(f"Migrated {migrated_count} nodes")
                    
            except Exception as e:
                self.logger.error(f"Error migrating node {node_id}: {e}")
        
        self.logger.info(f"Node migration complete: {migrated_count} nodes migrated")
        return migrated_count
    
    async def migrate_edges(self):
        """
        Migrate edges from legacy graph to modular graph.
        
        Returns:
            int: Number of edges migrated
        """
        if not self.legacy_graph or not self.modular_graph:
            self.logger.error("Both legacy and modular graphs must be initialized")
            return 0
        
        self.logger.info("Starting edge migration")
        migrated_count = 0
        
        # Get all edges from legacy graph
        all_edges = self.legacy_graph.get_all_edges()
        
        for source, targets in all_edges.items():
            for target, edges in targets.items():
                for edge_key, edge_data in edges.items():
                    try:
                        # Extract edge type and attributes
                        edge_type = edge_data.get("type", "related_to")
                        attributes = {k: v for k, v in edge_data.items() if k != "type"}
                        
                        # Add edge to modular graph
                        await self.modular_graph.add_edge(source, target, edge_type, attributes)
                        migrated_count += 1
                        
                        if migrated_count % 100 == 0:
                            self.logger.info(f"Migrated {migrated_count} edges")
                            
                    except Exception as e:
                        self.logger.error(f"Error migrating edge from {source} to {target}: {e}")
        
        self.logger.info(f"Edge migration complete: {migrated_count} edges migrated")
        return migrated_count
    
    async def run_migration(self):
        """
        Run the full migration process.
        
        Returns:
            dict: Migration statistics
        """
        start_time = datetime.now()
        self.logger.info(f"Starting migration at {start_time}")
        
        if not self.legacy_graph:
            self.logger.info("Initializing legacy knowledge graph")
            self.legacy_graph = LegacyKnowledgeGraph()
        
        # Initialize modular graph if not already done
        if not self.modular_graph:
            self.initialize_modular_graph()
            await self.modular_graph.initialize()
        
        # Migrate nodes and edges
        nodes_migrated = await self.migrate_nodes()
        edges_migrated = await self.migrate_edges()
        
        # Run post-migration setup
        if self.config.get("embedding", {}).get("enable_hyperbolic", False):
            self.logger.info("Converting nodes to hyperbolic embeddings")
            embedding_manager = self.modular_graph.module_registry.get_module("embedding_manager")
            await embedding_manager.convert_nodes_to_hyperbolic()
        
        # Run health check
        self.logger.info("Running health check")
        maintenance_manager = self.modular_graph.module_registry.get_module("maintenance_manager")
        health_result = await maintenance_manager.run_health_check()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile statistics
        stats = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "nodes_migrated": nodes_migrated,
            "edges_migrated": edges_migrated,
            "health_check": health_result
        }
        
        self.logger.info(f"Migration completed in {duration:.2f} seconds")
        self.logger.info(f"Migrated {nodes_migrated} nodes and {edges_migrated} edges")
        
        return stats

async def main():
    """Main function to run the migration utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate from legacy to modular knowledge graph')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--dump', type=str, help='Path to save migration statistics JSON')
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return 1
    
    # Initialize migrator
    migrator = KnowledgeGraphMigrator(config=config)
    
    # Run migration
    stats = await migrator.run_migration()
    
    # Save statistics if requested
    if args.dump:
        try:
            with open(args.dump, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Migration statistics saved to {args.dump}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
    
    return 0

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
