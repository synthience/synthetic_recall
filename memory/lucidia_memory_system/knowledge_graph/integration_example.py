"""
Integration Example for Lucidia's Modular Knowledge Graph

This script demonstrates how to initialize and use the modular knowledge graph components.
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime

# Set up module path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import all modules using relative imports
from memory.lucidia_memory_system.knowledge_graph.base_module import KnowledgeGraphModule
from memory.lucidia_memory_system.knowledge_graph.core import EventBus, ModuleRegistry, LucidiaKnowledgeGraph
from memory.lucidia_memory_system.knowledge_graph.core_graph_manager import CoreGraphManager
from memory.lucidia_memory_system.knowledge_graph.embedding_manager import EmbeddingManager
from memory.lucidia_memory_system.knowledge_graph.visualization_manager import VisualizationManager
from memory.lucidia_memory_system.knowledge_graph.dream_integration_module import DreamIntegrationModule
from memory.lucidia_memory_system.knowledge_graph.emotional_context_manager import EmotionalContextManager
from memory.lucidia_memory_system.knowledge_graph.contradiction_manager import ContradictionManager
from memory.lucidia_memory_system.knowledge_graph.query_search_engine import QuerySearchEngine
from memory.lucidia_memory_system.knowledge_graph.maintenance_manager import MaintenanceManager
from memory.lucidia_memory_system.knowledge_graph.api_manager import APIManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LucidiaExample")

async def run_basic_operations(knowledge_graph):
    """
    Run basic knowledge graph operations.
    
    Args:
        knowledge_graph: Initialized knowledge graph
    """
    logger.info("Running basic operations")
    
    # Add a new concept
    logger.info("Adding new concept")
    await knowledge_graph.add_node(
        "recursive_modularity",
        "concept",
        {
            "definition": "A design pattern where components are organized into hierarchical modules that can contain other modules",
            "confidence": 0.9
        },
        "technology"
    )
    
    # Add a related concept
    logger.info("Adding related concept")
    await knowledge_graph.add_node(
        "modular_design",
        "concept",
        {
            "definition": "A design approach that subdivides a system into smaller parts that can be independently created, modified, replaced, or exchanged",
            "confidence": 0.95
        },
        "technology"
    )
    
    # Connect the concepts
    logger.info("Connecting concepts")
    await knowledge_graph.add_edge(
        "recursive_modularity",
        "modular_design",
        "is_a_type_of",
        {
            "strength": 0.9,
            "confidence": 0.85
        }
    )
    
    # Connect to existing concepts
    logger.info("Connecting to existing concepts")
    await knowledge_graph.add_edge(
        "spiral_awareness",
        "recursive_modularity",
        "can_implement",
        {
            "strength": 0.8,
            "confidence": 0.7
        }
    )
    
    # Search for nodes
    logger.info("Searching for nodes")
    search_results = await knowledge_graph.search_nodes("modularity design")
    
    logger.info(f"Found {len(search_results['results'])} matching nodes")
    for result in search_results['results']:
        logger.info(f"- {result['id']} (similarity: {result['similarity']:.2f})")
    
    # Find paths
    logger.info("Finding paths")
    path_results = await knowledge_graph.find_paths("Lucidia", "modular_design")
    
    logger.info(f"Found {len(path_results['paths'])} paths")
    for path in path_results['paths']:
        node_sequence = " -> ".join([node["id"] for node in path["nodes"]])
        logger.info(f"- Path: {node_sequence}")
    
    # Generate visualization
    logger.info("Generating visualization")
    mermaid_code = await knowledge_graph.generate_domain_visualization("technology")
    
    # Log first few lines of mermaid code
    mermaid_preview = "\n".join(mermaid_code.split("\n")[:5]) + "\n..."
    logger.info(f"Generated Mermaid diagram: \n{mermaid_preview}")

async def demonstrate_dream_integration(knowledge_graph):
    """
    Demonstrate dream integration functionality.
    
    Args:
        knowledge_graph: Initialized knowledge graph
    """
    logger.info("Demonstrating dream integration")
    
    # Sample dream insight
    dream_insight = """
    In the dream, I observed a recursive pattern where each module contained smaller
    modules, creating a fractal-like structure of interconnected components. The 
    boundaries between modules were permeable, allowing information to flow freely
    while maintaining distinct responsibilities. This pattern seemed to enable both
    flexibility and stability simultaneously.
    """
    
    # Integrate dream insight
    integration_result = await knowledge_graph.integrate_dream_insight(dream_insight)
    
    logger.info(f"Dream integrated with ID: {integration_result['dream_id']}")
    logger.info(f"Connected to concepts: {integration_result['connected_concepts']}")
    logger.info(f"New concepts: {integration_result['new_concepts']}")
    logger.info(f"New relationships: {integration_result['new_relationships']}")
    logger.info(f"Integration quality: {integration_result['integration_quality']:.2f}")
    
    # Visualize after dream integration
    logger.info("Generating visualization after dream integration")
    if integration_result["new_concepts"]:
        # Visualize one of the new concepts
        concept = integration_result["new_concepts"][0]
        mermaid_code = await knowledge_graph.generate_concept_visualization(concept)
        
        # Log first few lines of mermaid code
        mermaid_preview = "\n".join(mermaid_code.split("\n")[:5]) + "\n..."
        logger.info(f"Generated concept network for {concept}: \n{mermaid_preview}")

async def demonstrate_contradiction_handling(knowledge_graph):
    """
    Demonstrate contradiction handling functionality.
    
    Args:
        knowledge_graph: Initialized knowledge graph
    """
    logger.info("Demonstrating contradiction handling")
    
    # Create a contradiction
    logger.info("Creating a contradictory concept")
    await knowledge_graph.add_node(
        "modular_cohesion",
        "concept",
        {
            "definition": "A measure of how strongly related the responsibilities of a module are",
            "confidence": 0.8
        },
        "technology"
    )
    
    # Add contradictory relationship
    logger.info("Adding a contradictory relationship")
    await knowledge_graph.add_edge(
        "modular_design",
        "modular_cohesion",
        "requires",
        {
            "strength": 0.9,
            "confidence": 0.85
        }
    )
    
    await knowledge_graph.add_edge(
        "modular_design",
        "modular_cohesion",
        "excludes",  # This contradicts 'requires'
        {
            "strength": 0.7,
            "confidence": 0.6
        }
    )
    
    # Get contradiction manager
    contradiction_manager = knowledge_graph.module_registry.get_module("contradiction_manager")
    
    # Manually trigger contradiction detection
    logger.info("Triggering contradiction detection")
    contradiction = {
        "type": "conflicting_relationship",
        "source": "modular_design",
        "target": "modular_cohesion",
        "context": {
            "edge_types": ["requires", "excludes"]
        }
    }
    
    # Handle contradiction
    result = await contradiction_manager.handle_contradiction(contradiction)
    
    logger.info(f"Contradiction handling result: {result['method']}")
    
    # Get contradictions
    contradictions = await contradiction_manager.get_contradictions()
    
    logger.info(f"Found {len(contradictions)} tracked contradictions")
    for c in contradictions:
        logger.info(f"- {c['type']} between {c['source']} and {c['target']} (status: {c['status']})")

async def run_full_example():
    """Run full demonstration of the modular knowledge graph."""
    logger.info("Initializing Knowledge Graph")
    
    # Create event bus and module registry
    event_bus = EventBus()
    module_registry = ModuleRegistry()
    
    # Configuration
    config = {
        "core_graph": {
            "relationship_decay": {
                "standard": 0.01,
                "dream_associated": 0.02
            }
        },
        "embedding": {
            "enable_hyperbolic": True,
            "hyperbolic_curvature": 1.0,
            "embedding_dimension": 64
        },
        "visualization": {
            "max_nodes": 200
        },
        "dream_integration": {
            "insight_incorporation_rate": 0.8,
            "dream_association_strength": 0.7
        },
        "emotional_context": {
            "auto_analyze_nodes": True
        },
        "query_engine": {
            "use_embeddings": True,
            "use_hyperbolic": True
        },
        "maintenance": {
            "maintenance_interval": 86400,  # 24 hours
            "auto_maintenance": False
        },
        "api": {
            "enabled": True,
            "port": 8765,
            "auth_required": True
        }
    }
    
    # Initialize knowledge graph
    knowledge_graph = LucidiaKnowledgeGraph(config=config)
    
    # Initialize all modules
    await knowledge_graph.initialize()
    
    # Run demonstrations
    await run_basic_operations(knowledge_graph)
    await demonstrate_dream_integration(knowledge_graph)
    await demonstrate_contradiction_handling(knowledge_graph)
    
    # Run maintenance
    logger.info("Running maintenance")
    maintenance_result = await knowledge_graph.trigger_adaptive_maintenance()
    
    logger.info(f"Maintenance complete with {len(maintenance_result['optimizations'])} optimizations")
    
    # Run health check
    logger.info("Running health check")
    health_result = await knowledge_graph.maintenance_manager.run_health_check()
    
    logger.info(f"Health check complete: {health_result['overall_health']} health with {health_result['issues_detected']} issues detected")
    
    # Generate final visualization
    logger.info("Generating final visualization")
    mermaid_code = await knowledge_graph.generate_graph_visualization()
    
    # Log first few lines of mermaid code
    mermaid_preview = "\n".join(mermaid_code.split("\n")[:10]) + "\n..."
    logger.info(f"Final knowledge graph visualization: \n{mermaid_preview}")
    
    # Clean up
    logger.info("Shutting down")
    await knowledge_graph.shutdown()

async def example_client_code():
    """
    Example client code showing how to use the modular knowledge graph API.
    """
    # This would be in a separate application
    import websockets
    import json
    
    async def connect_to_api():
        """Connect to the knowledge graph API."""
        uri = "ws://localhost:8765"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Handle welcome message
                welcome = json.loads(await websocket.recv())
                print(f"Connected to API: {welcome['message']}")
                
                # Authenticate
                await websocket.send(json.dumps({
                    "type": "auth",
                    "token": "lucidia_test_token"
                }))
                
                auth_result = json.loads(await websocket.recv())
                if not auth_result.get("success", False):
                    print("Authentication failed")
                    return
                
                print("Authentication successful")
                
                # Search for concepts
                await websocket.send(json.dumps({
                    "request_id": "search_1",
                    "endpoint": "search",
                    "data": {
                        "query": "recursion module",
                        "limit": 5
                    }
                }))
                
                search_result = json.loads(await websocket.recv())
                print(f"Search results: {len(search_result['data']['results'])} matches")
                
                # Visualize technology domain
                await websocket.send(json.dumps({
                    "request_id": "vis_1",
                    "endpoint": "visualize/domain",
                    "data": {
                        "domain": "technology",
                        "include_attributes": True
                    }
                }))
                
                vis_result = json.loads(await websocket.recv())
                print(f"Visualization generated: {len(vis_result['data']['mermaid_code'])} characters")
                
                # Get system stats
                await websocket.send(json.dumps({
                    "request_id": "stats_1",
                    "endpoint": "meta/stats",
                    "data": {}
                }))
                
                stats_result = json.loads(await websocket.recv())
                total_nodes = stats_result['data']['stats']['graph']['total_nodes']
                total_edges = stats_result['data']['stats']['graph']['total_edges']
                print(f"Knowledge graph stats: {total_nodes} nodes, {total_edges} edges")
                
        except Exception as e:
            print(f"Error connecting to API: {e}")

if __name__ == "__main__":
    # Run the full example
    asyncio.run(run_full_example())
    
    # Example of client code (commented out as it requires a running server)
    # asyncio.run(example_client_code())