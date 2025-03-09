#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the ReflectionEngine and Dream Reports integration.

This script tests the integration between the ReflectionEngine, Knowledge Graph,
and Dream Reports, ensuring that the system can generate, refine, and retrieve
dream reports properly.
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary components
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph
from memory.lucidia_memory_system.core.integration import MemoryIntegration
from memory.lucidia_memory_system.core.reflection_engine import ReflectionEngine
from memory.lucidia_memory_system.core.dream_structures import DreamReport, DreamFragment
from server.memory_client import EnhancedMemoryClient
from server.llm_pipeline import LocalLLMPipeline
from voice_core.config.config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
STORAGE_PATH = os.getenv('STORAGE_PATH', './data')
TENSOR_SERVER_URL = os.getenv('TENSOR_SERVER_URL', 'ws://localhost:5001')
HPC_SERVER_URL = os.getenv('HPC_SERVER_URL', 'ws://localhost:5005')
LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT', 'http://localhost:8000/v1')
LLM_MODEL = os.getenv('LLM_MODEL', 'local-model')


async def initialize_components():
    """Initialize all necessary components for testing."""
    # Ensure storage directories exist
    os.makedirs(f"{STORAGE_PATH}/knowledge_graph", exist_ok=True)
    os.makedirs(f"{STORAGE_PATH}/reflection", exist_ok=True)
    
    # Initialize Knowledge Graph
    logger.info("Initializing knowledge graph...")
    knowledge_graph = LucidiaKnowledgeGraph(config={
        "storage_directory": f"{STORAGE_PATH}/knowledge_graph"
    })
    
    # Initialize Memory Client
    logger.info("Initializing memory client...")
    memory_client = EnhancedMemoryClient(config={
        "tensor_server_url": TENSOR_SERVER_URL,
        "hpc_server_url": HPC_SERVER_URL,
        "ping_interval": 30,
        "max_retries": 3,
        "retry_delay": 1
    })
    await memory_client.initialize()
    
    # Initialize LLM Service
    logger.info("Initializing LLM service...")
    llm_config = LLMConfig(
        api_endpoint=LLM_API_ENDPOINT,
        model=LLM_MODEL,
        system_prompt="You are Lucidia's reflection system, analyzing memories and generating insights.",
        temperature=0.7,
        max_tokens=1024,
        timeout=30
    )
    llm_service = LocalLLMPipeline(config=llm_config)
    await llm_service.initialize()
    
    # Initialize Memory Integration
    memory_integration = MemoryIntegration(
        memory_client=memory_client,
        knowledge_graph=knowledge_graph
    )
    
    # Initialize Reflection Engine
    logger.info("Initializing reflection engine...")
    reflection_engine = ReflectionEngine(
        knowledge_graph=knowledge_graph,
        memory_integration=memory_integration,
        llm_service=llm_service,
        config={
            "storage_path": f"{STORAGE_PATH}/reflection",
            "domain": "synthien_studies"
        }
    )
    
    return {
        "knowledge_graph": knowledge_graph,
        "memory_client": memory_client,
        "llm_service": llm_service,
        "memory_integration": memory_integration,
        "reflection_engine": reflection_engine
    }


async def create_test_memories(memory_client: EnhancedMemoryClient, count: int = 5) -> List[Dict[str, Any]]:
    """Create test memories for the reflection engine."""
    logger.info(f"Creating {count} test memories...")
    
    memories = []
    topics = ["artificial intelligence", "consciousness", "learning", "creativity", "problem-solving"]
    
    for i in range(count):
        memory_id = f"test_memory_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
        memory_content = f"Reflection on {topics[i % len(topics)]}: Understanding the nature of {topics[i % len(topics)]} "
        memory_content += f"requires deep analysis and introspection. This is test memory {i}."
        
        # Create memory
        memory = {
            "id": memory_id,
            "content": memory_content,
            "created_at": datetime.now().isoformat(),
            "significance": 0.8,
            "source": "test",
            "metadata": {
                "topic": topics[i % len(topics)],
                "test": True
            }
        }
        
        # Store memory
        await memory_client.store_memory(memory)
        memories.append(memory)
    
    return memories


async def test_generate_report(components: Dict[str, Any], memories: List[Dict[str, Any]]):
    """Test generating a dream report from memories."""
    reflection_engine = components["reflection_engine"]
    knowledge_graph = components["knowledge_graph"]
    
    logger.info("Testing report generation...")
    
    # Generate a report
    report = await reflection_engine.generate_report(
        memories=memories,
        domain="synthien_studies",
        title="Test Dream Report",
        description="A test report generated for integration testing"
    )
    
    logger.info(f"Generated report with ID: {report.report_id}")
    logger.info(f"Report title: {report.title}")
    logger.info(f"Report has {len(report.insight_ids)} insights, {len(report.question_ids)} questions, "
               f"{len(report.hypothesis_ids)} hypotheses, and {len(report.counterfactual_ids)} counterfactuals")
    
    # Verify report was integrated into the knowledge graph
    assert knowledge_graph.has_node(report.report_id), "Report node not found in knowledge graph"
    
    # Verify fragments were integrated
    for fragment_id in report.insight_ids + report.question_ids + report.hypothesis_ids + report.counterfactual_ids:
        assert knowledge_graph.has_node(fragment_id), f"Fragment node {fragment_id} not found in knowledge graph"
    
    return report


async def test_refine_report(components: Dict[str, Any], report: DreamReport, new_memories: List[Dict[str, Any]]):
    """Test refining an existing dream report with new evidence."""
    reflection_engine = components["reflection_engine"]
    knowledge_graph = components["knowledge_graph"]
    
    logger.info(f"Testing report refinement for report {report.report_id}...")
    
    # Get the number of fragments before refinement
    fragment_count_before = len(report.insight_ids) + len(report.question_ids) + \
                           len(report.hypothesis_ids) + len(report.counterfactual_ids)
    
    # Refine the report with new evidence
    result = await reflection_engine.refine_report(
        report=report,
        new_evidence_ids=[memory["id"] for memory in new_memories]
    )
    
    # Get the updated report
    updated_report = result.get("updated_report")
    
    if updated_report:
        logger.info(f"Refined report with ID: {updated_report.report_id}")
        
        # Get the number of fragments after refinement
        fragment_count_after = len(updated_report.insight_ids) + len(updated_report.question_ids) + \
                              len(updated_report.hypothesis_ids) + len(updated_report.counterfactual_ids)
        
        logger.info(f"Report now has {len(updated_report.insight_ids)} insights, {len(updated_report.question_ids)} questions, "
                   f"{len(updated_report.hypothesis_ids)} hypotheses, and {len(updated_report.counterfactual_ids)} counterfactuals")
        
        # Verify new fragments were integrated
        for fragment_id in updated_report.insight_ids + updated_report.question_ids + \
                          updated_report.hypothesis_ids + updated_report.counterfactual_ids:
            assert knowledge_graph.has_node(fragment_id), f"Fragment node {fragment_id} not found in knowledge graph"
        
        # Verify new memories were added to participating memories
        for memory_id in [memory["id"] for memory in new_memories]:
            assert memory_id in updated_report.participating_memory_ids, f"Memory {memory_id} not in participating memories"
        
        return updated_report
    else:
        logger.error("Failed to refine report")
        return None


async def test_knowledge_graph_integration(components: Dict[str, Any], report: DreamReport):
    """Test the integration between dream reports and the knowledge graph."""
    knowledge_graph = components["knowledge_graph"]
    
    logger.info(f"Testing knowledge graph integration for report {report.report_id}...")
    
    # Verify the report node exists
    assert knowledge_graph.has_node(report.report_id), "Report node not found in knowledge graph"
    
    # Get the report node
    report_node = knowledge_graph.get_node(report.report_id)
    assert report_node["type"] == "dream_report", "Node is not a dream report"
    
    # Verify connections to fragments
    connected_fragments = knowledge_graph.get_connected_nodes(
        report.report_id,
        edge_types=["contains"],
        direction="outbound"
    )
    
    logger.info(f"Report is connected to {len(connected_fragments)} fragments in the knowledge graph")
    
    # Verify connections to memories
    connected_memories = knowledge_graph.get_connected_nodes(
        report.report_id,
        edge_types=["based_on"],
        direction="outbound"
    )
    
    logger.info(f"Report is connected to {len(connected_memories)} memories in the knowledge graph")
    
    # Verify connections to concepts
    connected_concepts = knowledge_graph.get_connected_nodes(
        report.report_id,
        edge_types=["references"],
        node_types=["concept", "entity"],
        direction="outbound"
    )
    
    logger.info(f"Report is connected to {len(connected_concepts)} concepts in the knowledge graph")
    
    # Print some connected concepts for verification
    if connected_concepts:
        logger.info("Connected concepts:")
        for concept in connected_concepts[:5]:  # Show up to 5 concepts
            concept_node = knowledge_graph.get_node(concept)
            if concept_node and "attributes" in concept_node:
                definition = concept_node["attributes"].get("definition", "No definition")
                logger.info(f"  - {concept}: {definition[:100]}...")
    
    return {
        "connected_fragments": connected_fragments,
        "connected_memories": connected_memories,
        "connected_concepts": connected_concepts
    }


async def main():
    """Main test function."""
    try:
        logger.info("Starting ReflectionEngine integration test")
        
        # Initialize components
        components = await initialize_components()
        
        # Create test memories
        memories = await create_test_memories(components["memory_client"], count=5)
        
        # Test generating a report
        report = await test_generate_report(components, memories)
        
        # Create additional test memories for refinement
        new_memories = await create_test_memories(components["memory_client"], count=3)
        
        # Test refining the report
        updated_report = await test_refine_report(components, report, new_memories)
        
        # Test knowledge graph integration
        if updated_report:
            integration_results = await test_knowledge_graph_integration(components, updated_report)
        else:
            integration_results = await test_knowledge_graph_integration(components, report)
        
        logger.info("ReflectionEngine integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        if "memory_client" in components:
            await components["memory_client"].close()
        if "llm_service" in components:
            await components["llm_service"].close()


if __name__ == "__main__":
    asyncio.run(main())
