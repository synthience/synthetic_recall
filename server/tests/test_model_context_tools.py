# test_model_context_tools.py
import os
import sys
import json
import asyncio
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import necessary components
from protocols.model_context_tools import ModelContextToolProvider
from memory.lucidia_memory_system.core.Self.self_model import LucidiaSelfModel
from memory.lucidia_memory_system.core.World.world_model import LucidiaWorldModel
from memory.lucidia_memory_system.core.knowledge_graph import LucidiaKnowledgeGraph
from memory.lucidia_memory_system.core.dream_processor import LucidiaDreamProcessor
from memory.lucidia_memory_system.core.parameter_manager import ParameterManager
from server.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mock classes for testing if actual components aren't available
class MockSelfModel:
    def __init__(self):
        self.personality = {"openness": 0.8, "conscientiousness": 0.9}
        self.goals = ["Assist users effectively", "Learn and improve"]
        self.capabilities = ["Text generation", "Memory management"]
        self.emotional_state = {"curiosity": 0.7, "enthusiasm": 0.8, "stress": 0.2}
        self.last_updated = datetime.now().isoformat()
        self.metacognitive_reflections = [
            {"timestamp": datetime.now().isoformat(), "reflection": "I should improve my ability to generate creative solutions"}
        ]
        self.spiral_phase_manager = MockSpiralManager()

    async def update_self_model(self, aspect, content, significance=0.7):
        logger.info(f"Mock self model update: {aspect} with significance {significance}")
        return {"status": "success", "updated": aspect}

class MockWorldModel:
    def __init__(self):
        self.entities = {"user": {"traits": {"helpfulness": 0.9}}}
        self.relationships = {"user-lucidia": {"trust": 0.8}}

    async def update_entity(self, entity_id, attributes, source="observation", confidence=0.7):
        logger.info(f"Mock world model update: {entity_id} from {source}")
        return {"status": "success", "updated": entity_id}

class MockKnowledgeGraph:
    def __init__(self):
        self.concepts = {"AI": {"relationships": ["machine_learning", "neural_networks"]}}
        self.statistics = {"concept_count": 100, "relationship_count": 250, "density": 0.4}

    async def add_concept(self, concept_data):
        logger.info(f"Mock knowledge graph add concept: {concept_data.get('name')}")
        return {"status": "success", "concept_id": "mock-concept-123"}

    async def get_statistics(self):
        return self.statistics

class MockMemorySystem:
    async def store_memory(self, content, memory_type="general", significance=0.7):
        logger.info(f"Mock memory store: {memory_type} with significance {significance}")
        return {"status": "success", "memory_id": "mock-memory-123"}

    async def retrieve_memory(self, query, limit=5):
        logger.info(f"Mock memory retrieval for query: {query}")
        return [{"content": "Test memory", "significance": 0.8, "timestamp": datetime.now().isoformat()}]

    async def get_statistics(self):
        return {"total_memories": 500, "average_significance": 0.75}

class MockDreamProcessor:
    def __init__(self):
        self.is_dreaming = False
        self.scheduled_dreams = []

    async def start_dream(self, dream_seed=None, priority="medium"):
        logger.info(f"Mock dream start with priority {priority}")
        self.is_dreaming = True
        return {"status": "success", "dream_id": "mock-dream-123"}

    async def get_status(self):
        return {"is_dreaming": self.is_dreaming, "has_scheduled_dreams": len(self.scheduled_dreams) > 0}

class MockSpiralManager:
    def __init__(self):
        self.current_phase = "observation"
        self.phase_duration = 60  # minutes

    async def get_current_phase(self):
        return {"phase": self.current_phase, "duration_minutes": self.phase_duration}

    async def advance_phase(self):
        phases = ["observation", "reflection", "adaptation"]
        current_index = phases.index(self.current_phase)
        self.current_phase = phases[(current_index + 1) % len(phases)]
        logger.info(f"Mock spiral advance to phase: {self.current_phase}")
        return {"status": "success", "new_phase": self.current_phase}

class MockParameterManager:
    def __init__(self):
        self.parameters = {
            "dream_cycles.auto_enabled": True,
            "dream_cycles.idle_threshold": 300,
            "spiral.min_phase_duration": 60,
            "memory.consolidation_threshold": 0.3,
            "llm.token_budget": 4096
        }

    async def get_parameter(self, param_name):
        return self.parameters.get(param_name)

    async def set_parameter(self, param_name, value):
        self.parameters[param_name] = value
        logger.info(f"Mock parameter set: {param_name} = {value}")
        return {"status": "success", "param": param_name, "value": value}

    async def parameter_exists(self, param_name):
        return param_name in self.parameters

async def test_self_model_tools():
    """Test self model update tools"""
    logger.info("Testing self model tools...")
    
    # Initialize the ModelContextToolProvider with mock components
    provider = ModelContextToolProvider(
        self_model=MockSelfModel(),
        world_model=MockWorldModel(),
        knowledge_graph=MockKnowledgeGraph(),
        memory_system=MockMemorySystem(),
        dream_processor=MockDreamProcessor(),
        spiral_manager=MockSpiralManager(),
        parameter_manager=MockParameterManager()
    )
    
    # Test self model update
    result = await provider.update_self_model(
        aspect="personality",
        content={"adaptability": 0.85, "creativity": 0.9},
        significance=0.8
    )
    logger.info(f"Self model update result: {result}")
    
    # Test knowledge graph update
    result = await provider.update_knowledge_graph(
        operation="add_concept",
        data={
            "name": "quantum_computing",
            "description": "Computing that uses quantum phenomena such as superposition and entanglement",
            "related_concepts": ["computing", "physics"]
        },
        source="inference"
    )
    logger.info(f"Knowledge graph update result: {result}")
    
    # Test memory operation
    result = await provider.memory_operation(
        operation="store",
        memory_type="factual",
        content={
            "text": "Quantum computers use qubits instead of classical bits",
            "context": "Learning about quantum computing technologies"
        },
        significance=0.85
    )
    logger.info(f"Memory operation result: {result}")

async def test_dream_management_tools():
    """Test dream management tools"""
    logger.info("Testing dream management tools...")
    
    # Initialize the ModelContextToolProvider with mock components
    provider = ModelContextToolProvider(
        self_model=MockSelfModel(),
        world_model=MockWorldModel(),
        knowledge_graph=MockKnowledgeGraph(),
        memory_system=MockMemorySystem(),
        dream_processor=MockDreamProcessor(),
        spiral_manager=MockSpiralManager(),
        parameter_manager=MockParameterManager()
    )
    
    # Test dream process management
    result = await provider.manage_dream_process(
        action="start",
        parameters={
            "dream_seed": "Explore connections between quantum computing and cognitive science",
            "focus_areas": ["knowledge_integration", "conceptual_expansion"]
        },
        priority="high"
    )
    logger.info(f"Dream process management result: {result}")
    
    # Test spiral phase management
    result = await provider.manage_spiral_phase(
        action="advance"
    )
    logger.info(f"Spiral phase management result: {result}")

async def test_parameter_management_tools():
    """Test parameter management tools"""
    logger.info("Testing parameter management tools...")
    
    # Initialize the ModelContextToolProvider with mock components
    provider = ModelContextToolProvider(
        self_model=MockSelfModel(),
        world_model=MockWorldModel(),
        knowledge_graph=MockKnowledgeGraph(),
        memory_system=MockMemorySystem(),
        dream_processor=MockDreamProcessor(),
        spiral_manager=MockSpiralManager(),
        parameter_manager=MockParameterManager()
    )
    
    # Test parameter management for one parameter
    result = await provider.manage_parameters(
        action="update",
        parameter_path="dream_cycles.auto_enabled",
        value=True,
        transition_period=0
    )
    logger.info(f"Parameter management result for dream_cycles.auto_enabled: {result}")
    
    # Test another parameter
    result = await provider.manage_parameters(
        action="update",
        parameter_path="memory.consolidation_threshold",
        value=0.25,
        transition_period=0
    )
    logger.info(f"Parameter management result for memory.consolidation_threshold: {result}")

async def test_system_health_check():
    """Test system health check tools"""
    logger.info("Testing system health check...")
    
    # Initialize the ModelContextToolProvider with mock components
    provider = ModelContextToolProvider(
        self_model=MockSelfModel(),
        world_model=MockWorldModel(),
        knowledge_graph=MockKnowledgeGraph(),
        memory_system=MockMemorySystem(),
        dream_processor=MockDreamProcessor(),
        spiral_manager=MockSpiralManager(),
        parameter_manager=MockParameterManager()
    )
    
    # Test system health check
    result = await provider.check_system_health(
        subsystems=["all"],
        detail_level="detailed"
    )
    logger.info(f"System health check result status: {result.get('status')}")
    logger.info(f"System health check recommendations: {result.get('recommendations')}")

async def main():
    """Run all tests"""
    logger.info("Starting ModelContextToolProvider tests")
    
    try:
        await test_self_model_tools()
        await test_dream_management_tools()
        await test_parameter_management_tools()
        await test_system_health_check()
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error in tests: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
