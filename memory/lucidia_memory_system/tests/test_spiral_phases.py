import sys
import os
import logging
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from core.spiral_phases import SpiralPhase, SpiralPhaseManager
from core.dream_processor import LucidiaDreamProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)


class TestSpiralPhases(unittest.TestCase):
    """Test cases for the SpiralPhase enumeration and related functionality"""
    
    def test_spiral_phase_enum(self):
        """Test that SpiralPhase enumeration has the expected values"""
        self.assertEqual(SpiralPhase.OBSERVATION.value, "observation")
        self.assertEqual(SpiralPhase.REFLECTION.value, "reflection")
        self.assertEqual(SpiralPhase.ADAPTATION.value, "adaptation")
        
    def test_phase_names(self):
        """Test that phase names are correct"""
        self.assertEqual(SpiralPhase.OBSERVATION.name, "OBSERVATION")
        self.assertEqual(SpiralPhase.REFLECTION.name, "REFLECTION")
        self.assertEqual(SpiralPhase.ADAPTATION.name, "ADAPTATION")


class TestSpiralPhaseManager(unittest.TestCase):
    """Test cases for the SpiralPhaseManager class"""
    
    def setUp(self):
        """Set up for each test"""
        self.self_model = MagicMock()
        self.spiral_manager = SpiralPhaseManager(self_model=self.self_model)
    
    def test_initialization(self):
        """Test spiral phase manager initialization"""
        # Initial phase should be OBSERVATION
        self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.OBSERVATION)
        
        # Configuration should be set for all phases
        for phase in SpiralPhase:
            self.assertIn(phase, self.spiral_manager.phase_config)
            
        # Check that each phase config has required parameters
        required_params = ['name', 'description', 'min_depth', 'max_depth', 
                          'min_creativity', 'max_creativity', 'transition_threshold',
                          'insight_weight', 'focus_areas', 'typical_duration']
        
        for phase in SpiralPhase:
            config = self.spiral_manager.phase_config[phase]
            for param in required_params:
                self.assertIn(param, config)
                
    def test_get_current_phase(self):
        """Test getting the current phase"""
        self.assertEqual(self.spiral_manager.get_current_phase(), SpiralPhase.OBSERVATION)
        
        # Change phase and verify
        self.spiral_manager.current_phase = SpiralPhase.REFLECTION
        self.assertEqual(self.spiral_manager.get_current_phase(), SpiralPhase.REFLECTION)
    
    def test_get_phase_params(self):
        """Test getting parameters for the current phase"""
        params = self.spiral_manager.get_phase_params()
        
        # Verify same parameters as in the config for OBSERVATION phase
        observation_config = self.spiral_manager.phase_config[SpiralPhase.OBSERVATION]
        self.assertEqual(params, observation_config)
        
        # Change phase and verify new parameters
        self.spiral_manager.current_phase = SpiralPhase.REFLECTION
        params = self.spiral_manager.get_phase_params()
        reflection_config = self.spiral_manager.phase_config[SpiralPhase.REFLECTION]
        self.assertEqual(params, reflection_config)
    
    def test_determine_next_phase(self):
        """Test determining the next phase in the cycle"""
        # Start in OBSERVATION, next should be REFLECTION
        self.spiral_manager.current_phase = SpiralPhase.OBSERVATION
        next_phase = self.spiral_manager._determine_next_phase()
        self.assertEqual(next_phase, SpiralPhase.REFLECTION)
        
        # REFLECTION → ADAPTATION
        self.spiral_manager.current_phase = SpiralPhase.REFLECTION
        next_phase = self.spiral_manager._determine_next_phase()
        self.assertEqual(next_phase, SpiralPhase.ADAPTATION)
        
        # ADAPTATION → OBSERVATION (cycle)
        self.spiral_manager.current_phase = SpiralPhase.ADAPTATION
        next_phase = self.spiral_manager._determine_next_phase()
        self.assertEqual(next_phase, SpiralPhase.OBSERVATION)
    
    def test_transition_phase(self):
        """Test the transition_phase method using insight significance"""
        # Start in OBSERVATION
        self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.OBSERVATION)
        
        # Set a high threshold that should not trigger transition
        original_threshold = self.spiral_manager.phase_config[SpiralPhase.OBSERVATION]['transition_threshold']
        self.spiral_manager.phase_config[SpiralPhase.OBSERVATION]['transition_threshold'] = 0.9
        
        # Low significance should not trigger transition
        transition_occurred = self.spiral_manager.transition_phase(0.5)
        self.assertFalse(transition_occurred)
        self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.OBSERVATION)
        
        # High significance should trigger transition
        transition_occurred = self.spiral_manager.transition_phase(1.0)
        self.assertTrue(transition_occurred)
        self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.REFLECTION)
        
        # Restore original threshold
        self.spiral_manager.phase_config[SpiralPhase.OBSERVATION]['transition_threshold'] = original_threshold
    
    def test_force_phase(self):
        """Test forcing the phase to a specific value"""
        self.spiral_manager.force_phase(SpiralPhase.ADAPTATION, "test_reason")
        self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.ADAPTATION)
        
        # Force to same phase should not add to history
        initial_history_len = len(self.spiral_manager.phase_history)
        self.spiral_manager.force_phase(SpiralPhase.ADAPTATION, "same_phase")
        self.assertEqual(len(self.spiral_manager.phase_history), initial_history_len)
        
        # Force to different phase should add to history
        self.spiral_manager.force_phase(SpiralPhase.OBSERVATION, "different_phase")
        self.assertEqual(len(self.spiral_manager.phase_history), initial_history_len + 1)
        self.assertEqual(self.spiral_manager.current_phase, SpiralPhase.OBSERVATION)
    
    def test_record_insight(self):
        """Test recording an insight in the current phase"""
        # Create a test insight
        test_insight = {
            "text": "Test insight",
            "significance": 0.85,
            "timestamp": datetime.now().isoformat()
        }
        
        # Record insight
        initial_insight_count = len(self.spiral_manager.phase_stats[SpiralPhase.OBSERVATION]['insights'])
        self.spiral_manager.record_insight(test_insight)
        
        # Verify insight was recorded
        insights = self.spiral_manager.phase_stats[SpiralPhase.OBSERVATION]['insights']
        self.assertEqual(len(insights), initial_insight_count + 1)
        
        # Test with low significance (should not be recorded)
        low_insight = {
            "text": "Low significance insight",
            "significance": 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
        self.spiral_manager.record_insight(low_insight)
        self.assertEqual(len(insights), initial_insight_count + 1)  # Still just one recorded
    
    def test_get_phase_stats(self):
        """Test getting statistics for a specific phase"""
        stats = self.spiral_manager.get_phase_stats(SpiralPhase.OBSERVATION)
        
        # Check that stats have expected fields
        self.assertIn('total_time', stats)
        self.assertIn('transitions', stats)
        self.assertIn('insights', stats)
        self.assertIn('last_entered', stats)
        
        # Check default values
        self.assertEqual(stats['transitions'], 1)  # Initial transition during initialization
        self.assertEqual(len(stats['insights']), 0)
    
    def test_get_phase_history(self):
        """Test getting the phase transition history"""
        # History should have at least one entry (initialization)
        history = self.spiral_manager.get_phase_history()
        self.assertGreaterEqual(len(history), 1)
        
        # Add more transitions and check history
        self.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test1")
        self.spiral_manager.force_phase(SpiralPhase.ADAPTATION, "test2")
        
        history = self.spiral_manager.get_phase_history(limit=2)
        self.assertEqual(len(history), 2)
        
        # Check most recent entry
        latest = history[-1]
        self.assertEqual(latest['phase'], SpiralPhase.ADAPTATION.value)
        self.assertEqual(latest['reason'], "test2")
    
    def test_get_status(self):
        """Test getting the overall status of the spiral phase system"""
        status = self.spiral_manager.get_status()
        
        # Check that status has required sections
        self.assertIn('current_phase', status)
        self.assertIn('spiral_stats', status)
        self.assertIn('recent_history', status)
        
        # Check current phase info
        self.assertIn('name', status['current_phase'])
        self.assertIn('description', status['current_phase'])
        self.assertIn('time_in_phase', status['current_phase'])
        
        # Check stats info
        self.assertIn('cycle_count', status['spiral_stats'])
        self.assertIn('phase_transitions', status['spiral_stats'])
        self.assertIn('total_insights', status['spiral_stats'])


class TestDreamProcessorSpiralIntegration(unittest.TestCase):
    """Test cases for the integration of SpiralPhaseManager with LucidiaDreamProcessor"""
    
    def setUp(self):
        """Set up for each test"""
        # Mock components
        self.self_model = MagicMock()
        self.self_model.self_awareness = {"current_level": 0.5, "insights": []}
        
        self.world_model = MagicMock()
        self.world_model.concept_network = {}
        
        self.knowledge_graph = MagicMock()
        self.knowledge_graph.add_node = MagicMock()
        self.knowledge_graph.add_edge = MagicMock()
        self.knowledge_graph.has_node = MagicMock(return_value=True)
        
        # Create dream processor with mocked dependencies
        self.dream_processor = LucidiaDreamProcessor(
            self_model=self.self_model,
            world_model=self.world_model,
            knowledge_graph=self.knowledge_graph
        )
    
    def test_spiral_manager_initialization(self):
        """Test that spiral phase manager is initialized in dream processor"""
        self.assertIsNotNone(self.dream_processor.spiral_manager)
        self.assertIsInstance(self.dream_processor.spiral_manager, SpiralPhaseManager)
    
    @patch.object(LucidiaDreamProcessor, '_process_dream')
    def test_start_dreaming_with_spiral_phase(self, mock_process_dream):
        """Test that start_dreaming uses spiral phase parameters"""
        # Force the spiral manager to a specific phase
        self.dream_processor.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test")
        
        # Get phase parameters
        phase_params = self.dream_processor.spiral_manager.get_phase_params()
        
        # Start dreaming (will be forced)
        self.dream_processor.start_dreaming(forced=True)
        
        # Check that dream parameters are set based on phase
        self.assertEqual(self.dream_processor.dream_state["current_spiral_phase"], "reflection")
        
        # Verify dream depth is within phase ranges
        self.assertGreaterEqual(self.dream_processor.dream_state["current_dream_depth"], phase_params["min_depth"])
        self.assertLessEqual(self.dream_processor.dream_state["current_dream_depth"], phase_params["max_depth"])
        
        # Verify creativity is within phase ranges
        self.assertGreaterEqual(self.dream_processor.dream_state["current_dream_creativity"], phase_params["min_creativity"])
        self.assertLessEqual(self.dream_processor.dream_state["current_dream_creativity"], phase_params["max_creativity"])
    
    def test_select_dream_seed_by_phase(self):
        """Test that seed selection is influenced by spiral phase"""
        # Test for each phase and verify different weights are used
        phase_weights = {}
        
        for phase in SpiralPhase:
            self.dream_processor.spiral_manager.force_phase(phase, "test")
            
            # Call _select_dream_seed directly multiple times to get a distribution
            seed_types = []
            for _ in range(50):
                seed = self.dream_processor._select_dream_seed()
                seed_types.append(seed["type"])
            
            # Store frequencies for each phase
            phase_weights[phase] = {
                seed_type: seed_types.count(seed_type)/len(seed_types)
                for seed_type in set(seed_types)
            }
        
        # Verify that OBSERVATION favors memory and concept seeds
        self.assertGreater(
            phase_weights[SpiralPhase.OBSERVATION].get("memory", 0) + 
            phase_weights[SpiralPhase.OBSERVATION].get("concept", 0),
            phase_weights[SpiralPhase.ADAPTATION].get("memory", 0) + 
            phase_weights[SpiralPhase.ADAPTATION].get("concept", 0)
        )
        
        # Verify that ADAPTATION favors identity and creative seeds
        self.assertGreater(
            phase_weights[SpiralPhase.ADAPTATION].get("identity", 0) + 
            phase_weights[SpiralPhase.ADAPTATION].get("creative", 0),
            phase_weights[SpiralPhase.OBSERVATION].get("identity", 0) + 
            phase_weights[SpiralPhase.OBSERVATION].get("creative", 0)
        )
    
    def test_build_dream_context_by_phase(self):
        """Test that context building is influenced by spiral phase"""
        # Create test seed with different phases
        base_seed = {
            "type": "concept",
            "content": {"id": "test_concept", "definition": "A test concept"},
            "description": "Test concept",
            "relevance": 0.8,
            "theme": {"name": "identity", "keywords": ["test"], "prompt_patterns": ["What is {0}?", "How does {0} relate to {1}?"]},
            "cognitive_style": {"name": "analytical", "description": "test", "prompt_templates": ["Analyze {0}", "What are the components of {0}?"]},
            "emotional_tone": "neutral"
        }
        
        contexts_by_phase = {}
        
        for phase in SpiralPhase:
            # Set phase and create seed with that phase
            self.dream_processor.spiral_manager.force_phase(phase, "test")
            seed = base_seed.copy()
            seed["spiral_phase"] = phase.value
            
            # Build context
            context = self.dream_processor._build_dream_context(seed)
            contexts_by_phase[phase] = context
        
        # OBSERVATION phase should have fewer reflections
        self.assertLessEqual(
            len(contexts_by_phase[SpiralPhase.OBSERVATION]["reflections"]),
            len(contexts_by_phase[SpiralPhase.ADAPTATION]["reflections"])
        )
        
        # ADAPTATION phase should have more reflections
        self.assertGreaterEqual(
            len(contexts_by_phase[SpiralPhase.ADAPTATION]["reflections"]),
            len(contexts_by_phase[SpiralPhase.OBSERVATION]["reflections"])
        )
    
    def test_generate_associations_by_phase(self):
        """Test that association generation is influenced by spiral phase"""
        # Set up a basic context
        base_context = {
            "seed": {"type": "concept", "content": {"id": "test"}},
            "theme": {"name": "identity"},
            "cognitive_style": {"name": "analytical"},
            "core_concepts": [{"id": "concept1"}, {"id": "concept2"}],
            "reflections": ["test reflection"],
            "questions": ["test question"],
            "depth": 0.5,
            "creativity": 0.5,
        }
        
        associations_by_phase = {}
        
        for phase in SpiralPhase:
            self.dream_processor.spiral_manager.force_phase(phase, "test")
            context = base_context.copy()
            context["spiral_phase"] = phase.value
            
            # Generate associations
            enhanced_context = self.dream_processor._generate_dream_associations(context)
            associations_by_phase[phase] = enhanced_context.get("associations", [])
        
        # Different phases should generate different numbers of target associations
        if associations_by_phase[SpiralPhase.ADAPTATION] and associations_by_phase[SpiralPhase.OBSERVATION]:
            # ADAPTATION phase should aim for more associations
            self.assertGreaterEqual(
                len(associations_by_phase[SpiralPhase.ADAPTATION]),
                len(associations_by_phase[SpiralPhase.OBSERVATION])
            )
    
    def test_generate_insight_tagging(self):
        """Test that insights are tagged with spiral phase"""
        # Force spiral manager to REFLECTION phase
        self.dream_processor.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test")
        
        # Set up a context with phase
        context = {
            "seed": {"type": "concept", "content": {"id": "test"}},
            "theme": {"name": "identity", "prompt_patterns": ["What is {0}?"]},
            "cognitive_style": {"name": "analytical", "prompt_templates": ["Analyze {0}"]},
            "core_concepts": [{"id": "consciousness"}],
            "reflections": ["Reflection on consciousness", "Reflection on identity"],
            "questions": ["What is consciousness?", "How does identity form?"],
            "spiral_phase": "reflection",
            "depth": 0.7,
            "creativity": 0.7,
        }
        
        # Generate insights
        insights = self.dream_processor._generate_dream_insights(context)
        
        # Check that insights are tagged with phase
        for insight in insights:
            self.assertIn("spiral_phase", insight)
            self.assertEqual(insight["spiral_phase"], "reflection")
            
            # Check that phase-specific characteristics are added
            self.assertIn("characteristics", insight)
            self.assertIn("analytical", insight["characteristics"])
    
    def test_insight_significance_by_phase(self):
        """Test that insight significance is influenced by phase"""
        base_insight = {
            "type": "theme",
            "text": "Test insight text",
            "concepts": ["consciousness"],
            "theme": "identity"
        }
        
        base_context = {
            "depth": 0.7,
            "creativity": 0.7
        }
        
        significance_by_phase = {}
        
        for phase in SpiralPhase:
            self.dream_processor.spiral_manager.force_phase(phase, "test")
            context = base_context.copy()
            context["spiral_phase"] = phase.value
            
            # Calculate significance
            significance = self.dream_processor._calculate_insight_significance(base_insight, context)
            significance_by_phase[phase] = significance
        
        # Higher phases should give higher significance
        self.assertGreaterEqual(
            significance_by_phase[SpiralPhase.ADAPTATION],
            significance_by_phase[SpiralPhase.OBSERVATION]
        )
        
        self.assertGreaterEqual(
            significance_by_phase[SpiralPhase.REFLECTION],
            significance_by_phase[SpiralPhase.OBSERVATION]
        )
    
    @patch.object(SpiralPhaseManager, 'transition_phase')
    def test_integrate_insights_triggers_transition(self, mock_transition_phase):
        """Test that integrating insights can trigger phase transitions"""
        # Create insights with high significance
        insights = [
            {
                "type": "theme",
                "text": "Test insight with high significance",
                "significance": 1.0,
                "spiral_phase": "observation",
                "concepts": ["consciousness"],
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Integrate insights
        self.dream_processor._integrate_dream_insights(insights)
        
        # Verify transition_phase was called with the highest significance
        mock_transition_phase.assert_called_with(1.0)
    
    def test_update_dream_stats_by_phase(self):
        """Test that dream statistics are updated according to phase"""
        # Create a test dream record
        dream_record = {
            "id": 1,
            "start_time": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": 300,  # 5 minutes
            "depth": 0.7,
            "creativity": 0.7,
            "spiral_phase": "reflection",
            "insights": [
                {"text": "Insight 1", "significance": 0.8, "spiral_phase": "reflection"},
                {"text": "Insight 2", "significance": 0.6, "spiral_phase": "reflection"}
            ],
            "integration_results": {
                "total_insights": 2,
                "integration_success": 1
            }
        }
        
        # Update stats
        self.dream_processor._update_dream_stats(dream_record)
        
        # Check that phase-specific stats were updated
        phase_stats = self.dream_processor.dream_stats["phase_stats"]["reflection"]
        self.assertEqual(phase_stats["total_dreams"], 1)
        self.assertEqual(phase_stats["total_insights"], 2)
        self.assertGreaterEqual(phase_stats["total_dream_time"], 300)
        self.assertEqual(len(phase_stats["insight_significance"]), 2)
    
    def test_force_dream(self):
        """Test the force_dream method"""
        with patch.object(LucidiaDreamProcessor, 'start_dreaming') as mock_start:
            mock_start.return_value = True
            
            # Force dream with concept seed
            success = self.dream_processor.force_dream(seed_type="concept", concepts=["consciousness"])
            
            # Verify start_dreaming was called with forced=True
            mock_start.assert_called_with(forced=True, seed=unittest.mock.ANY)
            self.assertTrue(success)
            
            # Check seed type
            call_args = mock_start.call_args
            seed = call_args[1]["seed"]
            self.assertEqual(seed["type"], "concept")
    
    def test_set_spiral_phase(self):
        """Test setting spiral phase from dream processor"""
        # Set phase to ADAPTATION
        success = self.dream_processor.set_spiral_phase("adaptation")
        self.assertTrue(success)
        self.assertEqual(self.dream_processor.spiral_manager.current_phase, SpiralPhase.ADAPTATION)
        
        # Set phase to REFLECTION
        success = self.dream_processor.set_spiral_phase("reflection")
        self.assertTrue(success)
        self.assertEqual(self.dream_processor.spiral_manager.current_phase, SpiralPhase.REFLECTION)
        
        # Invalid phase should return False
        success = self.dream_processor.set_spiral_phase("invalid")
        self.assertFalse(success)
    
    def test_get_spiral_phase_status(self):
        """Test getting spiral phase status from dream processor"""
        # Set a specific phase for testing
        self.dream_processor.spiral_manager.force_phase(SpiralPhase.REFLECTION, "test")
        
        # Get status
        status = self.dream_processor.get_spiral_phase_status()
        
        # Check status structure
        self.assertIn('current_phase', status)
        self.assertIn('spiral_stats', status)
        self.assertIn('recent_history', status)
        
        # Check current phase
        self.assertEqual(status['current_phase']['name'], "reflection")


if __name__ == '__main__':
    unittest.main()