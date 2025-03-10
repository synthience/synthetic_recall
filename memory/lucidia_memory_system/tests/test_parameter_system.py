import unittest
import sys
import os
import json
import time
import logging
import threading
from datetime import timedelta
from unittest.mock import MagicMock

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules to test
from core.parameter_manager import ParameterManager
from core.dream_parameter_adapter import DreamParameterAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)

class TestParameterManager(unittest.TestCase):
    """Test cases for the ParameterManager class"""
    
    def setUp(self):
        """Set up for each test"""
        # Use test data config
        self.config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test_data',
            'default_config.json'
        )
        self.param_manager = ParameterManager(initial_config=self.config_path)
        
        # Setup observer
        self.observer_called = False
        self.last_change = None
        
    def observer_callback(self, parameter_path, new_value, change_record):
        """Callback for parameter changes"""
        self.observer_called = True
        self.last_change = {
            "path": parameter_path,
            "value": new_value,
            "record": change_record
        }
    
    def test_initialization(self):
        """Test parameter manager initialization"""
        # Verify config loaded correctly
        self.assertEqual(self.param_manager.config["_version"], "1.0.0")
        self.assertEqual(self.param_manager.config["dream_cycles"]["idle_threshold"], 300)
        
        # Verify metadata initialized
        self.assertTrue("dream_cycles" in self.param_manager.parameter_metadata)
        self.assertTrue("idle_threshold" in self.param_manager.parameter_metadata["dream_cycles"])
    
    def test_get_nested_value(self):
        """Test getting nested values from configuration"""
        # Test existing path
        value = self.param_manager._get_nested_value(
            self.param_manager.config, 
            "dream_cycles.idle_threshold"
        )
        self.assertEqual(value, 300)
        
        # Test non-existent path
        value = self.param_manager._get_nested_value(
            self.param_manager.config, 
            "dream_cycles.nonexistent_parameter",
            default="default_value"
        )
        self.assertEqual(value, "default_value")
    
    def test_set_nested_value(self):
        """Test setting nested values in configuration"""
        # Set existing path
        result = self.param_manager._set_nested_value(
            self.param_manager.config, 
            "dream_cycles.idle_threshold",
            400
        )
        self.assertTrue(result)
        self.assertEqual(self.param_manager.config["dream_cycles"]["idle_threshold"], 400)
        
        # Set non-existent path (should create it)
        result = self.param_manager._set_nested_value(
            self.param_manager.config, 
            "dream_cycles.new_parameter",
            "new_value"
        )
        self.assertTrue(result)
        self.assertEqual(self.param_manager.config["dream_cycles"]["new_parameter"], "new_value")
    
    def test_validate_and_cast_value(self):
        """Test value validation and casting"""
        # Integer validation
        metadata = {"type": "integer", "min": 100, "max": 500}
        
        # Valid integer
        casted = self.param_manager._validate_and_cast_value("200", metadata)
        self.assertEqual(casted, 200)
        self.assertIsInstance(casted, int)
        
        # Invalid integer (too low)
        with self.assertRaises(ValueError):
            self.param_manager._validate_and_cast_value(50, metadata)
        
        # Invalid integer (too high)
        with self.assertRaises(ValueError):
            self.param_manager._validate_and_cast_value(600, metadata)
        
        # Boolean validation
        metadata = {"type": "boolean"}
        self.assertTrue(self.param_manager._validate_and_cast_value("true", metadata))
        self.assertTrue(self.param_manager._validate_and_cast_value("True", metadata))
        self.assertTrue(self.param_manager._validate_and_cast_value(1, metadata))
        self.assertFalse(self.param_manager._validate_and_cast_value("false", metadata))
        self.assertFalse(self.param_manager._validate_and_cast_value(0, metadata))
        
        # Array validation
        metadata = {"type": "array"}
        casted = self.param_manager._validate_and_cast_value("1,2,3", metadata)
        self.assertListEqual(casted, ["1", "2", "3"])
    
    def test_observer_notification(self):
        """Test that observers are notified of parameter changes"""
        # Register observer
        self.param_manager.register_observer(self.observer_callback)
        
        # Update parameter
        self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            500
        )
        
        # Verify observer was called
        self.assertTrue(self.observer_called)
        self.assertEqual(self.last_change["path"], "dream_cycles.idle_threshold")
        self.assertEqual(self.last_change["value"], 500)
    
    def test_path_specific_observer(self):
        """Test that path-specific observers are only notified for their paths"""
        # Register path-specific observer
        path_observer_called = [False, False]
        
        def dream_cycles_observer(path, value, record):
            path_observer_called[0] = True
        
        def integration_observer(path, value, record):
            path_observer_called[1] = True
        
        self.param_manager.register_observer(dream_cycles_observer, "dream_cycles")
        self.param_manager.register_observer(integration_observer, "integration")
        
        # Update dream_cycles parameter
        self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            400
        )
        
        # Verify only dream_cycles observer was called
        self.assertTrue(path_observer_called[0])
        self.assertFalse(path_observer_called[1])
        
        # Reset and update integration parameter
        path_observer_called = [False, False]
        
        self.param_manager.update_parameter(
            "integration.default_confidence",
            0.8
        )
        
        # Verify only integration observer was called
        self.assertFalse(path_observer_called[0])
        self.assertTrue(path_observer_called[1])
    
    def test_parameter_locking(self):
        """Test parameter locking mechanism"""
        # Lock a parameter
        component_id = "test_component"
        lock_result = self.param_manager.lock_parameter(
            "dream_cycles.idle_threshold",
            component_id,
            timedelta(seconds=10),
            "Testing lock"
        )
        
        self.assertEqual(lock_result["status"], "success")
        
        # Try to update the locked parameter
        update_result = self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            500
        )
        
        # Verify update was rejected
        self.assertEqual(update_result["status"], "locked")
        
        # Unlock the parameter
        unlock_result = self.param_manager.unlock_parameter(
            "dream_cycles.idle_threshold",
            component_id
        )
        
        self.assertEqual(unlock_result["status"], "success")
        
        # Try update again
        update_result = self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            500
        )
        
        # Verify update was successful
        self.assertEqual(update_result["status"], "success")
    
    def test_gradual_transition(self):
        """Test gradual parameter transition"""
        # Register observer
        transition_values = []
        
        def transition_observer(path, value, record):
            transition_values.append(value)
        
        self.param_manager.register_observer(transition_observer)
        
        # Start a short transition (1 second)
        self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            600,  # from 300 to 600
            transition_period=timedelta(seconds=1)
        )
        
        # Wait for transition to complete
        time.sleep(1.2)
        
        # Verify final value
        self.assertEqual(
            self.param_manager.config["dream_cycles"]["idle_threshold"],
            600
        )
        
        # Verify intermediate values were generated
        # We should have at least 3 values: start, intermediate, end
        self.assertGreaterEqual(len(transition_values), 3)
        
        # First value should be close to 300, last should be 600
        self.assertLess(transition_values[0], 400)  # First value
        self.assertEqual(transition_values[-1], 600)  # Last value


class TestDreamParameterAdapter(unittest.TestCase):
    """Test cases for the DreamParameterAdapter class"""
    
    def setUp(self):
        """Set up for each test"""
        # Mock DreamProcessor
        self.dream_processor = MagicMock()
        self.dream_processor.dream_state = {"current_dream_id": "test_dream_123"}
        self.dream_processor.is_dreaming = False
        
        # Create parameter manager
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test_data',
            'default_config.json'
        )
        self.param_manager = ParameterManager(initial_config=config_path)
        
        # Create adapter
        self.adapter = DreamParameterAdapter(self.dream_processor, self.param_manager)
    
    def test_initialization(self):
        """Test adapter initialization"""
        # Verify adapter initialized with correct references
        self.assertEqual(self.adapter.dream_processor, self.dream_processor)
        self.assertEqual(self.adapter.param_manager, self.param_manager)
        
        # Verify critical parameters list
        self.assertIn("dream_cycles.idle_threshold", self.adapter.critical_parameters)
        self.assertIn("dream_cycles.auto_dream_enabled", self.adapter.critical_parameters)
    
    def test_on_parameter_changed(self):
        """Test parameter change handling"""
        # Mock the handler methods
        self.adapter._handle_idle_threshold_change = MagicMock()
        self.adapter._handle_auto_dream_change = MagicMock()
        
        # Make sure the handler is in the parameter_handlers dictionary
        self.adapter.parameter_handlers["dream_cycles.idle_threshold"] = self.adapter._handle_idle_threshold_change
        
        # Trigger parameter change for idle_threshold
        self.adapter.on_parameter_changed(
            "dream_cycles.idle_threshold",
            500,
            {"transaction_id": "test_transaction"}
        )
        
        # Verify handler was called
        self.adapter._handle_idle_threshold_change.assert_called_once()
        
        # Verify dream processor config was updated
        self.assertEqual(self.dream_processor.config, self.param_manager.config)
    
    def test_lock_critical_parameters(self):
        """Test locking critical parameters during dream"""
        # Setup is_dreaming
        self.dream_processor.is_dreaming = True
        
        # Lock parameters
        self.adapter.lock_critical_parameters_during_dream()
        
        # Verify locks were created
        with self.param_manager._lock:
            for path in self.adapter.critical_parameters:
                self.assertTrue(
                    self.param_manager.parameter_locks[path]["locked"],
                    f"Parameter {path} should be locked"
                )
                
                # Verify component_id in lock
                self.assertIn(
                    "dream_processor:test_dream_123",
                    self.param_manager.parameter_locks[path]["holder"]
                )
        
        # Verify parameters can't be updated
        update_result = self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            500
        )
        self.assertEqual(update_result["status"], "locked")
    
    def test_unlock_critical_parameters(self):
        """Test unlocking critical parameters after dream"""
        # Lock parameters first
        self.adapter.lock_critical_parameters_during_dream()
        
        # Unlock parameters
        self.adapter.unlock_critical_parameters_after_dream()
        
        # Verify locks were released
        with self.param_manager._lock:
            for path in self.adapter.critical_parameters:
                self.assertFalse(
                    self.param_manager.parameter_locks[path]["locked"],
                    f"Parameter {path} should be unlocked"
                )
        
        # Verify parameters can be updated
        update_result = self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            500
        )
        self.assertEqual(update_result["status"], "success")


class TestRealTimeTransitions(unittest.TestCase):
    """Test real-time parameter transitions without mocks"""
    
    def setUp(self):
        """Set up for each test"""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'test_data',
            'default_config.json'
        )
        self.param_manager = ParameterManager(initial_config=config_path)
        
        # Create a simple observer that records all changes
        self.changes = []
        
        def observer(path, value, record):
            self.changes.append({"path": path, "value": value, "time": time.time()})
            
        self.param_manager.register_observer(observer)
    
    def test_numeric_transition(self):
        """Test transition of numeric parameter over time"""
        start_time = time.time()
        
        # Update with a 2-second transition
        self.param_manager.update_parameter(
            "dream_cycles.idle_threshold",
            600,  # from 300 to 600
            transition_period=timedelta(seconds=2)
        )
        
        # Wait for transition to complete
        time.sleep(2.5)
        
        # Verify final value
        self.assertEqual(
            self.param_manager.config["dream_cycles"]["idle_threshold"],
            600
        )
        
        # Verify changes were recorded over time
        self.assertGreaterEqual(len(self.changes), 3)  # At least start, intermediate, end
        
        # Analyze the transitions
        values = [c["value"] for c in self.changes if c["path"] == "dream_cycles.idle_threshold"]
        
        # Verify initial and final values
        self.assertLess(values[0], 600)  # First value
        self.assertEqual(values[-1], 600)  # Last value
        
        # Verify monotonic increase
        for i in range(1, len(values)):
            self.assertGreaterEqual(values[i], values[i-1])
    
    def test_array_transition(self):
        """Test transition of array parameter over time"""
        # Update with a 2-second transition
        self.param_manager.update_parameter(
            "dream_process.depth_range",
            [0.1, 0.7],  # from [0.3, 0.9] to [0.1, 0.7]
            transition_period=timedelta(seconds=2)
        )
        
        # Wait for transition to complete
        time.sleep(2.5)
        
        # Verify final value
        self.assertEqual(
            self.param_manager.config["dream_process"]["depth_range"],
            [0.1, 0.7]
        )
        
        # Verify changes were recorded over time
        depth_range_changes = [c for c in self.changes if c["path"] == "dream_process.depth_range"]
        self.assertGreaterEqual(len(depth_range_changes), 3)  # At least start, intermediate, end
        
        # Verify monotonic decrease for first element
        first_elements = [c["value"][0] for c in depth_range_changes]
        for i in range(1, len(first_elements)):
            self.assertLessEqual(first_elements[i], first_elements[i-1])


if __name__ == '__main__':
    unittest.main()