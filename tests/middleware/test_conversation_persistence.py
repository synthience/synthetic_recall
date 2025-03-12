import pytest
import asyncio
import json
import os
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from memory.middleware.conversation_persistence import ConversationPersistenceMiddleware
from memory.lucidia_memory_system.core.memory_types import MemoryTypes

@pytest.fixture
def memory_integration_mock():
    # Create a mock memory integration with a mock memory core
    mock = MagicMock()
    mock.memory_core = AsyncMock()
    mock.memory_core.process_and_store = AsyncMock(return_value=True)
    mock.memory_core.retrieve_by_similarity = AsyncMock(return_value=[])
    return mock

@pytest.fixture
def session_dir(tmpdir):
    # Create a temporary directory for session data
    session_path = tmpdir.mkdir("session_data")
    return str(session_path)

@pytest.fixture
def middleware(memory_integration_mock, session_dir):
    # Create a middleware instance with a mock memory integration
    config = {
        'session_dir': session_dir,
        'checkpointing_interval': 3,
        'context_window': 5,
        'interaction_significance': 0.7,
        'role_metadata': True,
        'max_history_size': 100
    }
    middleware = ConversationPersistenceMiddleware(memory_integration_mock, config)
    return middleware

@pytest.mark.asyncio
async def test_initialize_session(middleware):
    # Test initializing a new session
    await middleware.initialize_session("test_session")
    assert middleware.session_id == "test_session"
    assert middleware.conversation_history == []

@pytest.mark.asyncio
async def test_save_and_load_session_state(middleware, session_dir):
    # Test saving and loading session state
    await middleware.initialize_session("test_save_load")
    
    # Add some conversation history
    interaction = {
        'user': 'Hello',
        'response': 'Hi there!',
        'timestamp': datetime.now().isoformat(),
        'sequence_number': 0,
        'turn_id': 'test_save_load_0'
    }
    middleware.conversation_history.append(interaction)
    
    # Save the session state
    await middleware.save_session_state()
    
    # Verify the file exists
    session_file = os.path.join(session_dir, f"test_save_load.json")
    assert os.path.exists(session_file)
    
    # Create a new middleware instance and load the state
    new_middleware = ConversationPersistenceMiddleware(
        middleware.memory_integration, 
        middleware.config
    )
    await new_middleware.initialize_session("test_save_load")
    
    # Verify the state was loaded correctly
    assert len(new_middleware.conversation_history) == 1
    assert new_middleware.conversation_history[0]['user'] == 'Hello'
    assert new_middleware.conversation_history[0]['response'] == 'Hi there!'

@pytest.mark.asyncio
async def test_store_interaction(middleware, memory_integration_mock):
    # Test storing an interaction
    await middleware.initialize_session("test_store")
    
    result = await middleware.store_interaction("Hello", "Hi there!")
    
    # Verify the interaction was stored
    assert result is True
    assert len(middleware.conversation_history) == 1
    assert middleware.conversation_history[0]['user'] == 'Hello'
    assert middleware.conversation_history[0]['response'] == 'Hi there!'
    
    # Verify the memory core was called with the right parameters
    memory_integration_mock.memory_core.process_and_store.assert_called()
    # Should be called 3 times: for user input, assistant response, and combined interaction
    assert memory_integration_mock.memory_core.process_and_store.call_count == 3

@pytest.mark.asyncio
async def test_store_interaction_with_significance(middleware, memory_integration_mock):
    # Test storing an interaction with custom significance
    await middleware.initialize_session("test_significance")
    
    result = await middleware.store_interaction("Important question", "Critical answer", significance=0.95)
    
    # Verify the interaction was stored with the right significance
    assert result is True
    calls = memory_integration_mock.memory_core.process_and_store.call_args_list
    
    # The third call (combined interaction) should have the custom significance
    third_call = calls[2]
    assert third_call[1]['significance'] == 0.95

@pytest.mark.asyncio
async def test_retrieve_relevant_context(middleware, memory_integration_mock):
    # Test retrieving relevant context
    await middleware.initialize_session("test_retrieve")
    
    # Mock the memory retrieval
    memory_integration_mock.memory_core.retrieve_by_similarity.return_value = [
        {
            'content': 'Previous relevant conversation',
            'metadata': {'session_id': 'test_retrieve', 'turn_id': 'test_123'},
            'similarity': 0.85
        }
    ]
    
    context = await middleware.retrieve_relevant_context("Tell me more about that")
    
    # Verify the context was retrieved
    assert context['memories'][0]['content'] == 'Previous relevant conversation'
    assert context['memories'][0]['similarity'] == 0.85
    
    # Verify the memory core was called with the right parameters
    memory_integration_mock.memory_core.retrieve_by_similarity.assert_called_once()

@pytest.mark.asyncio
async def test_max_history_size(middleware):
    # Test that the history size is limited
    await middleware.initialize_session("test_max_size")
    middleware.config['max_history_size'] = 3
    
    # Add more interactions than the max size
    for i in range(5):
        await middleware.store_interaction(f"User input {i}", f"Response {i}")
    
    # Verify only the most recent interactions are kept
    assert len(middleware.conversation_history) == 3
    assert middleware.conversation_history[0]['user'] == 'User input 2'
    assert middleware.conversation_history[-1]['user'] == 'User input 4'

@pytest.mark.asyncio
async def test_concurrency_handling(middleware):
    # Test that the lock prevents concurrent modifications
    await middleware.initialize_session("test_concurrency")
    
    # Create a scenario with concurrent access
    async def concurrent_store(i):
        return await middleware.store_interaction(f"User {i}", f"Response {i}")
    
    # Run multiple store operations concurrently
    tasks = [concurrent_store(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Verify all operations succeeded
    assert all(results)
    
    # Verify the history contains all interactions in the correct order
    assert len(middleware.conversation_history) == 10
    for i in range(10):
        assert middleware.conversation_history[i]['sequence_number'] == i

@pytest.mark.asyncio
async def test_error_handling(middleware, memory_integration_mock):
    # Test error handling during interaction storage
    await middleware.initialize_session("test_error")
    
    # Make the memory core raise an exception
    memory_integration_mock.memory_core.process_and_store.side_effect = Exception("Storage error")
    
    # Attempt to store an interaction
    result = await middleware.store_interaction("Hello", "Error response")
    
    # Verify the operation failed gracefully
    assert result is False
    assert len(middleware.conversation_history) == 1  # The interaction is still added to history
