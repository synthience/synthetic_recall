import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from voice_core.state.voice_state_manager import VoiceStateManager
from voice_core.state.voice_state_enum import VoiceState

@pytest.fixture
async def voice_state_manager():
    manager = VoiceStateManager()
    yield manager
    # Cleanup
    if manager._voice_state != VoiceState.IDLE:
        await manager.transition_to(VoiceState.IDLE)

@pytest.mark.asyncio
async def test_initial_state(voice_state_manager):
    assert voice_state_manager.current_state == VoiceState.IDLE
    assert isinstance(voice_state_manager.session_metrics, dict)

@pytest.mark.asyncio
async def test_valid_state_transitions(voice_state_manager):
    # Test valid transition path
    assert await voice_state_manager.transition_to(VoiceState.LISTENING)
    assert voice_state_manager.current_state == VoiceState.LISTENING
    
    assert await voice_state_manager.transition_to(VoiceState.PROCESSING)
    assert voice_state_manager.current_state == VoiceState.PROCESSING
    
    assert await voice_state_manager.transition_to(VoiceState.SPEAKING)
    assert voice_state_manager.current_state == VoiceState.SPEAKING
    
    assert await voice_state_manager.transition_to(VoiceState.IDLE)
    assert voice_state_manager.current_state == VoiceState.IDLE

@pytest.mark.asyncio
async def test_invalid_state_transitions(voice_state_manager):
    # Test invalid transition (IDLE -> SPEAKING)
    assert not await voice_state_manager.transition_to(VoiceState.SPEAKING)
    assert voice_state_manager.current_state == VoiceState.ERROR
    
    # Verify metrics were updated
    metrics = voice_state_manager.session_metrics
    assert metrics['invalid_transitions'] == 1
    assert metrics['errors'] == 1

@pytest.mark.asyncio
async def test_concurrent_transitions(voice_state_manager):
    async def transition_sequence():
        await voice_state_manager.transition_to(VoiceState.LISTENING)
        await asyncio.sleep(0.1)
        await voice_state_manager.transition_to(VoiceState.PROCESSING)
        await asyncio.sleep(0.1)
        await voice_state_manager.transition_to(VoiceState.SPEAKING)
        await asyncio.sleep(0.1)
        await voice_state_manager.transition_to(VoiceState.IDLE)
    
    # Run multiple transition sequences concurrently
    tasks = [transition_sequence() for _ in range(3)]
    await asyncio.gather(*tasks)
    
    # Verify final state and metrics
    assert voice_state_manager.current_state == VoiceState.IDLE
    metrics = voice_state_manager.session_metrics
    assert metrics['state_transitions'] > 0

@pytest.mark.asyncio
async def test_metrics_collection(voice_state_manager):
    # Perform a series of transitions
    await voice_state_manager.transition_to(VoiceState.LISTENING)
    await asyncio.sleep(0.1)  # Simulate some time passing
    
    await voice_state_manager.transition_to(VoiceState.PROCESSING)
    await asyncio.sleep(0.1)
    
    await voice_state_manager.transition_to(VoiceState.SPEAKING)
    await asyncio.sleep(0.2)  # Longer speaking duration
    
    await voice_state_manager.transition_to(VoiceState.INTERRUPTED)
    
    # Check metrics
    metrics = voice_state_manager.session_metrics
    assert metrics['total_speaking_time'] > 0
    assert metrics['interrupts'] == 1
    assert metrics['state_transitions'] == 4

@pytest.mark.asyncio
async def test_tts_session(voice_state_manager):
    async with voice_state_manager.tts_session("test text"):
        assert voice_state_manager.current_state == VoiceState.SPEAKING
        # Simulate interrupt
        await voice_state_manager.transition_to(VoiceState.INTERRUPTED)
        assert voice_state_manager.current_state == VoiceState.INTERRUPTED
    
    # After session ends
    assert voice_state_manager.current_state == VoiceState.IDLE

@pytest.mark.asyncio
async def test_error_handling(voice_state_manager):
    # Test transition to ERROR state
    await voice_state_manager.transition_to(VoiceState.ERROR, {"error": "test_error"})
    assert voice_state_manager.current_state == VoiceState.ERROR
    
    # Test recovery
    await voice_state_manager.transition_to(VoiceState.RECOVERING)
    assert voice_state_manager.current_state == VoiceState.RECOVERING
    
    await voice_state_manager.transition_to(VoiceState.IDLE)
    assert voice_state_manager.current_state == VoiceState.IDLE

@pytest.mark.asyncio
async def test_state_history(voice_state_manager):
    # Perform multiple transitions
    states = [
        VoiceState.LISTENING,
        VoiceState.PROCESSING,
        VoiceState.SPEAKING,
        VoiceState.INTERRUPTED,
        VoiceState.IDLE
    ]
    
    for state in states:
        await voice_state_manager.transition_to(state)
        await asyncio.sleep(0.1)
    
    # Verify history
    history = voice_state_manager._state_history
    assert len(history) == len(states)
    
    # Check history entries
    for entry in history:
        assert 'from_state' in entry
        assert 'to_state' in entry
        assert 'timestamp' in entry
        assert 'duration' in entry
        assert isinstance(entry['duration'], float)
        assert entry['duration'] >= 0

@pytest.mark.asyncio
async def test_event_emission(voice_state_manager):
    events = []
    
    @voice_state_manager.on("state_changed")
    async def handle_state_change(event_data):
        events.append(event_data)
    
    await voice_state_manager.transition_to(VoiceState.LISTENING)
    await asyncio.sleep(0.1)  # Allow event to be processed
    
    assert len(events) == 1
    assert events[0]["old_state"] == VoiceState.IDLE
    assert events[0]["new_state"] == VoiceState.LISTENING
    assert "session_id" in events[0]
    assert "timestamp" in events[0]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
