import pytest
import pytest_asyncio
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, AsyncMock
from voice_core.stt.whisper_stt_service import WhisperSTTService
from voice_core.state.voice_state_manager import VoiceStateManager
from voice_core.state.voice_state_enum import VoiceState
from voice_core.audio.audio_frame import AudioFrame
import time

@pytest.fixture
def create_audio_frame():
    """Create an audio frame with synthetic data."""
    def _create_frame(duration=0.1, sample_rate=16000, frequency=440, amplitude=0.5, timestamp=None):
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        data = amplitude * np.sin(2 * np.pi * frequency * t)
        data = data.astype(np.float32)
        
        if timestamp is None:
            timestamp = time.time()
        
        frame = AudioFrame(
            data=data,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=num_samples,
            timestamp=timestamp
        )
        frame.energy = np.mean(np.abs(data))
        return frame
    return _create_frame

@pytest_asyncio.fixture
async def stt_service():
    """Create a test STT service instance."""
    service = WhisperSTTService(
        model_name="tiny",  # Use tiny model for faster tests
        device="cpu",
        use_vad=True,
        min_speech_prob=0.3,  # Lower threshold for tests
        min_speech_frames=2,  # Lower threshold for tests
        min_silence_frames=3,
        min_speech_duration=0.2,  # Lower duration for tests
        max_speech_duration=1.0,
        energy_threshold=0.05  # Lower threshold for tests
    )
    await service.initialize()
    yield service
    await service.cleanup()

@pytest.fixture
def mock_whisper():
    """Mock the Whisper model for testing."""
    with patch("whisper.load_model") as mock_load:
        mock_model = Mock()
        mock_model.transcribe.return_value = {"text": "test transcription"}
        mock_load.return_value = mock_model
        yield mock_model

@pytest.mark.asyncio
async def test_initialization(mock_whisper):
    service = WhisperSTTService(model_name="tiny", device="cpu")
    await service.initialize()
    assert service.model is not None
    assert service.audio_pipeline is not None
    assert service.state_manager is not None
    await service.cleanup()

@pytest.mark.asyncio
async def test_vad_fallback(mock_whisper):
    # Test fallback to energy-based VAD when Silero fails
    with patch("torch.hub.load", side_effect=Exception("VAD load failed")):
        service = WhisperSTTService(model_name="tiny", device="cpu", use_vad=True)
        await service.initialize()
        assert not service.use_vad
        assert service.vad_model is None
        await service.cleanup()

@pytest.mark.asyncio
async def test_speech_detection(stt_service, create_audio_frame):
    """Test speech detection with synthetic audio."""
    # Create frame with enough samples for VAD
    frame = create_audio_frame(
        duration=0.032,  # 32ms - typical frame size for VAD
        sample_rate=16000,
        frequency=440,
        amplitude=1.0  # Maximum amplitude for reliable detection
    )
    
    # Mock VAD for consistent behavior
    if stt_service.use_vad and stt_service.vad_model:
        stt_service.vad_model = Mock()
        stt_service.vad_model.return_value = torch.tensor([0.8])  # High speech probability
    
    # Process multiple frames to build up speech confidence
    speech_detected = 0
    for _ in range(5):  # Test with multiple frames
        is_speech = await stt_service._detect_speech(frame)
        if is_speech:
            speech_detected += 1
        await asyncio.sleep(0.05)  # Give time for VAD processing
    
    assert speech_detected >= 3, f"Expected at least 3 speech frames, got {speech_detected}"

@pytest.mark.asyncio
async def test_audio_processing_pipeline(stt_service, create_audio_frame):
    """Test the complete audio processing pipeline."""
    # Mock VAD to always detect speech
    if stt_service.use_vad and stt_service.vad_model:
        stt_service.vad_model = Mock()
        stt_service.vad_model.return_value = torch.tensor([0.8])  # High speech probability

    # Mock transcribe to return a fixed result
    stt_service.model.transcribe = Mock(return_value={"text": "test transcript"})

    # Create frames with sequential timestamps
    base_time = time.time()
    frame = create_audio_frame(
        duration=0.032,  # 32ms frame
        sample_rate=16000,
        frequency=440,
        amplitude=1.0,
        timestamp=base_time
    )

    # Process enough frames to trigger transcription
    # We need: min_speech_frames + min_silence_frames frames
    num_frames = stt_service.min_speech_frames + stt_service.min_silence_frames + 2

    # First send speech frames
    for i in range(stt_service.min_speech_frames + 1):
        frame.timestamp = base_time + (i * 0.032)  # Sequential timestamps
        async for transcript in stt_service.process_audio(frame):
            if transcript:
                stt_service.recent_transcripts.append(transcript)
        await asyncio.sleep(0.01)

    # Then send silence frames to trigger end of speech
    silent_frame = create_audio_frame(
        duration=0.032,
        sample_rate=16000,
        frequency=440,
        amplitude=0.01,  # Very low amplitude for silence
        timestamp=base_time + ((stt_service.min_speech_frames + 1) * 0.032)
    )

    for i in range(stt_service.min_silence_frames + 1):
        silent_frame.timestamp = base_time + ((stt_service.min_speech_frames + 1 + i) * 0.032)
        async for transcript in stt_service.process_audio(silent_frame):
            if transcript:
                stt_service.recent_transcripts.append(transcript)
        await asyncio.sleep(0.01)

    # Wait for transcription to complete
    await asyncio.sleep(0.1)

    # Check transcripts
    assert len(stt_service.recent_transcripts) > 0, "Expected at least one transcript"
    assert "test transcript" in stt_service.recent_transcripts

@pytest.mark.asyncio
async def test_state_transitions(stt_service, create_audio_frame):
    """Test state transitions during audio processing."""
    # Mock VAD to always detect speech
    if stt_service.use_vad and stt_service.vad_model:
        stt_service.vad_model = Mock()
        stt_service.vad_model.return_value = torch.tensor([0.9])  # High speech probability

    # Mock transcribe for faster processing
    stt_service.model.transcribe = Mock(return_value={"text": "test transcript"})

    # Initial state should be IDLE
    assert stt_service.state_manager.current_state == VoiceState.IDLE

    # Create frames with sequential timestamps
    base_time = time.time()
    frame = create_audio_frame(
        duration=0.032,
        sample_rate=16000,
        frequency=440,
        amplitude=1.0,
        timestamp=base_time
    )

    # Process frames to trigger LISTENING state
    for i in range(stt_service.min_speech_frames + 1):
        frame.timestamp = base_time + (i * 0.032)  # Sequential timestamps
        async for _ in stt_service.process_audio(frame):
            pass
        await asyncio.sleep(0.01)

    # Send silence to trigger end of speech and PROCESSING state
    silent_frame = create_audio_frame(
        duration=0.032,
        sample_rate=16000,
        frequency=440,
        amplitude=0.01,
        timestamp=base_time + ((stt_service.min_speech_frames + 1) * 0.032)
    )

    for i in range(stt_service.min_silence_frames + 1):
        silent_frame.timestamp = base_time + ((stt_service.min_speech_frames + 1 + i) * 0.032)
        async for _ in stt_service.process_audio(silent_frame):
            pass
        await asyncio.sleep(0.01)

    # Wait for all state transitions
    await asyncio.sleep(0.1)

    # Get the state history
    state_history = [state for state in stt_service.state_manager._state_history]

    # Verify state transitions
    assert len(state_history) >= 3, f"Expected at least 3 state transitions, got {len(state_history)}"
    assert state_history[0]["from_state"] == VoiceState.IDLE
    assert state_history[0]["to_state"] == VoiceState.LISTENING
    assert state_history[-1]["to_state"] == VoiceState.PROCESSING

@pytest.mark.asyncio
async def test_end_of_speech_detection(stt_service):
    # Test max duration exceeded
    base_time = 1000.0  # Use a fixed base time
    stt_service.speech_start_time = base_time
    current_time = base_time + stt_service.max_speech_duration + 1.0
    assert stt_service._check_speech_end(current_time)
    
    # Test silence duration with minimum speech duration met
    stt_service.speech_start_time = base_time
    current_time = base_time + stt_service.min_speech_duration + 0.1
    stt_service.silence_frames = stt_service.min_silence_frames + 1
    assert stt_service._check_speech_end(current_time)
    
    # Test no speech end when duration too short
    stt_service.speech_start_time = base_time
    current_time = base_time + stt_service.min_speech_duration - 0.1
    stt_service.silence_frames = stt_service.min_silence_frames + 1
    assert not stt_service._check_speech_end(current_time)

@pytest.mark.asyncio
async def test_transcript_validation(stt_service):
    # Test empty transcript
    assert not stt_service._validate_transcript("")
    assert not stt_service._validate_transcript("  ")
    
    # Test valid transcript
    assert stt_service._validate_transcript("Hello world")
    
    # Test duplicate transcript
    transcript = "Test transcript"
    assert stt_service._validate_transcript(transcript)
    assert not stt_service._validate_transcript(transcript)  # Should fail as duplicate

@pytest.mark.asyncio
async def test_cleanup(stt_service):
    """Test cleanup of resources."""
    await stt_service.cleanup()
    assert stt_service.executor is None, "Executor should be None after cleanup"
    assert stt_service.model is None, "Model should be None after cleanup"
    assert stt_service.vad_model is None, "VAD model should be None after cleanup"
    assert stt_service.audio_pipeline is None, "Audio pipeline should be None after cleanup"
    assert len(stt_service.vad_buffer) == 0, "VAD buffer should be empty after cleanup"

@pytest.mark.asyncio
async def test_error_handling(mock_whisper):
    service = WhisperSTTService(model_name="tiny", device="cpu")

    # Test model initialization error
    with patch("whisper.load_model", side_effect=RuntimeError("Model load failed")):
        with pytest.raises(RuntimeError, match="Model load failed"):
            await service.initialize()

    # Initialize service for remaining tests
    await service.initialize()
    try:
        # Test transcription error handling
        service.model.transcribe.side_effect = RuntimeError("Transcription failed")

        # Should handle transcription error gracefully
        audio_data = np.random.rand(48000).astype(np.float32)  # 1 second at 48kHz
        audio_frame = AudioFrame(
            data=audio_data,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=len(audio_data)
        )
        async for transcript in service.process_audio(audio_frame):
            assert transcript is None

        # Test invalid audio input - empty array
        with pytest.raises(ValueError, match="Input array is empty"):
            async for _ in service.process_audio(AudioFrame(
                data=np.array([], dtype=np.float32),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=0
            )):
                pass
    finally:
        await service.cleanup()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
