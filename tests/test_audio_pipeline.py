import pytest
import numpy as np
import asyncio
import time
from unittest.mock import patch
from voice_core.audio.audio_pipeline import AudioPipeline, AudioFrame, FrameManager

@pytest.fixture
def audio_pipeline():
    return AudioPipeline(
        input_sample_rate=48000,
        output_sample_rate=16000,
        frame_size_ms=30,
        frame_overlap_ms=10
    )

@pytest.fixture
def frame_manager():
    return FrameManager(
        frame_size=480,  # 30ms at 16kHz
        overlap=160,     # 10ms overlap
        sample_rate=16000
    )

def test_audio_pipeline_initialization(audio_pipeline):
    assert audio_pipeline.input_sample_rate == 48000
    assert audio_pipeline.output_sample_rate == 16000
    assert audio_pipeline.input_frame_size == 1440  # 30ms at 48kHz
    assert audio_pipeline.output_frame_size == 480   # 30ms at 16kHz

def test_frame_manager_processing(frame_manager):
    # Create 100ms of audio at 16kHz (1600 samples)
    test_audio = np.random.rand(1600).astype(np.float32)
    
    # Process frames
    frames = frame_manager.add_samples(test_audio)
    
    # Calculate expected frame count based on frame size and overlap
    frame_size = 480  # 30ms at 16kHz
    overlap = 160     # 10ms overlap
    stride = frame_size - overlap
    expected_frames = max(0, (len(test_audio) - frame_size) // stride + 1)
    
    assert len(frames) == expected_frames
    
    # Each frame should be 480 samples (30ms at 16kHz)
    assert all(len(frame.data) == frame_size for frame in frames)
    
    # Verify timestamps are monotonically increasing with correct spacing
    timestamps = [frame.timestamp for frame in frames]
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        expected_diff = stride / frame_manager.sample_rate  # Time between frame starts
        assert all(abs(diff - expected_diff) < 0.001 for diff in time_diffs)

def test_audio_pipeline_resampling(audio_pipeline):
    # Create 100ms of audio at 48kHz (4800 samples)
    input_audio = np.random.rand(4800).astype(np.float32)
    
    # Process through pipeline
    input_frames, output_frames = audio_pipeline.process_input(input_audio)
    
    # Verify frame counts
    assert len(input_frames) > 0
    assert len(output_frames) > 0
    
    # Verify sample rates
    assert all(frame.sample_rate == 48000 for frame in input_frames)
    assert all(frame.sample_rate == 16000 for frame in output_frames)

def test_audio_pipeline_normalization(audio_pipeline):
    # Create audio with int16 range
    input_audio = (np.random.rand(4800) * 65536 - 32768).astype(np.int16)
    
    # Process through pipeline
    input_frames, output_frames = audio_pipeline.process_input(input_audio)
    
    # Verify normalization to [-1, 1]
    for frame in output_frames:
        # Convert frame data to float64 for more precise comparison
        frame_data = frame.data.astype(np.float64)
        assert np.all(frame_data >= -1.0 - 1e-6)  # Allow small numerical errors
        assert np.all(frame_data <= 1.0 + 1e-6)

def test_recent_audio_retrieval(audio_pipeline):
    # Create 500ms of audio at 48kHz
    input_audio = np.random.rand(24000).astype(np.float32)
    
    # Process through pipeline
    audio_pipeline.process_input(input_audio)
    
    # Get last 200ms of audio
    duration = 0.2
    recent_audio = audio_pipeline.get_recent_audio(duration, use_input_rate=False)
    
    # Calculate expected length with frame alignment
    expected_samples = int(duration * audio_pipeline.output_sample_rate)
    max_samples = int(0.5 * audio_pipeline.output_sample_rate)  # 500ms worth of samples
    
    # Verify length is reasonable (should be close to expected, but may vary due to frame alignment)
    assert len(recent_audio) >= expected_samples * 0.8  # Allow some flexibility
    assert len(recent_audio) <= max_samples  # Shouldn't exceed input duration

def test_frame_energy_calculation(frame_manager):
    # Create silent audio
    silent_audio = np.zeros(480, dtype=np.float32)
    silent_frames = frame_manager.add_samples(silent_audio)
    
    # Test silent frames (should have very low energy)
    max_silent_energy = 1e-6
    assert all(frame.energy <= max_silent_energy for frame in silent_frames)
    
    # Create loud audio (sine wave with amplitude 0.5)
    t = np.linspace(0, 2*np.pi, 480)
    loud_audio = np.sin(t) * 0.5
    loud_frames = frame_manager.add_samples(loud_audio)
    
    # Calculate expected RMS energy (should be ~0.354 for sine wave with amplitude 0.5)
    actual_energy = loud_frames[0].energy
    assert 0.33 <= actual_energy <= 0.37  # Expected range for sine wave RMS

def test_frame_alignment(audio_pipeline):
    # Test various input sizes
    test_sizes = [
        audio_pipeline.input_frame_size - 100,  # Smaller than frame
        audio_pipeline.input_frame_size,        # Exact frame
        audio_pipeline.input_frame_size + 100   # Larger than frame
    ]
    
    for size in test_sizes:
        audio = np.random.rand(size).astype(np.float32)
        input_frames, output_frames = audio_pipeline.process_input(audio)
        
        # Verify frame sizes
        if input_frames:
            assert all(len(frame.data) == audio_pipeline.input_frame_size for frame in input_frames)
        if output_frames:
            assert all(len(frame.data) == audio_pipeline.output_frame_size for frame in output_frames)
        
        # Verify sample rates
        if input_frames:
            assert all(frame.sample_rate == audio_pipeline.input_sample_rate for frame in input_frames)
        if output_frames:
            assert all(frame.sample_rate == audio_pipeline.output_sample_rate for frame in output_frames)

@pytest.mark.asyncio
async def test_time_synchronization(audio_pipeline):
    current_time = 0.0
    def mock_time():
        nonlocal current_time
        current_time += 0.01
        return current_time
    
    with patch('time.monotonic', side_effect=mock_time):
        # Create 100ms of audio
        input_audio = np.random.rand(4800).astype(np.float32)
        
        # Process in two chunks with a delay
        input_frames1, _ = audio_pipeline.process_input(input_audio[:2400])
        await asyncio.sleep(0.01)  # Small delay
        input_frames2, _ = audio_pipeline.process_input(input_audio[2400:])
        
        # Verify monotonic timestamps
        all_frames = input_frames1 + input_frames2
        timestamps = [frame.timestamp for frame in all_frames]
        assert all(t1 < t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:]))

def test_error_handling(audio_pipeline):
    # Test invalid input types
    with pytest.raises((ValueError, TypeError)):
        audio_pipeline.process_input("invalid input")
    
    # Test empty input
    with pytest.raises(ValueError):
        audio_pipeline.process_input(np.array([], dtype=np.float32))
    
    # Test invalid sample rate conversion
    with pytest.raises(ValueError):
        AudioPipeline(
            input_sample_rate=44100,  # Non-multiple of output rate
            output_sample_rate=16000,
            frame_size_ms=30,
            frame_overlap_ms=10
        )
    
    # Test invalid frame parameters
    with pytest.raises(ValueError):
        AudioPipeline(
            input_sample_rate=48000,
            output_sample_rate=16000,
            frame_size_ms=0,  # Invalid frame size
            frame_overlap_ms=10
        )

def test_continuous_streaming(audio_pipeline):
    # Simulate continuous audio streaming
    chunk_size = 480  # 10ms at 48kHz
    num_chunks = 10
    frame_counts = []
    
    for _ in range(num_chunks):
        chunk = np.random.rand(chunk_size).astype(np.float32)
        input_frames, output_frames = audio_pipeline.process_input(chunk)
        frame_counts.append(len(output_frames))
    
    # Verify consistent frame generation
    non_zero_counts = [c for c in frame_counts if c > 0]
    assert len(non_zero_counts) > 0
    assert max(non_zero_counts) - min(non_zero_counts) <= 1  # Should be fairly consistent

if __name__ == "__main__":
    pytest.main([__file__])
