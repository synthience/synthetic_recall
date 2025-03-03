import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from voice_core.stt.enhanced_stt_pipeline import EnhancedSTTPipeline, WhisperConfig
from voice_core.utils.audio_buffer import AudioBuffer

@pytest.fixture
def whisper_config():
    return WhisperConfig(
        model_name="tiny.en",  # Use tiny model for faster tests
        device="cpu",  # Force CPU for consistent testing
        initial_silence_threshold=-35.0,
        min_audio_length=0.7,
        min_speech_duration=0.5,
        silence_duration=1.0
    )

@pytest.fixture
def mock_audio_frame():
    # Create synthetic audio data (1 second of 440Hz sine wave)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)
    # Normalize and convert to int16
    audio = (audio * 32767).astype(np.int16)
    
    # Mock LiveKit frame
    frame = Mock()
    frame.data = audio.tobytes()
    frame.sample_rate = sample_rate
    frame.num_channels = 1
    return frame

@pytest.fixture
def mock_audio_track():
    track = Mock()
    return track

@pytest.fixture
def mock_audio_stream(mock_audio_frame):
    class MockAudioStream:
        def __init__(self, track):
            self.track = track
            self.frame = mock_audio_frame
            
        async def __aiter__(self):
            # Yield 3 frames to simulate continuous audio
            for _ in range(3):
                yield Mock(frame=self.frame)
                await asyncio.sleep(0.1)
    
    return MockAudioStream

@pytest.mark.asyncio
async def test_pipeline_initialization(whisper_config):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    assert pipeline.model is not None
    assert pipeline.audio_buffer is not None
    assert isinstance(pipeline.audio_buffer, AudioBuffer)
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_audio_preprocessing(whisper_config, mock_audio_frame):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Convert frame to numpy array
    audio_data = np.frombuffer(mock_audio_frame.data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Test preprocessing
    processed_audio = pipeline.audio_processor.preprocess(audio_data)
    assert processed_audio is not None
    assert isinstance(processed_audio, np.ndarray)
    assert not np.any(np.isnan(processed_audio))
    assert np.max(np.abs(processed_audio)) <= 1.0
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_speech_detection(whisper_config, mock_audio_track, mock_audio_stream):
    with patch('livekit.rtc.AudioStream', mock_audio_stream):
        pipeline = EnhancedSTTPipeline(whisper_config)
        await pipeline.initialize()
        
        transcripts = []
        async for transcript in pipeline.process_audio(mock_audio_track):
            transcripts.append(transcript)
            if len(transcripts) >= 1:  # Get at least one transcript
                break
        
        # Verify speech detection and transcription
        assert pipeline.is_speaking is False  # Should reset after processing
        assert len(pipeline.speech_energy_history) > 0
        assert pipeline.audio_buffer is not None
        
        await pipeline.cleanup()

@pytest.mark.asyncio
async def test_noise_reduction(whisper_config, mock_audio_frame):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Create noisy audio
    audio_data = np.frombuffer(mock_audio_frame.data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    noise = np.random.normal(0, 0.1, len(audio_data))
    noisy_audio = audio_data + noise
    
    # Test noise reduction
    processed_audio = pipeline.audio_processor._apply_noise_reduction(noisy_audio)
    assert processed_audio is not None
    assert np.std(processed_audio) < np.std(noisy_audio)  # Verify noise reduction
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_pipeline_shutdown(whisper_config):
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Test graceful shutdown
    assert pipeline._shutdown is False
    await pipeline.stop()
    assert pipeline._shutdown is True
    assert pipeline._heartbeat_task is None
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_energy_score_calculation(whisper_config):
    """Test energy score calculation with different audio levels."""
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Test cases with different amplitudes and frequencies
    test_cases = [
        # Quiet speech (-20dB)
        {
            'amplitude': 0.1,
            'freq': 200,
            'expected_min': 0.1
        },
        # Normal speech (-10dB)
        {
            'amplitude': 0.3,
            'freq': 1000,
            'expected_min': 0.3
        },
        # Loud speech (-3dB)
        {
            'amplitude': 0.7,
            'freq': 2000,
            'expected_min': 0.5
        }
    ]
    
    for case in test_cases:
        # Generate test signal
        duration = 0.5
        t = np.linspace(0, duration, int(16000 * duration))
        signal = case['amplitude'] * np.sin(2 * np.pi * case['freq'] * t)
        
        # Calculate energy score
        raw_level = 20 * np.log10(np.sqrt(np.mean(signal**2)) + 1e-10)
        energy_score = pipeline._calculate_speech_energy_score(signal, raw_level)
        
        # Verify score is non-zero and scales with amplitude
        assert energy_score > 0, f"Energy score should be > 0 for amplitude {case['amplitude']}"
        assert energy_score >= case['expected_min'], f"Energy score {energy_score} below expected minimum {case['expected_min']}"
        
        print(f"Test case - Amplitude: {case['amplitude']}, Freq: {case['freq']}Hz")
        print(f"Raw Level: {raw_level:.1f}dB, Energy Score: {energy_score:.3f}")
    
    await pipeline.cleanup()

@pytest.mark.asyncio
async def test_voice_detection_thresholds(whisper_config):
    """Test voice detection with various signal levels."""
    pipeline = EnhancedSTTPipeline(whisper_config)
    await pipeline.initialize()
    
    # Test cases with different signal levels
    test_cases = [
        # Very quiet (-30dB) - should not trigger
        {
            'filtered_db': -30,
            'raw_level': -35,
            'expected_detection': False
        },
        # Moderate (-15dB) - should trigger
        {
            'filtered_db': -15,
            'raw_level': -20,
            'expected_detection': True
        },
        # Loud (-5dB) - should definitely trigger
        {
            'filtered_db': -5,
            'raw_level': -10,
            'expected_detection': True
        }
    ]
    
    for case in test_cases:
        # Generate test signal
        duration = 0.5
        t = np.linspace(0, duration, int(16000 * duration))
        amplitude = 10 ** (case['raw_level'] / 20)
        signal = amplitude * np.sin(2 * np.pi * 1000 * t)
        
        # Calculate energy score
        energy_score = pipeline._calculate_speech_energy_score(signal, case['raw_level'])
        
        # Test voice detection
        is_voice = pipeline._detect_voice(
            case['filtered_db'],
            case['raw_level'],
            signal,
            energy_score
        )
        
        assert is_voice == case['expected_detection'], \
            f"Voice detection failed for filtered_db: {case['filtered_db']}, " \
            f"raw_level: {case['raw_level']}, energy_score: {energy_score:.3f}"
        
        print(f"Test case - Filtered: {case['filtered_db']}dB, Raw: {case['raw_level']}dB")
        print(f"Energy Score: {energy_score:.3f}, Detection: {is_voice}")
    
    await pipeline.cleanup()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
