import asyncio
import logging
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import stt, transcription
from .vosk_stt_service import VoskSTTService
import sounddevice as sd
import numpy as np
import jwt  # Add JWT import for token generation
import time  # Add time import for token expiration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transcription-test")

# Load environment variables from absolute path
env_path = Path(__file__).parent.parent / '.env'
logger.info(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path)

# Audio settings
SAMPLE_RATE = 16000  # Hz - Vosk expects 16kHz
CHANNELS = 1
CHUNK_SIZE = 1024

class AudioSourceWrapper:
    def __init__(self, source: rtc.AudioSource):
        self.source = source
        
    async def write_samples(self, samples):
        """Write samples to the audio source"""
        try:
            # Convert to float32 if needed
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            # Ensure samples are in [-1, 1] range
            if samples.max() > 1 or samples.min() < -1:
                samples = samples / 32768.0
                
            # Create audio frame and push to LiveKit
            frame = rtc.AudioFrame(
                data=samples.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=len(samples) // CHANNELS
            )
            await self.source.capture_frame(frame)
        except Exception as e:
            logger.error(f"Error writing samples to audio source: {e}")
            raise

class AudioFrame:
    """Simple audio frame class that mimics the expected interface"""
    def __init__(self, data: np.ndarray):
        self.data = data
        self.samples_per_channel = len(data)
        self.num_channels = data.shape[1] if len(data.shape) > 1 else 1

class AudioPipeline:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, target_sample_rate=48000):
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.buffer = []
        self.level_monitor = LevelMonitor()
        self.stt_service = None
        self.stt_stream = None
        
    async def init_stt(self):
        """Initialize STT service"""
        self.stt_service = VoskSTTService()
        self.stt_stream = await self.stt_service.stream()
        logger.info("Initialized Vosk STT service")
        
    def process_audio(self, data: np.ndarray) -> np.ndarray:
        """Process audio data with format conversion and normalization"""
        try:
            # Ensure data is 1D array
            data = data.flatten()
            
            # Convert to float32 if needed
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            # Normalize to [-1, 1]
            if data.max() > 1 or data.min() < -1:
                data = data / 32768.0
            
            # Update audio level
            self.level_monitor.update(data)
            
            # Send to STT if initialized
            if self.stt_stream:
                self.stt_stream.write_samples(data)
            
            # Resample if needed
            if self.sample_rate != self.target_sample_rate:
                data = self._resample(data)
            
            return data
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return np.zeros(self.chunk_size, dtype=np.float32)
    
    def _resample(self, data: np.ndarray) -> np.ndarray:
        """Resample audio data to target sample rate"""
        # Simple linear interpolation for now
        ratio = self.target_sample_rate / self.sample_rate
        target_length = int(len(data) * ratio)
        return np.interp(
            np.linspace(0, len(data)-1, target_length),
            np.arange(len(data)),
            data
        )

class LevelMonitor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.frame_count = 0
        
    def update(self, data: np.ndarray):
        """Update audio level monitoring"""
        self.frame_count += 1
        if self.frame_count % self.window_size == 0:
            rms = np.sqrt(np.mean(data**2))
            level_db = 20 * np.log10(rms + 1e-10)
            logger.info(f"Audio level: {level_db:.1f} dB")

class CustomAudioSource(rtc.AudioSource):
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        # Initialize with target sample rate for LiveKit
        super().__init__(sample_rate=48000, num_channels=channels)
        self.pipeline = AudioPipeline(
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            target_sample_rate=48000  # LiveKit expects 48kHz
        )
        self.buffer = asyncio.Queue()
        self.transcription_callback = None
        
    async def init(self):
        """Initialize the audio source"""
        await self.pipeline.init_stt()
        # Start transcription processing
        asyncio.create_task(self._process_transcriptions())
        
    async def _process_transcriptions(self):
        """Process transcriptions from STT"""
        try:
            while True:
                try:
                    result = await self.pipeline.stt_stream._queue.get()
                    if self.transcription_callback:
                        await self.transcription_callback(result)
                    else:
                        if result["type"] == "final_transcript":
                            logger.info(f"Transcript: {result['alternatives'][0]['text']}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error processing transcription: {e}")
        except Exception as e:
            logger.error(f"Error in transcription loop: {e}")

    async def process_frame(self, data: np.ndarray):
        """Process and publish a frame of audio data"""
        try:
            # Process audio through our pipeline
            processed_data = self.pipeline.process_audio(data)
            
            # Ensure we have the right number of samples
            samples_per_channel = len(processed_data) // self.pipeline.channels
            
            # Create frame with processed data
            frame = rtc.AudioFrame(
                data=processed_data.tobytes(),
                sample_rate=48000,  # Fixed to LiveKit's expected rate
                num_channels=self.pipeline.channels,
                samples_per_channel=samples_per_channel
            )
            
            await self.capture_frame(frame)
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            logger.error(f"Data shape: {data.shape}, dtype: {data.dtype}")

async def capture_microphone(audio_source: AudioSourceWrapper):
    """Capture audio from microphone and send to LiveKit"""
    loop = asyncio.get_event_loop()
    
    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio input status: {status}")
        # Convert to float32 and normalize to [-1, 1]
        audio_data = indata.flatten()
        # Schedule write_samples in event loop
        asyncio.run_coroutine_threadsafe(audio_source.write_samples(audio_data), loop)

    logger.info("Starting microphone capture...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.int16,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        # Keep the stream running
        while True:
            await asyncio.sleep(0.1)

async def create_audio_source():
    """Create an audio source with custom pipeline"""
    logger.info("Creating custom audio source...")
    audio_source = CustomAudioSource(
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        chunk_size=CHUNK_SIZE
    )

    def audio_callback(indata, frames, time, status):
        if status:
            logger.warning(f"Audio input status: {status}")
        try:
            # Ensure data is the right shape
            data = indata.copy()  # Make a copy to avoid modifying the input buffer
            audio_source.buffer.put_nowait(data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        callback=audio_callback,
        blocksize=CHUNK_SIZE
    )
    stream.start()
    logger.info(f"Started audio stream with {SAMPLE_RATE}Hz, {CHANNELS} channels")
    
    # Start frame processing
    async def process_frames():
        while True:
            try:
                data = await audio_source.buffer.get()
                await audio_source.process_frame(data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing frames: {e}")
                logger.error(f"Frame data shape: {data.shape if isinstance(data, np.ndarray) else 'unknown'}")
    
    process_task = asyncio.create_task(process_frames())
    
    # Store for cleanup
    audio_source._stream = stream
    audio_source._process_task = process_task
    return audio_source

async def cleanup_audio_source(audio_source):
    """Clean up audio source resources"""
    if hasattr(audio_source, '_stream'):
        audio_source._stream.stop()
        audio_source._stream.close()
        logger.info("Stopped and closed audio stream")
    
    if hasattr(audio_source, '_process_task'):
        audio_source._process_task.cancel()
        try:
            await audio_source._process_task
        except asyncio.CancelledError:
            pass
        logger.info("Cancelled audio processing task")

async def publish_track_with_retry(room, track, max_retries=3):
    """Publish track with retry logic"""
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Check for connected state (1)
            if room.connection_state != 1:  # 1 is CONNECTED state
                logger.warning(f"Room not connected, waiting... State: {room.connection_state}")
                await asyncio.sleep(2)
                continue
                
            logger.info(f"Attempting to publish track (attempt {retry_count + 1}/{max_retries})")
            
            # Publish track directly
            publication = await asyncio.wait_for(
                room.local_participant.publish_track(track),
                timeout=5.0
            )
            logger.info(f"Track published successfully with sid: {publication.sid}")
            return publication
            
        except asyncio.TimeoutError:
            logger.error(f"Track publication timed out (attempt {retry_count + 1})")
            retry_count += 1
        except Exception as e:
            if "track already published" in str(e):
                logger.info("Track was already published successfully")
                return None
            logger.error(f"Failed to publish track (attempt {retry_count + 1}): {e}")
            logger.error(f"Error type: {type(e)}")
            retry_count += 1
        
        if retry_count < max_retries:
            await asyncio.sleep(2)  # Wait before retrying
            
    raise Exception("Failed to publish track after all retries")

async def connect_to_livekit():
    """Connect to LiveKit server"""
    url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    logger.info(f"Connecting to LiveKit server at {url}")
    
    # Generate unique identity
    identity = f"vosk_test_{uuid.uuid4().hex[:8]}"
    logger.info(f"Connecting with identity: {identity}")
    
    # Generate access token
    at = generate_token(api_key, api_secret, "playground", identity)
    logger.info("Generated access token")
    
    # Create room
    room = rtc.Room()
    
    # Set up event listeners
    @room.on("connection_state_changed")
    def on_connection_state_change(state: int):
        logger.info(f"Connection state changed to: {state}")

    @room.on("connected")
    def on_connected():
        logger.info("Successfully connected to room")
        
    @room.on("disconnected")
    def on_disconnected():
        logger.error("Disconnected from room")
        
    @room.on("participant_connected")
    def on_participant_connected(participant):
        logger.info(f"Participant connected: {participant.identity}")

    # Connect to room
    try:
        await room.connect(url, at)
        logger.info(f"Connected to room: {room.name}")
        return room
    except Exception as e:
        logger.error(f"Failed to connect to room: {e}")
        raise

async def test_transcription_pipeline():
    """Test the transcription pipeline with live microphone input"""
    logger.info("Starting live transcription test")
    
    # Load environment variables
    url = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        logger.error("Missing LIVEKIT_API_KEY or LIVEKIT_API_SECRET environment variables")
        return

    logger.info(f"Connecting to LiveKit server at {url}")
    
    # Initialize room variable
    room = None
    
    try:
        # Connect to LiveKit
        room = await connect_to_livekit()
        
        # Wait for connection to stabilize
        await asyncio.sleep(2)

        # Create audio track
        logger.info("Creating audio track...")
        audio_source = await create_audio_source()
        await audio_source.init()  # Initialize STT
        
        track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
        logger.info(f"Created audio track: {track}")
        
        # Wait for track to be ready
        logger.info("Waiting for track to be ready...")
        await asyncio.sleep(1)
        
        # Publish track
        logger.info("Publishing audio track...")
        await publish_track_with_retry(room, track)
        
        # Keep the connection alive
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.error(f"Error in transcription pipeline: {e}")
    finally:
        if 'audio_source' in locals():
            await cleanup_audio_source(audio_source)
        if 'room' in locals():
            await room.disconnect()

def generate_token(api_key: str, api_secret: str, room_name: str, identity: str) -> str:
    """Generate a LiveKit access token"""
    at = {
        "video": {
            "room": room_name,
            "roomCreate": True,
            "roomJoin": True,
            "roomAdmin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
            "canPublishSources": ["microphone"]
        },
        "iss": api_key,
        "sub": identity,
        "exp": int(time.time()) + 3600  # Token expires in 1 hour
    }
    return jwt.encode(at, api_secret, algorithm="HS256")

if __name__ == "__main__":
    try:
        asyncio.run(test_transcription_pipeline())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")