import speech_recognition as sr
import threading
import queue
import logging
from typing import Optional, AsyncGenerator, Callable, Dict, Any, Awaitable
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class StreamingRecognizer:
    """A class for streaming audio recognition with interruption support and testing features."""

    def __init__(self, 
                device_index: Optional[int] = None,
                on_energy_update: Optional[Callable[[float], Awaitable[None]]] = None,
                on_speech_start: Optional[Callable[[], Awaitable[None]]] = None,
                on_speech_end: Optional[Callable[[], Awaitable[None]]] = None,
                on_recognition: Optional[Callable[[str, float], Awaitable[None]]] = None) -> None:
        """
        Initialize the recognizer with the specified device.
        
        Args:
            device_index (Optional[int]): The index of the microphone device to use.
        """
        # Device-related attributes
        self.device_index: Optional[int] = device_index
        self.recognizer: sr.Recognizer = sr.Recognizer()

        # Callback functions
        self._on_energy_update = on_energy_update
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_recognition = on_recognition

        # Performance monitoring and timing
        self._recognition_start_time: Optional[float] = None
        self._speech_start_time: Optional[float] = None
        
        # Thread control attributes
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.processing_speech: bool = False
        
        # Queue management
        self.text_queue: queue.Queue[str] = queue.Queue(maxsize=100)  # Bounded queue
        self.audio_buffer: queue.Queue[bytes] = queue.Queue(maxsize=1000)  # Audio buffer queue
        self.thread_pool = ThreadPoolExecutor(max_workers=2)  # Thread pool for concurrent processing
        
        # Locks for thread safety
        self._processing_lock = threading.Lock()
        self._queue_lock = threading.Lock()

        # Apply recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.5

    async def process_file(self, file_path: str) -> str:
        """
        Process an audio file and return the recognized text.
        
        Args:
            file_path (str): Path to the audio file to process
            
        Returns:
            str: The recognized text from the audio file
            
        Raises:
            Exception: If there's an error processing the file
        """
        try:
            logger.debug(f"Processing file: {file_path}")
            
            async with asyncio.Lock():  # Ensure thread-safe file processing
                # Create a new recognizer instance for file processing
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = self.recognizer.energy_threshold
                recognizer.dynamic_energy_threshold = self.recognizer.dynamic_energy_threshold
                recognizer.dynamic_energy_adjustment_damping = self.recognizer.dynamic_energy_adjustment_damping
                recognizer.dynamic_energy_ratio = self.recognizer.dynamic_energy_ratio
                recognizer.pause_threshold = self.recognizer.pause_threshold
                recognizer.phrase_threshold = self.recognizer.phrase_threshold
                recognizer.non_speaking_duration = self.recognizer.non_speaking_duration
                
                # Read the audio file
                logger.debug("Opening audio file")
                with sr.AudioFile(file_path) as source:
                    # Record the audio file data
                    logger.debug("Recording audio data")
                    audio = recognizer.record(source)
                    
                    # Process recognition with retries and backoff
                    result = await self._process_recognition_with_retries(recognizer, audio)
                    if result:
                        return result
                    
                    return ""  # Return empty string if all attempts fail
                    
        except Exception as e:
            logger.error(f"Error processing audio file: {e}", exc_info=True)
            raise Exception(f"Failed to process audio file: {str(e)}")

    async def _process_recognition_with_retries(self, recognizer: sr.Recognizer, audio: sr.AudioData, max_retries: int = 3) -> Optional[str]:
        """Process recognition with retries and exponential backoff"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"Starting recognition attempt {attempt + 1}")
                
                # Adjust recognition parameters based on attempt
                if attempt == 1:
                    recognizer.energy_threshold = 150
                    recognizer.dynamic_energy_threshold = False
                elif attempt == 2:
                    recognizer.pause_threshold = 1.0
                    recognizer.phrase_threshold = 0.5
                
                # Use thread pool for recognition to prevent blocking
                text = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    recognizer.recognize_google,
                    audio
                )
                
                if text:
                    logger.debug(f"Recognition complete: {text}")
                    return text
                    
            except sr.UnknownValueError:
                if attempt == max_retries - 1:
                    logger.warning("Speech not recognized in any attempt")
                    return None
                    
                # Exponential backoff between retries
                await asyncio.sleep(0.5 * (2 ** attempt))
                logger.debug(f"Recognition attempt {attempt + 1} failed, trying different settings")
                continue
                
            except sr.RequestError as e:
                logger.error(f"Could not request results from speech recognition service: {e}")
                raise
        
        return None

    def start(self) -> None:
        """Start the recognition process."""
        with self._processing_lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._recognition_thread)
                self.thread.daemon = True
                self.thread.start()
                logger.info("Started recognition process.")

    def stop(self) -> None:
        """Stop the audio processing and wait for the thread to terminate."""
        with self._processing_lock:
            if self.running:
                self.running = False
                logger.info("Stopping audio processing...")
                
                # Clear queues
                with self._queue_lock:
                    while not self.text_queue.empty():
                        try:
                            self.text_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    while not self.audio_buffer.empty():
                        try:
                            self.audio_buffer.get_nowait()
                        except queue.Empty:
                            break
                
                if self.thread:
                    self.thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish
                    if self.thread.is_alive():
                        logger.warning("Audio processing thread did not terminate cleanly")
                    else:
                        logger.info("Audio processing thread stopped.")
                
                # Shutdown thread pool
                self.thread_pool.shutdown(wait=False)

    def get_text(self) -> str:
        """
        Retrieve recognized text from the queue.
        
        Returns:
            str: The next recognized text, or an empty string if the queue is empty.
        """
        try:
            with self._queue_lock:
                return self.text_queue.get_nowait()
        except queue.Empty:
            return ""

    def _recognition_thread(self) -> None:
        """Main recognition thread that processes audio data"""
        while self.running:
            try:
                # Process audio from buffer
                if not self.audio_buffer.empty():
                    with self._queue_lock:
                        audio_data = self.audio_buffer.get_nowait()
                        
                    if audio_data:
                        self.processing_speech = True
                        try:
                            # Process audio chunk
                            text = self.recognizer.recognize_google(audio_data)
                            if text:
                                with self._queue_lock:
                                    if self.text_queue.full():
                                        # Remove oldest item if queue is full
                                        try:
                                            self.text_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                    self.text_queue.put(text)
                        finally:
                            self.processing_speech = False
                            
            except Exception as e:
                logger.error(f"Error in recognition thread: {e}")
                
            # Small sleep to prevent tight loop
            time.sleep(0.1)
