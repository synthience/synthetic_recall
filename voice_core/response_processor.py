import threading
import queue
import re
import asyncio
import logging
from typing import Iterable, Tuple, Union

from voice_core.shared_state import should_interrupt
from voice_core.tts_utils import text_to_speech, markdown_to_text
from voice_core.llm_communication import update_conversation_history
from voice_core.audio_playback import playback_worker

# Configure logging for detailed traceability.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum word threshold for processing a text chunk.
THRESHOLD_WORDS: int = 20

# TTS timeout settings
MIN_TTS_TIMEOUT: float = 2.0  # Minimum timeout for any TTS conversion
BASE_TTS_TIMEOUT: float = 5.0  # Base timeout for longer text
CHARS_PER_SECOND: float = 15.0  # Expected TTS processing speed


async def async_process_word(text: str, voice: str, playback_queue: queue.Queue) -> bool:
    """
    Asynchronously convert a text chunk to speech and enqueue the resulting audio data.
    
    Args:
        text (str): The text chunk to convert.
        voice (str): The desired TTS voice.
        playback_queue (queue.Queue): Queue where resulting audio data is enqueued.
    
    Returns:
        bool: True if conversion succeeded and audio data was enqueued; False otherwise.
    """
    if should_interrupt.is_set():
        logger.debug("Interrupt set before processing; skipping TTS conversion.")
        return False

    if len(text.strip()) < 2:
        logger.debug("Text chunk is too short; skipping TTS conversion.")
        return False

    logger.info(f"Starting TTS conversion for chunk: '{text}'")
    try:
        # Calculate timeout based on text length with a minimum threshold
        # For short phrases, use MIN_TTS_TIMEOUT
        # For longer text, scale based on character count but cap at 10 seconds
        char_count = len(text.strip())
        if char_count < 20:  # Very short phrases
            chunk_timeout = MIN_TTS_TIMEOUT
        else:
            # Calculate expected time based on character count
            expected_time = char_count / CHARS_PER_SECOND
            chunk_timeout = min(BASE_TTS_TIMEOUT + expected_time, 10.0)
        
        logger.debug(f"Using {chunk_timeout:.1f}s timeout for {char_count} characters")
        audio_data = await asyncio.wait_for(text_to_speech(text, voice), timeout=chunk_timeout)
        
        if audio_data and not should_interrupt.is_set():
            logger.info("TTS conversion succeeded; enqueuing audio data for playback.")
            playback_queue.put(audio_data)
            return True
        else:
            logger.info("TTS conversion produced no audio data or was interrupted post-conversion.")
            return False
            
    except asyncio.TimeoutError:
        logger.warning(f"TTS conversion timed out after {chunk_timeout:.1f}s for text: '{text}'")
        return False
    except Exception as e:
        logger.error(f"Error during TTS conversion: {e}")
        return False


class WorkerManager:
    """
    Manages TTS and playback worker threads.
    
    This class encapsulates a TTS worker (which uses its own asyncio event loop)
    and a playback worker. It exposes helper methods to enqueue text for TTS conversion
    and cleanly shuts down the worker threads.
    """
    def __init__(self, voice: str) -> None:
        self.voice: str = voice
        self.tts_queue: queue.Queue[Tuple[Union[str, None], Union[str, None]]] = queue.Queue()
        self.playback_queue: queue.Queue = queue.Queue()
        self.tts_thread: threading.Thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.playback_thread: threading.Thread = threading.Thread(target=playback_worker, args=(self.playback_queue,), daemon=True)

    def start(self) -> None:
        """
        Start both TTS and playback worker threads.
        """
        self.tts_thread.start()
        self.playback_thread.start()
        logger.info("WorkerManager: Started TTS and playback threads.")

    def _tts_worker(self) -> None:
        """
        Worker thread function that processes TTS tasks.
        
        Runs an asyncio event loop to process text chunks asynchronously.
        Terminates on receiving a sentinel or if an interrupt is detected.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("TTS worker: Event loop created.")
        try:
            while not should_interrupt.is_set():
                try:
                    item = self.tts_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Sentinel check: (None, None) signals termination.
                if item[0] is None:
                    logger.info("TTS worker: Received termination sentinel.")
                    self.tts_queue.task_done()
                    break

                text, voice = item
                logger.info(f"TTS worker: Processing text '{text}' with voice '{voice}'.")
                try:
                    loop.run_until_complete(async_process_word(text, voice, self.playback_queue))
                except Exception as e:
                    logger.error(f"TTS worker: Error processing text '{text}': {e}")
                finally:
                    self.tts_queue.task_done()
        finally:
            loop.close()
            logger.info("TTS worker: Event loop closed.")

    def enqueue(self, text: str) -> None:
        """
        Enqueue a text chunk for TTS processing.
        
        Args:
            text (str): The text chunk to be processed.
        """
        logger.info(f"WorkerManager: Enqueuing text chunk: '{text}'")
        self.tts_queue.put((text, self.voice))

    def join(self) -> None:
        """
        Wait for all enqueued tasks to complete and then cleanly terminate worker threads.
        """
        self.tts_queue.join()
        self.playback_queue.join()
        # Send sentinel values to signal termination.
        self.tts_queue.put((None, None))
        self.playback_queue.put(None)
        self.tts_thread.join()
        self.playback_thread.join()
        logger.info("WorkerManager: Worker threads have terminated.")


def enqueue_chunks_for_streaming(response: Iterable[str], manager: WorkerManager) -> str:
    """
    Process a streaming response by accumulating text chunks, splitting on sentence
    boundaries (or by word threshold), and enqueuing each chunk for TTS.
    
    Args:
        response (Iterable[str]): Generator or iterable yielding text chunks.
        manager (WorkerManager): Manager instance for enqueuing TTS tasks.
    
    Returns:
        str: The complete response text accumulated from the stream.
    """
    buffer = ""
    complete_response = ""
    for chunk in response:
        if should_interrupt.is_set():
            logger.info("Interrupt detected; aborting streaming chunk enqueuing.")
            break
        if not chunk:
            continue

        buffer += chunk
        complete_response += chunk
        logger.debug(f"Streaming: Accumulated buffer length is {len(buffer)} characters.")

        # Split text on sentence boundaries using regex with lookbehind.
        sentences = re.split(r'(?<=[.!?])\s+', buffer)
        if len(sentences) > 1 or len(buffer.split()) >= THRESHOLD_WORDS:
            if len(sentences) > 1:
                *complete_sentences, buffer = sentences
                to_process = " ".join(complete_sentences)
                logger.info(f"Streaming: Extracted {len(complete_sentences)} complete sentence(s) for TTS.")
            else:
                to_process = buffer
                buffer = ""
                logger.info("Streaming: Buffer reached word threshold for TTS processing.")

            if to_process.strip():
                manager.enqueue(to_process.strip())

    if buffer.strip() and not should_interrupt.is_set():
        logger.info(f"Streaming: Enqueuing final TTS chunk from buffer: '{buffer.strip()}'")
        manager.enqueue(buffer.strip())

    return complete_response


def enqueue_chunks_for_non_streaming(response: str, manager: WorkerManager) -> None:
    """
    Process a non-streaming response by converting Markdown to plain text,
    splitting it into natural sentence chunks, and enqueuing each for TTS.
    
    Args:
        response (str): The complete response text in Markdown.
        manager (WorkerManager): Manager instance for enqueuing TTS tasks.
    """
    plain_text = markdown_to_text(response)
    sentences = re.split(r'(?<=[.!?])\s+', plain_text)
    current_chunk = ""
    for sentence in sentences:
        current_chunk += sentence + " "
        if len(current_chunk.split()) >= THRESHOLD_WORDS:
            if current_chunk.strip():
                manager.enqueue(current_chunk.strip())
            current_chunk = ""
    if current_chunk.strip():
        manager.enqueue(current_chunk.strip())
    if plain_text.strip():
        logger.info("Non-streaming: Updating conversation history with assistant response.")
        update_conversation_history("assistant", plain_text.strip())


async def process_response(
    response: Union[Iterable[str], str],
    voice: str,
    streaming: bool = False
) -> None:
    """
    Process a TTS response (streaming or non-streaming) and coordinate playback.
    
    This function creates a WorkerManager to handle TTS and playback tasks. For a
    streaming response, it accumulates text chunks, splits them appropriately, and
    enqueues each for TTS conversion. For non-streaming responses, it converts Markdown
    to plain text, splits the text into natural chunks, and enqueues them.
    
    After enqueuing, it waits for all tasks to complete, updates conversation history,
    and terminates the worker threads.
    
    Args:
        response (Union[Iterable[str], str]): The response text or generator of text chunks.
        voice (str): The TTS voice parameter.
        streaming (bool): Flag indicating whether the response is streaming.
    """
    logger.info("Processing response for TTS and playback.")
    manager = WorkerManager(voice)
    manager.start()

    if streaming:
        logger.info("Processing streaming response.")
        complete_response = enqueue_chunks_for_streaming(response, manager)
        if complete_response.strip():
            logger.info("Streaming: Updating conversation history with full response.")
            update_conversation_history("assistant", complete_response.strip())
    else:
        logger.info("Processing non-streaming response.")
        enqueue_chunks_for_non_streaming(response, manager)

    logger.info("Waiting for all TTS and playback tasks to complete.")
    manager.join()
    logger.info("Response processing complete; all tasks finished.")
