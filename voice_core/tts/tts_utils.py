import edge_tts
import io
import logging
import markdown
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Union, BinaryIO

# Configure logging
logger = logging.getLogger(__name__)

# Default voice
DEFAULT_VOICE = "en-US-AvaMultilingualNeural"

async def list_voices() -> List[Dict[str, str]]:
    """
    Fetch available voices from Edge TTS and return them.
    
    Returns:
        List[Dict[str, str]]: List of voice dictionaries containing voice metadata.
    """
    logger.info("Fetching Edge TTS voices...")
    try:
        voices = await edge_tts.list_voices()
        logger.debug(f"Found {len(voices)} available voices")
        return voices
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return []

async def select_voice(voice_name: Optional[str] = None) -> str:
    """
    Get the voice to use for TTS. If voice_name is provided, validates and returns it.
    Otherwise, returns the default voice.
    """
    if not voice_name:
        voice_name = DEFAULT_VOICE
        
    # Validate the voice exists
    voices = await list_voices()
    voice_names = [v["ShortName"] for v in voices]
    
    if voice_name in voice_names:
        logger.info(f"Using voice: {voice_name}")
        return voice_name
    else:
        logger.warning(f"Voice {voice_name} not found, using default: {DEFAULT_VOICE}")
        return DEFAULT_VOICE


def markdown_to_text(markdown_string):
    """Convert Markdown to plain text."""
    try:
        html = markdown.markdown(markdown_string)
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Error converting markdown to text: {e}")
        return ""


async def text_to_speech(text: str, voice: str) -> Optional[BinaryIO]:
    """
    Convert text to audio using Edge TTS and return as BytesIO.
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice ID to use for conversion
        
    Returns:
        Optional[BinaryIO]: BytesIO containing audio data if successful, None otherwise
    """
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_data = io.BytesIO()
        
        # Track progress for longer conversions
        total_chunks = 0
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
                total_chunks += 1
                
                # Log progress for longer texts
                if total_chunks % 10 == 0:
                    logger.debug(f"Processed {total_chunks} audio chunks")
                    
        audio_data.seek(0)
        logger.info("Text-to-speech conversion complete")
        return audio_data
        
    except ConnectionError as e:
        logger.error(f"Connection error during TTS conversion: {e}")
        return None
    except OSError as e:
        logger.error(f"IO error during text-to-speech conversion: {e}")
        return None
    except RuntimeError as e:
        logger.error(f"Runtime error during text-to-speech conversion: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during text-to-speech conversion: {e}")
        return None
