import threading
import asyncio
from typing import Dict, Any

# Global flag to signal interruption.
# This event can be set when, for example, the user wants to cancel ongoing speech recognition.
should_interrupt = asyncio.Event()

# Global microphone settings.
# 'selected_microphone' will store the device index or identifier of the chosen microphone.
selected_microphone = None

# Global recognizer settings.
# These settings control the sensitivity and behavior of the speech recognizer.
# Fine-tune these values based on the environment, microphone quality, and the desired balance between responsiveness and accuracy.
recognizer_settings: Dict[str, Any] = {
    # Base energy threshold for distinguishing speech from background noise.
    # A lower value makes the recognizer more sensitive, but may pick up ambient sounds.
    "energy_threshold": 300,

    # If True, the recognizer will automatically adjust the energy threshold over time
    # based on ambient noise levels. This helps maintain recognition accuracy in variable environments.
    "dynamic_energy_threshold": True,

    # The maximum length of silence (in seconds) allowed within a phrase.
    # A higher value means the recognizer will wait longer before considering a pause as the end of speech.
    "pause_threshold": 0.8,

    # The amount of non-speaking duration (in seconds) required before finalizing the speech input.
    # Setting this to a higher value (e.g., 0.8) causes the recognizer to wait longer for continued speech.
    "operation_timeout": None,

    # Additional granular settings for enhanced control:

    # The sample rate (in Hz) of the audio input.
    # A common value for many applications is 16000 Hz, balancing detail and processing load.
    "sample_rate": 16000,

    # Duration (in milliseconds) of each audio chunk processed.
    # Smaller chunks (e.g., 20 ms) allow near real-time processing but may require more frequent computation.
    "chunk_duration_ms": 20,

    # Maximum number of consecutive silent audio chunks allowed before the recognizer decides the phrase has ended.
    # Higher values (e.g., 10) allow for more natural pauses in speech.
    "max_silence_chunks": 10,

    # Minimum total phrase length (in seconds) required before processing.
    # This avoids triggering on very short utterances or noise.
    "min_phrase_length": 0.3,
    'dynamic_energy_adjustment_damping': 0.15,
    'dynamic_energy_ratio': 1.5,
}

class InterruptHandler:
    def __init__(self):
        self.interrupt_event = asyncio.Event()
        self.keywords = {"stop", "wait", "pause", "cancel"}

    async def check_for_interrupt(self, text: str) -> bool:
        """Check if text contains interrupt keywords"""
        words = text.lower().split()
        if any(kw in words for kw in self.keywords):
            self.interrupt_event.set()
            should_interrupt.set()  # Set global interrupt
            return True
        return False

    async def reset(self):
        """Reset interrupt flags"""
        self.interrupt_event.clear()
        should_interrupt.clear()

# Global interrupt handler
interrupt_handler = InterruptHandler()
