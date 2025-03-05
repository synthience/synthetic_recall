from __future__ import annotations

import asyncio
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
import logging

# Configure logging for demonstration purposes.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class TurnConfig:
    """Configuration for turn detection."""
    min_silence_duration: float = 1.0  # Minimum silence duration to trigger end of turn
    max_turn_duration: float = 30.0      # Maximum duration of a single turn
    initial_buffer_duration: float = 0.5 # Initial buffer before starting turn detection
    energy_threshold: float = -40        # Energy threshold in dB for silence detection


class TurnDetector:
    """Detect conversation turns based on audio energy and timing."""
    
    def __init__(
        self,
        config: Optional[TurnConfig] = None,
        on_turn_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_turn_end: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.config = config or TurnConfig()
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        
        self._turn_active = False
        self._last_audio_time = 0.0
        self._turn_start_time = 0.0
        self._last_energy = -60.0
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the turn detection process."""
        self.logger.info("Starting turn detector")
        self._running = True
        # Launch the monitoring loop as a background task.
        self._monitor_task = asyncio.create_task(self._monitor_turns())

    async def stop(self):
        """Stop the turn detection process."""
        self.logger.info("Stopping turn detector")
        self._running = False
        if self._monitor_task:
            await self._monitor_task
        # Allow time for any pending tasks to finish.
        await asyncio.sleep(0.1)
    
    def is_active(self) -> bool:
        """Return whether a turn is currently active."""
        return self._turn_active

    def update_audio_level(self, energy: float):
        """
        Update the current audio energy level.
        
        Args:
            energy: Raw energy value (e.g. mean squared amplitude)
        """
        # Convert energy to decibels, with a floor at -60 dB.
        energy_db = max(-60, 10 * np.log10(energy + 1e-10))
        self._last_energy = energy_db
        self._last_audio_time = time.time()
        self.logger.debug(f"Updated energy level: {energy_db:.2f} dB")

    async def _monitor_turns(self):
        """Monitor audio levels and detect turn boundaries."""
        try:
            while self._running:
                now = time.time()
                
                if self._turn_active:
                    # Calculate silence duration and turn duration.
                    silence_duration = now - self._last_audio_time
                    turn_duration = now - self._turn_start_time
                    
                    if (silence_duration >= self.config.min_silence_duration or 
                        turn_duration >= self.config.max_turn_duration):
                        self.logger.info("Turn ended (silence or max duration reached)")
                        self._turn_active = False
                        if self.on_turn_end:
                            await self.on_turn_end()
                
                else:
                    # Check if conditions to start a turn are met.
                    if (self._last_energy > self.config.energy_threshold and
                        now - self._last_audio_time <= self.config.initial_buffer_duration):
                        self.logger.info("Turn started")
                        self._turn_active = True
                        self._turn_start_time = now
                        if self.on_turn_start:
                            await self.on_turn_start()
                
                await asyncio.sleep(0.1)
        except Exception as e:
            self.logger.error(f"Error in turn detection loop: {e}")
            raise


# ------------------------- DEMONSTRATION USAGE -------------------------

async def demo_turn_detector():
    # Define asynchronous callbacks for turn events.
    async def on_turn_start():
        print(">>> Turn started!")
        
    async def on_turn_end():
        print("<<< Turn ended!")

    # Create a turn detector instance with default configuration and callbacks.
    detector = TurnDetector(on_turn_start=on_turn_start, on_turn_end=on_turn_end)
    detector.start()

    # Simulate audio level updates.
    # We'll simulate a "speech turn" by providing high energy values,
    # then simulate silence with very low energy values.
    try:
        print("Simulating speech turn (2 seconds)...")
        # Simulate speech (high energy values) for 2 seconds.
        for _ in range(20):
            # An energy value that converts to a high dB level (above threshold).
            detector.update_audio_level(energy=0.01)
            await asyncio.sleep(0.1)
        
        print("Simulating silence (1.5 seconds)...")
        # Simulate silence by using very low energy values.
        for _ in range(15):
            detector.update_audio_level(energy=1e-12)
            await asyncio.sleep(0.1)
        
        print("Simulating another speech turn (1 second)...")
        # Simulate another speech turn.
        for _ in range(10):
            detector.update_audio_level(energy=0.01)
            await asyncio.sleep(0.1)
        
        # Let the detector run a bit longer.
        await asyncio.sleep(2)
        
    finally:
        await detector.stop()
        print("Turn detector stopped.")

if __name__ == "__main__":
    asyncio.run(demo_turn_detector())
