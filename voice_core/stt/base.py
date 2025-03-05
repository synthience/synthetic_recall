"""Base class for Speech-to-Text services."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from livekit import rtc

class STTService(ABC):
    """Abstract base class for Speech-to-Text services."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the STT service."""
        pass

    @abstractmethod
    async def process_audio(self, track: Optional[rtc.AudioTrack]) -> Optional[str]:
        """Process audio from track and return transcription."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
