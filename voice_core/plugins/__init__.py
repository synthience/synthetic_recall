"""Voice core plugins for enhanced functionality."""

from .tts_segments_forwarder import TTSSegment, TTSSegmentsForwarder
from .turn_detector import TurnConfig, TurnDetector

__all__ = [
    'TTSSegment',
    'TTSSegmentsForwarder',
    'TurnConfig',
    'TurnDetector',
]
