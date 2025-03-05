"""Pipeline logging utilities for voice agents."""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Metrics for voice pipeline performance tracking."""
    
    start_time: float = field(default_factory=time.time)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric with timestamp."""
        self.metrics[name] = {
            'value': value,
            'timestamp': time.time() - self.start_time
        }
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a recorded metric."""
        return self.metrics.get(name)
    
    def get_duration(self, start_event: str, end_event: str) -> Optional[float]:
        """Get duration between two events."""
        start = self.get_metric(start_event)
        end = self.get_metric(end_event)
        if start and end:
            return end['timestamp'] - start['timestamp']
        return None

class PipelineLogger:
    """Logger for voice pipeline events and metrics."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.metrics = PipelineMetrics()
        
    def _log(self, level: int, stage: str, message: str, **kwargs) -> None:
        """Internal logging with consistent format."""
        metadata = {
            'session_id': self.session_id,
            'stage': stage,
            'timestamp': datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.log(level, f"[{stage}] {message}", extra={'metadata': metadata})
        
    # STT Events
    def stt_started(self, config: Dict[str, Any]) -> None:
        """Log STT initialization."""
        self.metrics.record_metric('stt_start', config)
        self._log(logging.INFO, 'STT', 'Speech recognition started', config=config)
        
    def stt_partial(self, text: str) -> None:
        """Log partial STT results."""
        self._log(logging.DEBUG, 'STT', f'Partial transcript: {text}', text=text)
        
    def stt_final(self, text: str, confidence: float) -> None:
        """Log final STT results."""
        self.metrics.record_metric('stt_final', {'text': text, 'confidence': confidence})
        self._log(logging.INFO, 'STT', f'Final transcript: {text}', 
                 text=text, confidence=confidence)
        
    def stt_error(self, error: Exception) -> None:
        """Log STT errors."""
        self._log(logging.ERROR, 'STT', f'Recognition error: {str(error)}', 
                 error=str(error))
        
    # LLM Events
    def llm_request(self, prompt: str) -> None:
        """Log LLM request."""
        self.metrics.record_metric('llm_request', prompt)
        self._log(logging.INFO, 'LLM', 'Sending request to LLM', prompt=prompt)
        
    def llm_response(self, response: str, metadata: Dict[str, Any]) -> None:
        """Log LLM response."""
        self.metrics.record_metric('llm_response', response)
        duration = self.metrics.get_duration('llm_request', 'llm_response')
        self._log(logging.INFO, 'LLM', f'Received response in {duration:.2f}s', 
                 response=response, metadata=metadata)
        
    def llm_error(self, error: Exception) -> None:
        """Log LLM errors."""
        self._log(logging.ERROR, 'LLM', f'LLM error: {str(error)}', 
                 error=str(error))
        
    # TTS Events
    def tts_started(self, text: str, config: Dict[str, Any]) -> None:
        """Log TTS initialization."""
        self.metrics.record_metric('tts_start', {'text': text, 'config': config})
        self._log(logging.INFO, 'TTS', 'Speech synthesis started', 
                 text=text, config=config)
        
    def tts_progress(self, bytes_processed: int) -> None:
        """Log TTS progress."""
        self._log(logging.DEBUG, 'TTS', f'Generated {bytes_processed} bytes', 
                 bytes_processed=bytes_processed)
        
    def tts_complete(self, duration: float, total_bytes: int) -> None:
        """Log TTS completion."""
        self.metrics.record_metric('tts_complete', {
            'duration': duration,
            'total_bytes': total_bytes
        })
        self._log(logging.INFO, 'TTS', 
                 f'Speech synthesis completed in {duration:.2f}s ({total_bytes} bytes)',
                 duration=duration, total_bytes=total_bytes)
        
    def tts_error(self, error: Exception) -> None:
        """Log TTS errors."""
        self._log(logging.ERROR, 'TTS', f'Synthesis error: {str(error)}', 
                 error=str(error))
        
    # LiveKit Events
    def livekit_connected(self, room_name: str, participant_id: str) -> None:
        """Log LiveKit connection."""
        self.metrics.record_metric('livekit_connect', {
            'room': room_name,
            'participant_id': participant_id
        })
        self._log(logging.INFO, 'LiveKit', 'Connected to room', 
                 room=room_name, participant_id=participant_id)
        
    def livekit_track_published(self, track_id: str, kind: str) -> None:
        """Log track publication."""
        self._log(logging.INFO, 'LiveKit', f'Published {kind} track', 
                 track_id=track_id, kind=kind)
        
    def livekit_track_subscribed(self, track_id: str, kind: str) -> None:
        """Log track subscription."""
        self._log(logging.INFO, 'LiveKit', f'Subscribed to {kind} track', 
                 track_id=track_id, kind=kind)
        
    def livekit_error(self, error: Exception) -> None:
        """Log LiveKit errors."""
        self._log(logging.ERROR, 'LiveKit', f'LiveKit error: {str(error)}', 
                 error=str(error))
        
    # Pipeline Events
    def pipeline_started(self, config: Dict[str, Any]) -> None:
        """Log pipeline start."""
        self.metrics = PipelineMetrics()  # Reset metrics
        self._log(logging.INFO, 'Pipeline', 'Voice pipeline started', config=config)
        
    def pipeline_stopped(self) -> None:
        """Log pipeline stop with performance metrics."""
        total_duration = time.time() - self.metrics.start_time
        self._log(logging.INFO, 'Pipeline', 
                 f'Voice pipeline stopped after {total_duration:.2f}s',
                 total_duration=total_duration,
                 metrics=self.metrics.metrics)
        
    def pipeline_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log pipeline errors with context."""
        self._log(logging.ERROR, 'Pipeline', 
                 f'Pipeline error: {str(error)}', 
                 error=str(error), context=context)
