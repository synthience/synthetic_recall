"""Voice pipeline metrics tracking."""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class TimingMetric:
    """Timing metric for a pipeline stage."""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    count: int = 0
    error_count: int = 0
    cancel_count: int = 0

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def end(self, error: bool = False, cancelled: bool = False):
        """End timing and update metrics."""
        self.end_time = time.time()
        self.duration += self.end_time - self.start_time
        self.count += 1
        if error:
            self.error_count += 1
        if cancelled:
            self.cancel_count += 1

class VoicePipelineMetrics:
    """Tracks metrics for the voice pipeline."""
    
    def __init__(self):
        """Initialize metrics."""
        self.logger = logging.getLogger(__name__)
        self._metrics: Dict[str, TimingMetric] = {
            'speech': TimingMetric(),
            'stt': TimingMetric(),
            'llm': TimingMetric(),
            'tts': TimingMetric()
        }
        self._start_time = time.time()

    def speech_start(self):
        """Mark start of speech."""
        self._metrics['speech'].start()
        self.logger.debug("Speech started")

    def speech_end(self):
        """Mark end of speech."""
        self._metrics['speech'].end()
        self.logger.debug("Speech ended")

    def start_stt(self):
        """Mark start of STT processing."""
        self._metrics['stt'].start()
        self.logger.debug("STT started")

    def end_stt(self, error: bool = False, cancelled: bool = False):
        """Mark end of STT processing."""
        self._metrics['stt'].end(error, cancelled)
        self.logger.debug("STT ended")

    def start_llm(self):
        """Mark start of LLM processing."""
        self._metrics['llm'].start()
        self.logger.debug("LLM started")

    def end_llm(self, error: bool = False, cancelled: bool = False):
        """Mark end of LLM processing."""
        self._metrics['llm'].end(error, cancelled)
        self.logger.debug("LLM ended")

    def start_tts(self):
        """Mark start of TTS processing."""
        self._metrics['tts'].start()
        self.logger.debug("TTS started")

    def end_tts(self, error: bool = False, cancelled: bool = False):
        """Mark end of TTS processing."""
        self._metrics['tts'].end(error, cancelled)
        self.logger.debug("TTS ended")

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current metrics."""
        metrics = {}
        for name, metric in self._metrics.items():
            if metric.count > 0:
                avg_duration = metric.duration / metric.count
                metrics[name] = {
                    'avg_duration': avg_duration,
                    'count': metric.count,
                    'error_rate': metric.error_count / metric.count if metric.count > 0 else 0,
                    'cancel_rate': metric.cancel_count / metric.count if metric.count > 0 else 0
                }
        return metrics

    def log_metrics(self):
        """Log current metrics."""
        metrics = self.get_metrics()
        for name, stats in metrics.items():
            self.logger.info(
                f"{name.upper()}: avg={stats['avg_duration']:.3f}s, " +
                f"count={stats['count']}, errors={stats['error_rate']:.1%}, " +
                f"cancels={stats['cancel_rate']:.1%}"
            )
