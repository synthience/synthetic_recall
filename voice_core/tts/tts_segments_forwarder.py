"""
Enhanced TTSSegmentsForwarder that combines transcription API and data publishing 
for maximum UI compatibility.
"""

import asyncio
import uuid
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from livekit import rtc

logger = logging.getLogger(__name__)

@dataclass
class TTSSegment:
    """Represents a TTS text segment."""
    text: str
    id: str = ""
    final: bool = True
    language: str = "en-US"


class EnhancedTTSForwarder:
    """
    Enhanced TTS forwarder that ensures UI compatibility across different LiveKit clients.
    Publishes both transcriptions and custom data for maximum compatibility.
    """
    
    def __init__(self, room: rtc.Room, participant_identity: str):
        """
        Initialize the forwarder.
        
        Args:
            room: LiveKit room
            participant_identity: Participant identity
        """
        self.room = room
        self.participant_identity = participant_identity
        
        # Current state
        self.current_text = ""
        self.segment_id = ""
        self.is_playing = False
        
        # Track information
        self.track_sid = None
        self._find_audio_track()
        
        # Task management
        self._data_task = None
        self._publish_task = None
        self._queue = asyncio.Queue()
        self._running = True
        
        # Error handling
        self._last_error = None
        self._publish_failures = 0
        self._max_retries = 3
        
        # Start processing task
        self._publish_task = asyncio.create_task(self._process_queue())
        logger.info("Enhanced TTS forwarder initialized")
        
    def _find_audio_track(self) -> None:
        """Find audio track SID."""
        if not self.room or not self.room.local_participant:
            logger.warning("No room or local participant available")
            return
            
        for pub in self.room.local_participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                self.track_sid = pub.sid
                logger.info(f"Found audio track: {self.track_sid}")
                return
                
        logger.warning("No audio track found")
        
    async def _process_queue(self) -> None:
        """Process segments from the queue."""
        try:
            while self._running:
                try:
                    # Get segment from queue
                    segment = await self._queue.get()
                    
                    # Process segment
                    await self._publish_segment(segment)
                    
                    # Mark as done
                    self._queue.task_done()
                    
                except asyncio.CancelledError:
                    logger.info("Process queue task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing queue: {e}")
                    await asyncio.sleep(0.5)  # Prevent tight loop on error
                    
        except Exception as e:
            logger.error(f"Fatal error in process queue: {e}")
            
    async def _publish_segment(self, segment: TTSSegment) -> None:
        """
        Publish segment with multiple methods for maximum compatibility.
        
        Args:
            segment: TTS segment to publish
        """
        try:
            if not self.room or not self.room.local_participant:
                logger.warning("Cannot publish segment: no room or participant")
                return
                
            # Update state
            self.current_text = segment.text
            self.segment_id = segment.id
            
            # Method 1: Use Transcription API
            if self.track_sid:
                trans = rtc.Transcription(
                    participant_identity=self.participant_identity,
                    track_sid=self.track_sid,
                    segments=[
                        rtc.TranscriptionSegment(
                            id=segment.id,
                            text=segment.text,
                            start_time=0,
                            end_time=0,
                            final=segment.final,
                            language=segment.language
                        )
                    ]
                )
                await self.room.local_participant.publish_transcription(trans)
                logger.debug(f"Published transcription: {segment.text[:30]}...")
                
            # Method 2: Custom data messages for UI compatibility
            await self._publish_data_message(
                "transcript", 
                {"text": segment.text, "sender": "assistant", "timestamp": time.time()}
            )
            
            # Method 3: Additional UI type messages for custom UIs
            await self._publish_data_message(
                "tts_segment",
                {"text": segment.text, "id": segment.id, "timestamp": time.time()}
            )
            
            # Reset error counter on success
            self._publish_failures = 0
            
        except Exception as e:
            self._last_error = str(e)
            self._publish_failures += 1
            logger.error(f"Failed to publish segment: {e}")
            
            # Try fallback method if available and failures are within limit
            if self._publish_failures <= self._max_retries:
                logger.info(f"Trying fallback publishing method (attempt {self._publish_failures})")
                try:
                    # Fallback to simpler data message
                    await self._publish_data_message(
                        "message",
                        {"text": segment.text, "timestamp": time.time()}
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback publishing also failed: {fallback_error}")
            
    async def _publish_data_message(self, type_name: str, payload: Dict[str, Any]) -> None:
        """
        Publish data message with retry logic.
        
        Args:
            type_name: Message type
            payload: Message payload
        """
        if not self.room or not self.room.local_participant:
            logger.warning(f"Cannot publish {type_name}: no room or participant")
            return
            
        # Prepare message
        message = {"type": type_name, **payload}
        message_data = json.dumps(message).encode()
        
        # Publish with retries
        for attempt in range(self._max_retries):
            try:
                await self.room.local_participant.publish_data(
                    message_data,
                    reliable=True
                )
                return
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise
                logger.warning(f"Publish attempt {attempt+1} failed: {e}, retrying...")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    async def push_text(self, text: str, language: str = "en-US", final: bool = True) -> None:
        """
        Push text to be forwarded.
        
        Args:
            text: Text to forward
            language: Text language
            final: Whether this is a final segment
        """
        if not text:
            return
            
        segment = TTSSegment(
            text=text,
            id=str(uuid.uuid4()),
            final=final,
            language=language
        )
        
        # Add to queue
        await self._queue.put(segment)
        
    def segment_playout_started(self) -> None:
        """Mark segment playout as started."""
        self.is_playing = True
        
        # Start data task if needed
        if not self._data_task or self._data_task.done():
            self._data_task = asyncio.create_task(self._publish_ui_state("speaking"))
            
    def segment_playout_finished(self) -> None:
        """Mark segment playout as finished."""
        self.is_playing = False
        
        # Start data task if needed
        if not self._data_task or self._data_task.done():
            self._data_task = asyncio.create_task(self._publish_ui_state("idle"))
            
    async def _publish_ui_state(self, state: str) -> None:
        """
        Publish UI state update.
        
        Args:
            state: UI state
        """
        try:
            await self._publish_data_message(
                "ui_state",
                {"state": state, "timestamp": time.time()}
            )
            
            # Also publish as agent-status for agent playground
            await self._publish_data_message(
                "agent-status",
                {"status": state, "timestamp": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to publish UI state: {e}")
            
    async def close(self) -> None:
        """Close the forwarder and clean up resources."""
        self._running = False
        
        # Cancel tasks
        if self._publish_task:
            self._publish_task.cancel()
            try:
                await self._publish_task
            except asyncio.CancelledError:
                pass
                
        if self._data_task:
            self._data_task.cancel()
            try:
                await self._data_task
            except asyncio.CancelledError:
                pass
                
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
                
        logger.info("TTS forwarder closed")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get forwarder statistics."""
        return {
            "is_playing": self.is_playing,
            "publish_failures": self._publish_failures,
            "last_error": self._last_error,
            "track_sid": self.track_sid,
            "queue_size": self._queue.qsize() if self._queue else 0
        }