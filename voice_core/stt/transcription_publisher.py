# voice_core/stt/transcription_publisher.py
import logging
import json
import time
import uuid
from typing import Dict, Any, Optional
import livekit.rtc as rtc

logger = logging.getLogger(__name__)

class TranscriptionPublisher:
    """
    Publishes transcriptions to LiveKit with correct speaker identity.
    Ensures transcripts are properly attributed in both data messages and Transcription API.
    """
    
    def __init__(self, state_manager):
        """
        Initialize the transcription publisher.
        
        Args:
            state_manager: Voice state manager instance
        """
        self.state_manager = state_manager
        self.room = None
        self._transcript_sequence = 0
        self._publish_stats = {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0
        }
        self.logger = logging.getLogger(__name__)
        
    def set_room(self, room: rtc.Room) -> None:
        """
        Set the LiveKit room for publishing.
        
        Args:
            room: LiveKit room instance
        """
        self.room = room
        
    async def publish_transcript(
        self, 
        text: str, 
        participant_identity: str, 
        is_final: bool = True,
        confidence: float = 1.0
    ) -> bool:
        """
        Publish transcript with correct identity attribution.
        
        Args:
            text: Transcript text
            participant_identity: Participant identity for attribution
            is_final: Whether this is a final transcript
            confidence: Confidence score for the transcript
            
        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False
            
        success = True
        self._transcript_sequence += 1
        
        try:
            # 1. Publish via state manager if available
            if self.state_manager:
                try:
                    await self.state_manager.publish_transcription(
                        text,
                        "user",  # Clearly identify sender type
                        is_final,
                        participant_identity=participant_identity
                    )
                    self.logger.debug(f"Published transcript via state manager: '{text[:30]}...'")
                    self._publish_stats["successes"] += 1
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to publish via state manager: {e}")
                    success = False
                    self._publish_stats["failures"] += 1
                
            # 2. Fallback: Direct data channel publish
            if not success and self.room and self.room.local_participant:
                try:
                    # Prepare message
                    message = {
                        "type": "transcript",
                        "text": text,
                        "sender": "user",
                        "participant_identity": participant_identity,
                        "sequence": self._transcript_sequence,
                        "timestamp": time.time(),
                        "is_final": is_final,
                        "confidence": confidence
                    }
                    
                    # Publish with retry
                    await self._publish_with_retry(json.dumps(message).encode(), "transcript")
                    
                    self.logger.debug(f"Published transcript via data channel: '{text[:30]}...'")
                    self._publish_stats["successes"] += 1
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Failed to publish via data channel: {e}")
                    success = False
                    self._publish_stats["failures"] += 1
                
            # 3. Fallback: Transcription API for LiveKit compatibility
            if not success and self.room and self.room.local_participant:
                try:
                    # Find suitable track_sid
                    track_sid = self._find_track_sid(participant_identity)
                    
                    if track_sid:
                        # Create transcription
                        segment_id = str(uuid.uuid4())
                        current_time = int(time.time() * 1000)  # milliseconds
                        
                        trans = rtc.Transcription(
                            participant_identity=participant_identity,
                            track_sid=track_sid,
                            segments=[
                                rtc.TranscriptionSegment(
                                    id=segment_id,
                                    text=text,
                                    start_time=current_time,
                                    end_time=current_time,
                                    final=is_final,
                                    language="en"
                                )
                            ]
                        )
                        
                        await self.room.local_participant.publish_transcription(trans)
                        self.logger.debug(f"Published via Transcription API for '{participant_identity}'")
                        self._publish_stats["successes"] += 1
                        return True
                    else:
                        self.logger.warning(f"No track_sid found for {participant_identity}")
                        self._publish_stats["failures"] += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to publish via Transcription API: {e}")
                    self._publish_stats["failures"] += 1
                    
            return False
                
        except Exception as e:
            self.logger.error(f"Error in publish_transcript: {e}")
            self._publish_stats["failures"] += 1
            return False
            
    def _find_track_sid(self, participant_identity: str) -> Optional[str]:
        """
        Find the audio track SID for a participant.
        
        Args:
            participant_identity: Participant identity to search for
            
        Returns:
            Track SID if found, None otherwise
        """
        if not self.room:
            return None
            
        # Search for track by participant identity
        for participant in self.room.remote_participants.values():
            if participant.identity == participant_identity:
                for pub in participant.track_publications.values():
                    if pub.kind == rtc.TrackKind.KIND_AUDIO:
                        return pub.sid
                        
        # If not found, use any audio track as fallback
        for participant in self.room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.kind == rtc.TrackKind.KIND_AUDIO:
                    return pub.sid
                    
        return None
        
    async def _publish_with_retry(self, data: bytes, description: str, max_retries: int = 3) -> bool:
        """
        Publish data with retry logic.
        
        Args:
            data: Data to publish
            description: Description for logging
            max_retries: Maximum retry attempts
            
        Returns:
            True if successful, False otherwise
        """
        if not self.room or not self.room.local_participant:
            return False
            
        self._publish_stats["attempts"] += 1
        
        for attempt in range(max_retries + 1):
            try:
                await self.room.local_participant.publish_data(data, reliable=True)
                
                if attempt > 0:
                    self._publish_stats["retries"] += attempt
                    
                return True
                
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(f"Failed to publish {description} after {max_retries} attempts: {e}")
                    return False
                    
                self.logger.warning(f"Publish attempt {attempt+1} failed, retrying...")
                await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
        return False
        
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        success_rate = 0
        if self._publish_stats["attempts"] > 0:
            success_rate = self._publish_stats["successes"] / self._publish_stats["attempts"]
            
        return {
            "transcript_sequence": self._transcript_sequence,
            "publish_attempts": self._publish_stats["attempts"],
            "publish_successes": self._publish_stats["successes"],
            "publish_failures": self._publish_stats["failures"],
            "publish_retries": self._publish_stats["retries"],
            "success_rate": success_rate
        }