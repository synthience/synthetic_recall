# voice_core/stt/livekit_identity_manager.py
import logging
from typing import Optional
import livekit.rtc as rtc

logger = logging.getLogger(__name__)

class LiveKitIdentityManager:
    """
    Manages LiveKit participant identity tracking for accurate transcript attribution.
    Extracts participant identity from tracks in a consistent manner.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_participant_identity(self, track: rtc.AudioTrack, room: Optional[rtc.Room] = None) -> Optional[str]:
        """
        Extract participant identity from LiveKit track with fallbacks.
        
        Args:
            track: The audio track to identify
            room: Optional room object for additional lookup methods
            
        Returns:
            Participant identity or None if not identifiable
        """
        participant_identity = None
        
        # Method 1: Direct participant identity from track
        if hasattr(track, 'participant') and track.participant:
            if hasattr(track.participant, 'identity'):
                participant_identity = track.participant.identity
                self.logger.debug(f"Got identity from track.participant: {participant_identity}")
                return participant_identity
        
        # Method 2: Look up by track SID in room participants
        if not participant_identity and room and hasattr(track, 'sid'):
            track_sid = track.sid
            for participant in room.remote_participants.values():
                for pub in participant.track_publications.values():
                    if pub.track and pub.track.sid == track_sid:
                        participant_identity = participant.identity
                        self.logger.debug(f"Found identity by track SID lookup: {participant_identity}")
                        return participant_identity
        
        # Method 3: Check stream if available
        if not participant_identity and hasattr(track, 'stream_id'):
            stream_id = track.stream_id
            if stream_id and "-" in stream_id:
                # Sometimes the stream ID contains the participant identity
                parts = stream_id.split("-")
                if len(parts) >= 2:
                    participant_identity = parts[0]
                    self.logger.debug(f"Extracted identity from stream ID: {participant_identity}")
                    return participant_identity
                
        self.logger.warning(f"Could not determine participant identity for track {track.sid if hasattr(track, 'sid') else 'unknown'}")
        return "unknown_user"  # Default fallback
    
    def get_stats(self) -> dict:
        """Get statistics about identity resolution."""
        return {
            "identity_manager_active": True,
        }