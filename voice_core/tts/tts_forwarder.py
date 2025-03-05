import logging
import json
import time
import asyncio
from livekit import rtc

logger = logging.getLogger(__name__)

class EnhancedTTSForwarder:
    """Wrapper for TTSSegmentsForwarder with proper UI synchronization"""
    def __init__(self, room, participant, audio_source=None):
        self.room = room
        self.participant = participant
        self.audio_source = audio_source
        self._active = False
        
        # Initialize the LiveKit TTSSegmentsForwarder
        from livekit.agents.transcription.tts_forwarder import TTSSegmentsForwarder
        self.forwarder = TTSSegmentsForwarder(
            room=room,
            participant=participant,
            language="en",
            speed=1.0
        )
        
        logger.info("Enhanced TTS Forwarder initialized")

    async def _setup_track_sid(self):
        """Set up track_sid for proper UI synchronization"""
        if not self.participant:
            logger.warning("No participant available for track setup")
            return
            
        try:
            # Wait for up to 5 seconds for an audio track to be published
            for _ in range(50):  # 50 * 0.1s = 5s
                try:
                    # Try getting track directly from participant's track_publications
                    if hasattr(self.participant, 'track_publications'):
                        for pub in self.participant.track_publications.values():
                            if pub.kind == rtc.TrackKind.AUDIO:
                                if pub.sid:
                                    self.forwarder.track_sid = pub.sid
                                    logger.info(f"Set track_sid to {pub.sid}")
                                    return
                                    
                    # Try getting track from published_tracks
                    elif hasattr(self.participant, 'published_tracks'):
                        for track in self.participant.published_tracks.values():
                            if isinstance(track, rtc.LocalAudioTrack):
                                if hasattr(track, 'sid') and track.sid:
                                    self.forwarder.track_sid = track.sid
                                    logger.info(f"Set track_sid to {track.sid}")
                                    return
                                    
                except Exception as e:
                    logger.debug(f"Error accessing tracks: {e}")
                    
                await asyncio.sleep(0.1)
                
            logger.warning("No audio track found after timeout")
            
        except Exception as e:
            logger.error(f"Error setting up track_sid: {e}")
            
    async def display_text(self, text, is_user=False):
        """Display text in the UI"""
        if not self.forwarder:
            logger.warning("No TTS forwarder available")
            return False
            
        try:
            self._active = True
            
            # Start a new segment
            self.forwarder.segment_playout_started()
            
            # Push text - this makes it display in the UI
            self.forwarder.push_text(text)
            
            # Mark text segment end
            self.forwarder.mark_text_segment_end()
            
            # Also publish in standard format for compatibility
            if not is_user and self.room and self.participant:
                await self.participant.publish_data(
                    json.dumps({
                        "type": "agent-message",
                        "text": text,
                        "timestamp": time.time()
                    }).encode(),
                    reliable=True
                )
                
            return True
        except Exception as e:
            logger.error(f"Error displaying text: {e}")
            return False
    
    async def process_message(self, text, tts_service):
        """Process a full message with text display and audio"""
        try:
            # Display text
            await self.display_text(text)
            
            # Use the speak method directly instead of process_text
            # This is compatible with InterruptibleTTSService
            await tts_service.speak(text)
            
            # Complete segment
            await self.complete_segment()
            return True
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.complete_segment()  # Try to complete segment anyway
            return False
    
    async def complete_segment(self):
        """Complete the current segment"""
        if self._active:
            try:
                self.forwarder.segment_playout_finished()
                self._active = False
                return True
            except Exception as e:
                logger.error(f"Error completing segment: {e}")
        return False
            
    async def close(self):
        """Close the forwarder"""
        try:
            if self._active:
                await self.complete_segment()
                
            # Close the forwarder
            if hasattr(self.forwarder, 'aclose'):
                await self.forwarder.aclose()
            elif hasattr(self.forwarder, 'close'):
                await self.forwarder.close()
                
            self.forwarder = None
            return True
        except Exception as e:
            logger.error(f"Error closing forwarder: {e}")
            return False