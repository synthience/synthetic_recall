from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

from livekit import rtc


@dataclass
class TTSSegment:
    text: str
    id: str = ""
    final: bool = True
    language: str = "en-US"


class TTSSegmentsForwarder:
    """
    Forwards TTS transcription segments to a LiveKit room.

    This component maintains an internal asynchronous queue of TTSSegment
    objects. As segments are added, they are packaged into a LiveKit
    Transcription object and published using the room's local participant.
    """

    def __init__(
        self,
        *,
        room: rtc.Room,
        participant: rtc.Participant | str,
        language: str = "en-US",
        speed: float = 1.0,
    ):
        self.room = room
        # If a participant object is provided, use its identity; otherwise assume a string
        self.participant = participant if isinstance(participant, str) else participant.identity
        self.language = language
        self.speed = speed
        self._queue = asyncio.Queue[Optional[TTSSegment]]()
        self._task = asyncio.create_task(self._run())

        # Retrieve the audio track SID from the participant's publications if available.
        if not isinstance(participant, str):
            audio_tracks = [
                track for track in participant.track_publications.values()
                if track.kind == rtc.TrackKind.KIND_AUDIO
            ]
            if audio_tracks:
                self.track_sid = audio_tracks[0].sid
            else:
                self.track_sid = None
        else:
            self.track_sid = None

    async def _run(self):
        """Process segments from the queue and forward them to LiveKit."""
        try:
            while True:
                segment = await self._queue.get()
                if segment is None:
                    break

                # If no audio track is available, skip processing.
                if not self.track_sid:
                    print("No audio track available for transcription")
                    continue

                # Create a transcription object using the segment data.
                transcription = rtc.Transcription(
                    participant_identity=self.participant,
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

                # Only publish if the room is connected.
                if self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    try:
                        await self.room.local_participant.publish_transcription(transcription)
                    except Exception as e:
                        print(f"Error publishing transcription: {e}")
        except Exception as e:
            print(f"Error in TTS forwarder: {e}")

    async def add_text(self, text: str, final: bool = True):
        """Add a text segment to be forwarded as a transcription."""
        segment = TTSSegment(
            text=text,
            id=str(uuid.uuid4()),
            final=final,
            language=self.language
        )
        await self._queue.put(segment)

    async def close(self):
        """Close the forwarder and wait for the processing task to complete."""
        await self._queue.put(None)
        if self._task:
            await self._task


# Example usage for testing purposes
if __name__ == "__main__":
    async def main():
        # Create dummy LiveKit objects for testing.
        class DummyParticipant:
            identity = "dummy_participant"
            async def publish_transcription(self, transcription):
                print("Publishing transcription:")
                print(transcription)
        class DummyRoom:
            connection_state = rtc.ConnectionState.CONN_CONNECTED
            local_participant = DummyParticipant()
            # Dummy implementations for compatibility.
            def __init__(self):
                self.track_publications = {}
        dummy_room = DummyRoom()

        # Instantiate the forwarder with a dummy participant.
        forwarder = TTSSegmentsForwarder(room=dummy_room, participant="dummy_participant", language="en-US")
        await forwarder.add_text("Hello, this is a test transcription.", final=True)
        await forwarder.close()

    asyncio.run(main())
