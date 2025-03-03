import asyncio
import edge_tts
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
import logging
import livekit
from livekit import Room, RoomOptions, LocalTrack, TrackKind

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FILE = "/tmp/test.mp3"
app = Flask(__name__)
CORS(app, supports_credentials=True)

class LiveKitTTS:
    def __init__(self, url: str = "ws://localhost:7880", api_key: str = "devkey", api_secret: str = "secret"):
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room = None
        self.participant = None
        self.audio_track = None
        logger.info("Initialized LiveKitTTS")
        
    async def connect(self, room_name: str):
        """Connect to LiveKit room for TTS"""
        try:
            options = RoomOptions(
                auto_subscribe=False,  # We don't need to subscribe to other participants
                adaptive_stream=False,
                dynacast=False
            )
            
            self.room = Room(options=options)
            await self.room.connect(
                self.url,
                token=self.api_key,  # Using api_key as token for development
                participant_name="tts_service",
                room_name=room_name
            )
            
            self.participant = self.room.local_participant
            self.audio_track = await LocalTrack.create_audio_track("tts")
            await self.participant.publish_track(self.audio_track)
            logger.info(f"Connected to LiveKit room {room_name} for TTS")
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit: {e}")
            raise
            
    async def synthesize_speech(self, text: str, voice: str = "en-US-AvaMultilingualNeural"):
        """Synthesize speech and stream it through LiveKit"""
        if not self.audio_track:
            logger.error("No audio track available. Make sure to connect first.")
            return
            
        try:
            communicate = edge_tts.Communicate(text, voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Convert audio chunk to proper format for LiveKit
                    await self.audio_track.write_frame(chunk["data"])
                    
            logger.info("Speech synthesis completed")
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from LiveKit room"""
        if self.room:
            await self.room.disconnect()
            self.room = None
            self.participant = None
            self.audio_track = None
            logger.info("Disconnected from LiveKit room")

async def get_available_voices():
    """Get list of available voices"""
    try:
        voices = await edge_tts.list_voices()
        return [voice["Name"] for voice in voices]
    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        return []

livekit_tts = LiveKitTTS()

@app.route('/tts/stream', methods=['POST'])
async def stream_audio_route():
    data = request.get_json()
    text = data['text']
    voice = data.get('voice', 'en-US-AvaMultilingualNeural')
    
    await livekit_tts.connect("tts_room")
    await livekit_tts.synthesize_speech(text, voice)
    await livekit_tts.disconnect()
    
    return jsonify({"message": "TTS streamed successfully"})

@app.route('/voices', methods=['GET'])
async def voices():
    try:
        voices = await get_available_voices()
        return jsonify({"voices": voices})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8001)