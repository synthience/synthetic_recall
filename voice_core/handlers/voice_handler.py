import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging
import os
from voice_core.llm_communication import get_llm_response
from voice_core.response_processor import process_response
from voice_core.tts_utils import select_voice
from voice_core.conversation_manager import ConversationManager
from voice_core.custom_speech_recognition import StreamingRecognizer
from voice_core.mic_utils import select_microphone
from voice_core.shared_state import interrupt_handler, should_interrupt
from voice_core.livekit_stt_service import LiveKitSTTService
from voice_core.livekit_tts_service import LiveKitTTSService
from voice_core.config.config import LucidiaConfig
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pre-selected voice
VOICE_NAME = "en-US-AvaMultilingualNeural"

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

# Tensor server configuration
TENSOR_SERVER_URL = os.getenv("TENSOR_SERVER_URL", "ws://localhost:5001")

@dataclass
class VoiceSession:
    conversation_manager: ConversationManager
    voice: str
    active: bool = True
    recognizer: Optional[StreamingRecognizer] = None
    mic_device: Optional[str] = None
    stt_service: Optional[LiveKitSTTService] = None
    tts_service: Optional[LiveKitTTSService] = None
    room_name: Optional[str] = None
    tensor_ws: Optional[websockets.WebSocketClientProtocol] = None

class VoiceHandler:
    def __init__(self):
        self.sessions: Dict[str, VoiceSession] = {}
        self.config = LucidiaConfig()
        
    async def initialize_session(self, client_id: str) -> VoiceSession:
        """Initialize a new voice session for a client"""
        if client_id in self.sessions:
            await self.cleanup_session(client_id)
            
        # Initialize conversation manager
        conversation_manager = ConversationManager()
        
        # Select voice
        voice = await select_voice(VOICE_NAME)
        
        # Create session
        session = VoiceSession(
            conversation_manager=conversation_manager,
            voice=voice
        )
        
        # Initialize LiveKit services
        session.stt_service = LiveKitSTTService(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            on_transcript=lambda text: self.handle_transcript(client_id, text)
        )
        
        session.tts_service = LiveKitTTSService(
            url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        
        # Connect to tensor server
        try:
            session.tensor_ws = await websockets.connect(TENSOR_SERVER_URL)
            logger.info(f"Connected to tensor server for client {client_id}")
        except Exception as e:
            logger.error(f"Failed to connect to tensor server: {e}")
            
        self.sessions[client_id] = session
        return session

    async def handle_transcript(self, client_id: str, text: str):
        """Handle transcribed text by sending it to tensor server"""
        session = self.sessions.get(client_id)
        if not session or not session.tensor_ws:
            return
            
        try:
            # Send transcript to tensor server
            await session.tensor_ws.send(json.dumps({
                'type': 'transcript',
                'text': text,
                'client_id': client_id
            }))
            
            # Wait for processed response
            response = await session.tensor_ws.recv()
            response_data = json.loads(response)
            
            if response_data['type'] == 'response':
                # Send to TTS service
                await session.tts_service.synthesize_speech(
                    response_data['text'],
                    session.voice
                )
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")

    async def handle_voice_message(self, message: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle incoming voice messages"""
        try:
            # Get or create session
            session = await self.initialize_session(client_id)
            
            message_type = message.get('type', '')
            if message_type == 'voice_input':
                return await self.handle_voice_input(message, session)
            elif message_type == 'session_control':
                return await self.handle_session_control(message, session)
            elif message_type == 'start_listening':
                return await self.handle_start_listening(session)
            elif message_type == 'stop_listening':
                return await self.handle_stop_listening(session)
            elif message_type == 'livekit_connect':
                return await self.handle_livekit_connect(message, session)
            else:
                return {
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                }
                
        except Exception as e:
            logger.error(f"Error handling voice message: {str(e)}")
            return {
                "type": "error",
                "error": str(e)
            }

    async def handle_livekit_connect(self, message: Dict[str, Any], session: VoiceSession) -> Dict[str, Any]:
        """Handle LiveKit connection request"""
        try:
            if not session.stt_service or not session.tts_service:
                return {"type": "error", "error": "LiveKit services not initialized"}
                
            token = message.get('token')
            if not token:
                return {"type": "error", "error": "Token not provided"}
                
            # Connect both services
            await session.stt_service.connect(LIVEKIT_URL, token, session.room_name)
            await session.tts_service.connect(LIVEKIT_URL, token, session.room_name)
            
            return {
                "type": "livekit_connected",
                "room": session.room_name,
                "stt_room": f"{session.room_name}_stt",
                "tts_room": f"{session.room_name}_tts"
            }
        except Exception as e:
            return {"type": "error", "error": str(e)}

    async def handle_start_listening(self, session: VoiceSession) -> Dict[str, Any]:
        """Start listening for voice input"""
        if session.stt_service:
            # Using LiveKit for audio processing
            return {
                "type": "listening_started",
                "mode": "livekit"
            }
        elif session.recognizer:
            # Fallback to local recognition
            try:
                async def text_callback(text: str):
                    await self.process_voice_input(text, session)
                    
                session.recognizer.set_text_callback(text_callback)
                session.recognizer.start()
                return {
                    "type": "listening_started",
                    "mode": "local"
                }
            except Exception as e:
                return {"type": "error", "error": str(e)}
        else:
            return {
                "type": "error",
                "error": "No audio processing service available"
            }

    async def handle_stop_listening(self, session: VoiceSession) -> Dict[str, Any]:
        """Stop listening for voice input"""
        if session.stt_service:
            await session.stt_service.disconnect()
            
        if session.tts_service:
            await session.tts_service.disconnect()
            
        if session.recognizer:
            session.recognizer.stop()
            
        return {"type": "listening_stopped"}

    async def process_voice_input(self, text: str, session: VoiceSession):
        """Process transcribed voice input"""
        try:
            # Get LLM response
            response = await get_llm_response(text, session.conversation_manager)
            
            # Process response
            processed_response = await process_response(response)
            
            if session.tts_service:
                # Send response through LiveKit TTS
                await session.tts_service.synthesize_speech(processed_response)
            else:
                # Handle response through existing pipeline
                # (implement your existing TTS handling here)
                pass
                
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            
    async def cleanup_session(self, client_id: str):
        """Cleanup session resources"""
        session = self.sessions.get(client_id)
        if session:
            if session.tensor_ws:
                await session.tensor_ws.close()
            if session.stt_service:
                await session.stt_service.cleanup()
            if session.tts_service:
                await session.tts_service.cleanup()
            self.sessions.pop(client_id)
