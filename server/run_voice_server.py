import asyncio
import logging
from server.websocket_server import WebSocketServer, WebSocketMessage
from voice_core.voice_handler import VoiceHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create voice handler instance and make it globally available
voice_handler = VoiceHandler()

async def ensure_session(client_id: str):
    """Ensure a session exists for the client"""
    if client_id not in voice_handler.sessions:
        await voice_handler.initialize_session(client_id)
    return voice_handler.sessions[client_id]

async def handle_voice_input(message: WebSocketMessage) -> dict:
    """Handle voice input messages"""
    session = await ensure_session(message.client_id)
    return await voice_handler.handle_voice_input(message.data, message.client_id)

async def handle_session_control(message: WebSocketMessage) -> dict:
    """Handle session control messages"""
    session = await ensure_session(message.client_id)
    return await voice_handler.handle_session_control(message.data, message.client_id)

async def handle_start_listening(message: WebSocketMessage) -> dict:
    """Handle start listening messages"""
    session = await ensure_session(message.client_id)
    await voice_handler.handle_start_listening(session)
    return {"status": "started"}

async def handle_stop_listening(message: WebSocketMessage) -> dict:
    """Handle stop listening messages"""
    session = await ensure_session(message.client_id)
    await voice_handler.handle_stop_listening(session)
    return {"status": "stopped"}

async def main():
    try:
        # Create and configure WebSocket server
        server = WebSocketServer(host="127.0.0.1", port=5410)
        
        # Register specific handlers for each message type
        server.register_handler("voice_input", handle_voice_input)
        server.register_handler("session_control", handle_session_control)
        server.register_handler("start_listening", handle_start_listening)
        server.register_handler("stop_listening", handle_stop_listening)
        
        # Start server
        logger.info("Starting voice WebSocket server...")
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await voice_handler.cleanup()
        await server.stop()
    except Exception as e:
        logger.error(f"Error running server: {str(e)}")
        await voice_handler.cleanup()
        await server.stop()
        raise

if __name__ == "__main__":
    asyncio.run(main())
