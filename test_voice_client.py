import asyncio
import websockets
import json
import logging
from typing import Optional, Dict, Any
import keyboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoiceClient:
    def __init__(self, url: str = "ws://127.0.0.1:5410"):
        self.url = url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.client_id: Optional[str] = None
        self.is_listening: bool = False
        
    async def connect(self):
        """Connect to the voice WebSocket server"""
        try:
            self.websocket = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=10
            )
            
            # Wait for connection status message
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get('type') == 'connection_status' and data.get('status') == 'connected':
                self.client_id = data.get('client_id')
                logger.info(f"Connected to voice server. Client ID: {self.client_id}")
                return True
            else:
                logger.error("Unexpected connection response")
                await self.websocket.close()
                self.websocket = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            return False

    async def start_listening(self) -> Dict[str, Any]:
        """Start listening for voice input"""
        if not self.websocket:
            if not await self.connect():
                return {"type": "error", "error": "Not connected"}
                
        try:
            message = {"type": "start_listening"}
            await self.websocket.send(json.dumps(message))
            response = await self.websocket.recv()
            data = json.loads(response)
            if data.get('status') == 'started':
                self.is_listening = True
                logger.info("Started listening...")
            return data
        except Exception as e:
            logger.error(f"Error starting voice input: {str(e)}")
            return {"type": "error", "error": str(e)}

    async def stop_listening(self) -> Dict[str, Any]:
        """Stop listening and get recognized text"""
        if not self.websocket:
            return {"type": "error", "error": "Not connected"}
            
        try:
            message = {"type": "stop_listening"}
            await self.websocket.send(json.dumps(message))
            response = await self.websocket.recv()
            data = json.loads(response)
            if data.get('status') != 'error':
                self.is_listening = False
                logger.info("Stopped listening.")
            return data
        except Exception as e:
            logger.error(f"Error stopping voice input: {str(e)}")
            return {"type": "error", "error": str(e)}
            
    async def control_session(self, command: str) -> Dict[str, Any]:
        """Send session control command"""
        if not self.websocket:
            if not await self.connect():
                return {"type": "error", "error": "Not connected"}
                
        message = {
            "type": "session_control",
            "command": command
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error sending control command: {str(e)}")
            return {"type": "error", "error": str(e)}
        
    async def close(self):
        """Close the WebSocket connection"""
        if self.websocket:
            try:
                if self.is_listening:
                    await self.stop_listening()
                await self.control_session("stop")
            except:
                pass
            await self.websocket.close()
            self.websocket = None
            self.client_id = None
            self.is_listening = False

async def main():
    """Run voice interaction client"""
    client = VoiceClient()
    
    try:
        # Connect to server
        logger.info("Connecting to voice server...")
        if not await client.connect():
            return
            
        logger.info("\nVoice Commands:")
        logger.info("Press SPACE to start/stop listening")
        logger.info("Press ESC to exit")
        
        while True:
            if keyboard.is_pressed('space'):
                if client.is_listening:
                    response = await client.stop_listening()
                    if response.get('text'):
                        logger.info(f"Recognized: {response['text']}")
                else:
                    await client.start_listening()
                await asyncio.sleep(0.5)  # Debounce
                
            if keyboard.is_pressed('esc'):
                logger.info("Exiting...")
                break
                
            await asyncio.sleep(0.1)  # Prevent CPU spin
            
    except Exception as e:
        logger.error(f"Error during voice interaction: {str(e)}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
