"""LiveKit diagnostic tool for troubleshooting UI update issues."""

import asyncio
import logging
import time
import json
import sys
import argparse
import jwt
from typing import Dict, Any, List, Optional
import os
import socket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import LiveKit SDK with fallback options
try:
    import livekit
    import livekit.rtc as rtc
    logger.info(f"LiveKit SDK version: {livekit.__version__}")
except ImportError:
    logger.error("LiveKit SDK not found. Install with: pip install livekit")
    logger.info("Trying alternative import...")
    try:
        from livekit import rtc
        logger.info("LiveKit rtc module imported through alternative path")
    except ImportError:
        logger.error("Could not import LiveKit rtc. Please install the SDK properly.")
        sys.exit(1)

# Default settings
DEFAULT_URL = "ws://localhost:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "secret"
DEFAULT_ROOM = "test-room"

class LiveKitDiagnosticTool:
    """
    Diagnostic tool for LiveKit UI publishing issues.
    Tests various publishing methods to help diagnose UI update problems.
    """
    
    def __init__(self, url: str, api_key: str, api_secret: str, room_name: str):
        """Initialize the diagnostic tool."""
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.room_name = room_name
        self.identity = f"diagnostic_{int(time.time())}"
        
        # Connection state
        self.room = None
        self.log_messages = []
        
        # Publishing success tracking
        self.successful_publishes = {
            "data": 0,
            "transcription": 0
        }
        self.failed_publishes = {
            "data": 0,
            "transcription": 0
        }
        
    def log(self, level: int, message: str, **kwargs) -> None:
        """Log message with timestamp and optional metadata."""
        timestamp = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
        
        entry = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "level": logging.getLevelName(level),
            "message": message,
            **kwargs
        }
        
        self.log_messages.append(entry)
        logger.log(level, message)
        
    def generate_token(self) -> str:
        """Generate JWT token for LiveKit with proper permissions."""
        exp_time = int(time.time()) + 3600  # 1 hour validity
        
        claims = {
            "iss": self.api_key,
            "sub": self.identity,
            "exp": exp_time,
            "nbf": int(time.time()) - 60,  # Valid from 1 minute ago (allow for clock drift)
            "video": {
                "room": self.room_name,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,  # Critical for UI updates
                "roomAdmin": True,       # Helpful for diagnostics
                "roomCreate": True       # Create room if needed
            },
            "metadata": json.dumps({
                "type": "diagnostic", 
                "version": "1.0"
            })
        }
        
        token = jwt.encode(claims, self.api_secret, algorithm="HS256")
        self.log(logging.INFO, "Token generated with permissions", permissions=claims["video"])
        return token
        
    async def network_diagnostics(self) -> bool:
        """Run network diagnostics for LiveKit connection."""
        self.log(logging.INFO, "Running network diagnostics...")
        
        # Parse URL
        if self.url.startswith("ws://"):
            host = self.url[5:].split(":")[0]
            port = int(self.url.split(":")[-1])
            secure = False
        elif self.url.startswith("wss://"):
            host = self.url[6:].split(":")[0]
            port = int(self.url.split(":")[-1]) if ":" in self.url[6:] else 443
            secure = True
        else:
            host = self.url
            port = 7880
            secure = False
            
        # DNS lookup
        try:
            self.log(logging.INFO, f"DNS lookup for {host}")
            ip_address = socket.gethostbyname(host)
            self.log(logging.INFO, f"DNS lookup successful: {ip_address}")
        except socket.gaierror as e:
            self.log(logging.ERROR, f"DNS lookup failed: {e}")
            return False
            
        # Socket connection test
        try:
            self.log(logging.INFO, f"Testing socket connection to {host}:{port}")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                self.log(logging.INFO, f"Socket connection successful")
            else:
                self.log(logging.ERROR, f"Socket connection failed with error {result}")
                return False
        except Exception as e:
            self.log(logging.ERROR, f"Socket connection test failed: {e}")
            return False
            
        # Basic HTTP(S) connectivity test
        try:
            import aiohttp
            test_url = f"{'https' if secure else 'http'}://{host}:{port}/rtc"
            self.log(logging.INFO, f"Testing HTTP connectivity to {test_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url, timeout=5) as response:
                    if response.status != 404:
                        self.log(logging.WARNING, f"Unexpected response from LiveKit server: {response.status}")
                    else:
                        self.log(logging.INFO, f"HTTP connectivity test successful (404 expected)")
        except Exception as e:
            self.log(logging.WARNING, f"HTTP connectivity test failed: {e}")
            # Continue anyway - this is just an extra check
            
        return True
        
    async def test_publish_methods(self) -> Dict[str, Any]:
        """Test various publishing methods and return results."""
        results = {
            "data_publish": False,
            "transcription_publish": False,
            "data_methods_tested": 0,
            "transcription_methods_tested": 0,
            "errors": []
        }
        
        if not self.room or self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
            self.log(logging.ERROR, "Room not connected, cannot test publishing")
            results["errors"].append("Room not connected")
            return results
            
        if not self.room.local_participant:
            self.log(logging.ERROR, "No local participant, cannot test publishing")
            results["errors"].append("No local participant")
            return results
            
        # 1. Test basic data publishing
        try:
            results["data_methods_tested"] += 1
            test_data = json.dumps({
                "type": "diagnostic_test",
                "message": "Basic data test",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(test_data, reliable=True)
            self.log(logging.INFO, "Basic data publishing successful")
            results["data_publish"] = True
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Basic data publishing failed: {e}")
            results["errors"].append(f"Basic data publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 2. Test publishing UI state message
        try:
            results["data_methods_tested"] += 1
            ui_data = json.dumps({
                "type": "state_update",
                "state": "listening",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(ui_data, reliable=True)
            self.log(logging.INFO, "UI state data publishing successful")
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"UI state data publishing failed: {e}")
            results["errors"].append(f"UI state publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 3. Test publishing transcript data
        try:
            results["data_methods_tested"] += 1
            transcript_data = json.dumps({
                "type": "transcript",
                "text": "This is a test transcript",
                "sender": "diagnostic",
                "timestamp": time.time()
            }).encode()
            
            await self.room.local_participant.publish_data(transcript_data, reliable=True)
            self.log(logging.INFO, "Transcript data publishing successful")
            self.successful_publishes["data"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Transcript data publishing failed: {e}")
            results["errors"].append(f"Transcript data publishing: {str(e)}")
            self.failed_publishes["data"] += 1
            
        # 4. Test publishing transcription API
        # First publish a local audio track to get a track_sid
        local_track = None
        try:
            audio_source = rtc.AudioSource()
            local_track = rtc.LocalAudioTrack.create_audio_track("diagnostic_audio", audio_source)
            await self.room.local_participant.publish_track(local_track)
            self.log(logging.INFO, "Audio track published successfully")
            
            # Now test transcription API
            results["transcription_methods_tested"] += 1
            trans = rtc.Transcription(
                text="Test transcription API",
                participant_identity=self.identity
            )
            await self.room.local_participant.publish_transcription(trans)
            self.log(logging.INFO, "Transcription API publishing successful")
            results["transcription_publish"] = True
            self.successful_publishes["transcription"] += 1
        except Exception as e:
            self.log(logging.ERROR, f"Transcription API publishing failed: {e}")
            results["errors"].append(f"Transcription API publishing: {str(e)}")
            self.failed_publishes["transcription"] += 1
        finally:
            # Clean up track
            if local_track:
                try:
                    await self.room.local_participant.unpublish_track(local_track)
                except:
                    pass
                    
        return results
        
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run full diagnostics and return results."""
        start_time = time.time()
        self.log(logging.INFO, "Starting LiveKit UI diagnostics")
        
        # Run network diagnostics first
        network_ok = await self.network_diagnostics()
        if not network_ok:
            self.log(logging.ERROR, "Network diagnostics failed, cannot continue")
            return {
                "success": False,
                "stage": "network",
                "duration": time.time() - start_time,
                "logs": self.log_messages
            }
            
        # Generate token
        token = self.generate_token()
        
        # Connect to room
        try:
            self.log(logging.INFO, f"Connecting to room: {self.room_name}")
            self.room = rtc.Room()
            await self.room.connect(self.url, token)
            
            # Wait for connection to stabilize
            for _ in range(5):  # Wait up to 5 seconds
                if self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    break
                await asyncio.sleep(1)
                
            if self.room.connection_state != rtc.ConnectionState.CONN_CONNECTED:
                self.log(logging.ERROR, f"Failed to connect to room: {self.room.connection_state}")
                return {
                    "success": False,
                    "stage": "connection",
                    "duration": time.time() - start_time,
                    "logs": self.log_messages
                }
                
            self.log(logging.INFO, "Successfully connected to room")
            
            # Test publish methods
            publish_results = await self.test_publish_methods()
            
            # Final results
            success = publish_results["data_publish"] or publish_results["transcription_publish"]
            diagnostic_results = {
                "success": success,
                "duration": time.time() - start_time,
                "network_ok": network_ok,
                "room_connected": self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED,
                "publish_results": publish_results,
                "publish_stats": {
                    "successful": self.successful_publishes,
                    "failed": self.failed_publishes
                },
                "logs": self.log_messages,
                "recommendations": []
            }
            
            # Add recommendations based on results
            if not success:
                diagnostic_results["recommendations"].append(
                    "Check token permissions, especially 'canPublishData: true'"
                )
                
            if self.failed_publishes["data"] > 0 and self.successful_publishes["data"] == 0:
                diagnostic_results["recommendations"].append(
                    "Check Docker networking configuration (exposing port 7880)"
                )
                
            if self.failed_publishes["transcription"] > 0 and self.successful_publishes["transcription"] == 0:
                diagnostic_results["recommendations"].append(
                    "Check LiveKit version compatibility (Transcription API requires newer versions)"
                )
                
            if success:
                diagnostic_results["recommendations"].append(
                    "UI updates should be working. If still having issues, check client-side subscription."
                )
                
            return diagnostic_results
            
        except Exception as e:
            self.log(logging.ERROR, f"Error during diagnostics: {e}")
            return {
                "success": False,
                "error": str(e),
                "stage": "unknown",
                "duration": time.time() - start_time,
                "logs": self.log_messages
            }
        finally:
            # Clean up
            if self.room:
                try:
                    await self.room.disconnect()
                except:
                    pass

async def print_diagnostic_results(results: Dict[str, Any]) -> None:
    """Print diagnostic results in a readable format."""
    print("\n" + "=" * 50)
    print("LIVEKIT UI CONNECTION DIAGNOSTIC RESULTS")
    print("=" * 50)
    
    print(f"\nOverall success: {'✅' if results['success'] else '❌'}")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if 'network_ok' in results:
        print(f"\nNetwork connectivity: {'✅' if results['network_ok'] else '❌'}")
        
    if 'room_connected' in results:
        print(f"Room connection: {'✅' if results['room_connected'] else '❌'}")
        
    if 'publish_results' in results:
        pr = results['publish_results']
        print("\nPublishing tests:")
        print(f"  Data publishing: {'✅' if pr['data_publish'] else '❌'} ({pr['data_methods_tested']} methods tested)")
        print(f"  Transcription publishing: {'✅' if pr['transcription_publish'] else '❌'} ({pr['transcription_methods_tested']} methods tested)")
        
        if pr.get('errors'):
            print("\nPublishing errors:")
            for err in pr['errors']:
                print(f"  - {err}")
                
    if 'publish_stats' in results:
        ps = results['publish_stats']
        print("\nPublishing statistics:")
        print(f"  Successful data publishes: {ps['successful']['data']}")
        print(f"  Failed data publishes: {ps['failed']['data']}")
        print(f"  Successful transcription publishes: {ps['successful']['transcription']}")
        print(f"  Failed transcription publishes: {ps['failed']['transcription']}")
        
    if 'recommendations' in results:
        print("\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
            
    print("\nDetailed logs:")
    for log in results['logs'][-10:]:  # Show last 10 logs
        level_color = "\033[92m" if log['level'] == "INFO" else "\033[91m" if log['level'] == "ERROR" else "\033[93m"
        reset_color = "\033[0m"
        print(f"  {log['formatted_time']} {level_color}{log['level']}{reset_color}: {log['message']}")
        
    print("\n" + "=" * 50)
    
    if results['success']:
        print("\n✅ DIAGNOSIS: UI updates should be working correctly.")
        print("If issues persist, check your client-side subscription and UI code.")
    else:
        print("\n❌ DIAGNOSIS: UI updates are not working correctly.")
        print("Follow the recommendations above to fix the issues.")
        
    print("=" * 50 + "\n")

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LiveKit UI Connection Diagnostic Tool")
    parser.add_argument("--url", default=DEFAULT_URL, help="LiveKit server URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LiveKit API key")
    parser.add_argument("--api-secret", default=DEFAULT_API_SECRET, help="LiveKit API secret")
    parser.add_argument("--room", default=DEFAULT_ROOM, help="Room name")
    parser.add_argument("--output", choices=["console", "json"], default="console", help="Output format")
    
    args = parser.parse_args()
    
    # Allow environment variable overrides
    url = os.environ.get("LIVEKIT_URL", args.url)
    api_key = os.environ.get("LIVEKIT_API_KEY", args.api_key)
    api_secret = os.environ.get("LIVEKIT_API_SECRET", args.api_secret)
    room = os.environ.get("LIVEKIT_ROOM", args.room)
    
    # Run diagnostics
    tool = LiveKitDiagnosticTool(url, api_key, api_secret, room)
    results = await tool.run_diagnostics()
    
    # Output results
    if args.output == "json":
        print(json.dumps(results, default=str, indent=2))
    else:
        await print_diagnostic_results(results)

if __name__ == "__main__":
    asyncio.run(main())
