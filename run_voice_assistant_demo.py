#!/usr/bin/env python
"""
Demo script to run the voice assistant with the implemented fixes.
This script demonstrates the fixed multi-turn conversation flow,
proper Transcription API usage, and improved task management.
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import LiveKit components
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
)

# Import voice assistant components
from voice_core.agent2 import LucidiaVoiceAgent
from voice_core.state.voice_state_manager import VoiceState
from scripts.generate_token import create_token

# Constants
DEFAULT_ROOM = "demo-room"
DEFAULT_URL = "ws://localhost:7880"
DEFAULT_API_KEY = "devkey"
DEFAULT_API_SECRET = "secret"
DEFAULT_IDENTITY = f"demo-user-{datetime.now().strftime('%Y%m%d%H%M%S')}"


async def run_demo(args):
    """Run the voice assistant demo with the specified arguments."""
    logger = logging.getLogger("demo")
    logger.info(f"Starting voice assistant demo in room: {args.room}")
    
    # Create token
    token = create_token(
        api_key=args.api_key,
        api_secret=args.api_secret,
        room_name=args.room,
        identity=args.identity
    )
    
    # Create job context
    ctx = JobContext(
        url=args.url,
        token=token,
        room_name=args.room,
        identity=args.identity
    )
    
    # Connect to room
    logger.info(f"Connecting to LiveKit server at {args.url}")
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"Connected to room: {args.room}")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return
    
    # Create and initialize agent
    agent = None
    try:
        logger.info("Creating voice agent")
        agent = LucidiaVoiceAgent(ctx, "Hello! I'm the voice assistant demo with the latest fixes.")
        
        logger.info("Initializing voice agent")
        await agent.initialize()
        
        logger.info("Starting voice agent")
        await agent.start()
        
        # Find audio track
        logger.info("Looking for audio track")
        audio_track = await find_audio_track(ctx)
        
        if audio_track:
            logger.info("Found audio track, processing")
            await agent.process_audio(audio_track)
        else:
            logger.info("No audio track found, waiting for user to join")
            
            # Keep running until disconnected or interrupted
            while ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                # Check for new audio tracks every second
                audio_track = await find_audio_track(ctx)
                if audio_track:
                    logger.info("Found audio track, processing")
                    await agent.process_audio(audio_track)
                    break
                    
                await asyncio.sleep(1)
                
        # Keep running until disconnected or interrupted
        logger.info("Voice assistant running. Press Ctrl+C to exit.")
        while ctx.room and ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in voice assistant: {e}", exc_info=True)
    finally:
        # Clean up
        if agent:
            logger.info("Cleaning up agent")
            await agent.cleanup()
            
        # Disconnect
        if ctx.room:
            logger.info("Disconnecting from room")
            await ctx.room.disconnect()
            
        logger.info("Demo completed")


async def find_audio_track(ctx):
    """Find a suitable audio track from remote participants."""
    if not ctx.room:
        return None
        
    for participant in ctx.room.remote_participants.values():
        for pub in participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track:
                return pub.track
                
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run voice assistant demo")
    parser.add_argument("--room", default=DEFAULT_ROOM, help="LiveKit room name")
    parser.add_argument("--url", default=DEFAULT_URL, help="LiveKit server URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LiveKit API key")
    parser.add_argument("--api-secret", default=DEFAULT_API_SECRET, help="LiveKit API secret")
    parser.add_argument("--identity", default=DEFAULT_IDENTITY, help="Participant identity")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run the demo
    asyncio.run(run_demo(args))