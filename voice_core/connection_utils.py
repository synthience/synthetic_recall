"""Connection utilities for LiveKit room management."""
import asyncio
import logging
from typing import Optional, Any
from livekit import rtc
from livekit.agents import JobContext
from livekit.rtc import Room

logger = logging.getLogger(__name__)

async def cleanup_connection(assistant: Optional[Any], ctx: JobContext) -> None:
    """Gracefully cleanup the connection and resources with enhanced state management."""
    if not ctx or not hasattr(ctx, 'room'):
        return

    try:
        if assistant:
            await assistant.cleanup()

        if ctx.room:
            # First attempt graceful disconnect
            try:
                async with asyncio.timeout(5.0):
                    await ctx.room.disconnect()
            except asyncio.TimeoutError:
                logger.warning("Room disconnect timed out, forcing cleanup")
            except Exception as e:
                logger.error(f"Error during room disconnect: {e}")

            # Force cleanup of internal state
            await force_room_cleanup(ctx.room)

    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}")

async def force_room_cleanup(room: Room) -> None:
    """Force cleanup of room resources."""
    if not room or not room.local_participant:
        return
        
    try:
        # Unpublish all tracks
        if hasattr(room.local_participant, 'published_tracks'):
            for track_pub in room.local_participant.published_tracks:
                try:
                    await room.local_participant.unpublish_track(track_pub)
                except Exception as e:
                    logger.error(f"Error unpublishing track {track_pub.sid}: {e}")
                
        # Close room connection
        await room.disconnect()
    except Exception as e:
        logger.error(f"Error during room cleanup: {e}")

async def wait_for_disconnect(ctx: JobContext, timeout: int = 15) -> bool:
    """
    Wait for room to fully disconnect with extended timeout.
    Returns True if disconnect confirmed, False if timeout.
    """
    try:
        async with asyncio.timeout(timeout):
            while ctx.room and (
                ctx.room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED or
                (hasattr(ctx.room, '_ws') and ctx.room._ws and ctx.room._ws.connected)
            ):
                await asyncio.sleep(0.1)
            return True
    except asyncio.TimeoutError:
        return False

async def verify_room_state(ctx: JobContext, check_connection_state: bool = True) -> bool:
    """
    Verify that the room is truly clean and ready for a new connection.
    Returns True if room is clean, False otherwise.
    """
    if not ctx or not hasattr(ctx, 'room'):
        return True

    state = {
        'ws_connected': bool(ctx.room._ws and ctx.room._ws.connected) if hasattr(ctx.room, '_ws') else False,
        'participants': len(ctx.room._participants) if hasattr(ctx.room, '_participants') else 0,
        'connection_state': ctx.room.connection_state if hasattr(ctx.room, 'connection_state') else None
    }

    if not check_connection_state:
        is_clean = (not state['ws_connected'] and state['participants'] == 0)
    else:
        is_clean = (
            not state['ws_connected'] and
            state['participants'] == 0 and
            (state['connection_state'] is None or state['connection_state'] == rtc.ConnectionState.CONN_DISCONNECTED)
        )

    if not is_clean:
        logger.warning(f"Room not fully cleaned: {state}")

    return is_clean