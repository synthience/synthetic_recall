"""Factory for creating and managing LiveKit workers and agents with proper token permissions."""

import os
import time
import logging
import asyncio
import jwt
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from functools import partial

from livekit.agents import (
    JobContext, 
    WorkerOptions, 
    JobExecutorType,
    JobRequest
)

from voice_core.config.config import LucidiaConfig, LLMConfig
from voice_core.agents.livekit_voice_agent import LiveKitVoiceAgent
from voice_core.stt.enhanced_stt_service import EnhancedSTTService
from voice_core.tts.interruptible_tts_service import InterruptibleTTSService
from voice_core.llm.llm_pipeline import LocalLLMPipeline
from voice_core.utils.pipeline_logger import PipelineLogger
from voice_core.state.voice_state_manager import VoiceStateManager

logger = logging.getLogger(__name__)

@dataclass
class WorkerConfig:
    """Configuration for worker initialization."""
    ws_url: str
    api_key: str
    api_secret: str
    executor_type: JobExecutorType = JobExecutorType.PROCESS
    dev_mode: bool = False
    initial_greeting: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'WorkerConfig':
        """Create config from environment variables."""
        return cls(
            ws_url=os.getenv('LIVEKIT_URL', 'ws://localhost:7880'),
            api_key=os.getenv('LIVEKIT_API_KEY', ''),
            api_secret=os.getenv('LIVEKIT_API_SECRET', ''),
            dev_mode=os.getenv('DEV_MODE', '').lower() == 'true',
            initial_greeting=os.getenv('INITIAL_GREETING')
        )

class VoiceWorkerFactory:
    """Factory for creating LiveKit workers with voice agents."""
    
    def __init__(self, worker_config: WorkerConfig):
        """Initialize the worker factory."""
        self.worker_config = worker_config
        self.pipeline_logger = PipelineLogger("worker_factory")
        self._token_cache = {}
        self._refresh_tasks = {}
        
    def _generate_token(self, room_name: str, identity: str) -> str:
        """Generate a new LiveKit token with proper permissions for UI updates."""
        try:
            # Set token expiration to 12 hours from now for testing reliability
            # In production, use a more reasonable expiration time (e.g., 1 hour)
            exp_time = int(time.time()) + (12 * 60 * 60)
            
            # Token claims
            # CRITICAL: Include canPublishData: true for UI updates to work
            claims = {
                "iss": self.worker_config.api_key,  # Issuer
                "sub": identity,  # Subject (participant identity)
                "exp": exp_time,  # Expiration time
                "nbf": int(time.time()) - 300,  # Not before time (allow 5 min clock skew)
                "video": {
                    "room": room_name,  # Room name
                    "roomJoin": True,  # Allow joining room
                    "canPublish": True,  # Allow publishing audio/video
                    "canSubscribe": True,  # Allow subscribing
                    "canPublishData": True,  # CRITICAL for UI updates
                    "roomAdmin": True,  # Useful for troubleshooting
                    "roomCreate": True   # Allow creating room if it doesn't exist
                },
                "name": identity,  # Participant name
                "metadata": json.dumps({  # Useful metadata for debugging
                    "type": "voice_assistant",
                    "created": time.time(),
                    "version": "1.0"
                })
            }
            
            # Generate token
            token = jwt.encode(
                claims,
                self.worker_config.api_secret,
                algorithm="HS256"
            )
            
            # Cache token with expiration
            self._token_cache[f"{room_name}:{identity}"] = {
                "token": token,
                "expires": exp_time
            }
            
            # Schedule token refresh
            self._schedule_token_refresh(room_name, identity, exp_time)
            
            # Log token details (only in debug mode, redacting sensitive parts)
            if self.worker_config.dev_mode:
                redacted_token = token[:20] + "..." + token[-20:] if token else None
                logger.debug(f"Generated token: {redacted_token} for room: {room_name}, identity: {identity}")
                logger.debug(f"Token permissions: canPublish: True, canSubscribe: True, canPublishData: True")
                
            return token
            
        except Exception as e:
            self.pipeline_logger.pipeline_error(e, {
                "stage": "token_generation",
                "room": room_name,
                "identity": identity
            })
            raise
            
    async def _refresh_token_task(self, room_name: str, identity: str):
        """Background task to refresh token before expiration."""
        try:
            while True:
                cache_key = f"{room_name}:{identity}"
                cached = self._token_cache.get(cache_key)
                
                if not cached:
                    # Token no longer in cache, stop refresh task
                    break
                    
                # Get time until expiration
                time_to_exp = cached["expires"] - time.time()
                
                # Refresh when token is within 30 minutes of expiring
                if time_to_exp <= 1800:  # 30 minutes before expiration
                    logger.info(f"Refreshing token for {identity} in room {room_name}")
                    # Generate new token
                    self._generate_token(room_name, identity)
                    
                # Check every 15 minutes
                await asyncio.sleep(900)
                
        except Exception as e:
            self.pipeline_logger.pipeline_error(e, {
                "stage": "token_refresh",
                "room": room_name,
                "identity": identity
            })
        finally:
            # Remove refresh task
            task_key = f"{room_name}:{identity}"
            if task_key in self._refresh_tasks:
                del self._refresh_tasks[task_key]
                
    def _schedule_token_refresh(self, room_name: str, identity: str, exp_time: int):
        """Schedule a token refresh task."""
        task_key = f"{room_name}:{identity}"
        
        # Cancel existing refresh task if any
        existing_task = self._refresh_tasks.get(task_key)
        if existing_task:
            existing_task.cancel()
            
        # Create new refresh task
        refresh_task = asyncio.create_task(
            self._refresh_token_task(room_name, identity)
        )
        self._refresh_tasks[task_key] = refresh_task
            
    def _get_valid_token(self, room_name: str, identity: str) -> str:
        """Get a valid token, generating new one if needed."""
        cache_key = f"{room_name}:{identity}"
        cached = self._token_cache.get(cache_key)
        
        # Check if we have a valid cached token
        if cached:
            # Add 30 minute buffer before expiration
            if cached["expires"] > (time.time() + 1800):
                return cached["token"]
                
        # Generate new token
        return self._generate_token(room_name, identity)
            
    def _create_agent_services(self, job_context: JobContext) -> Dict[str, Any]:
        """Create agent services (STT, TTS, LLM) for a job."""
        try:
            # Create configs
            config = LucidiaConfig()
            llm_config = LLMConfig()
            
            # Create state manager first (centralized state handling)
            state_manager = VoiceStateManager()
            
            # Initialize services with state manager
            stt_service = EnhancedSTTService(
                state_manager=state_manager,
                vosk_model='small',
                whisper_model='small.en',
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            tts_service = InterruptibleTTSService(
                state_manager=state_manager,
                voice=config.tts.get('voice', 'en-US-AvaMultilingualNeural')
            )
            
            llm_service = LocalLLMPipeline(llm_config)
            
            return {
                "config": config,
                "state_manager": state_manager,
                "stt_service": stt_service,
                "tts_service": tts_service,
                "llm_service": llm_service
            }
        except Exception as e:
            self.pipeline_logger.pipeline_error(e, {
                "stage": "service_initialization",
                "job_id": job_context.job_id
            })
            raise
            
    def _job_request_handler(self, request: JobRequest) -> bool:
        """Handle incoming job requests."""
        try:
            # Log request
            self.pipeline_logger._log(
                logging.INFO,
                "JobRequest",
                f"Received job request: {request.job_id}",
                job_id=request.job_id,
                room_name=request.room_name
            )
            
            # Validate request
            if not request.room_name:
                logger.warning("Rejecting job: missing room name")
                return False
                
            # Generate fresh token for the job
            try:
                identity = f"agent-{request.job_id}"
                self._get_valid_token(request.room_name, identity)
            except Exception as e:
                logger.error(f"Failed to generate token: {e}")
                return False
                
            # Accept request
            return True
            
        except Exception as e:
            logger.error(f"Error handling job request: {e}")
            return False
            
    async def _agent_entrypoint(self, job_context: JobContext) -> None:
        """Entrypoint function for agent jobs."""
        try:
            # Get fresh token
            identity = f"agent-{job_context.job_id}"
            token = self._get_valid_token(job_context.room_name, identity)
            
            # Update job context with fresh token
            job_context.token = token
            
            # Create services
            services = self._create_agent_services(job_context)
            
            # Set room in state manager first
            await job_context.connect(auto_subscribe=True)
            await services["state_manager"].set_room(job_context.room)
            
            # Create and start agent
            agent = LiveKitVoiceAgent(
                job_context=job_context,
                config=services["config"],
                state_manager=services["state_manager"],
                stt_service=services["stt_service"],
                tts_service=services["tts_service"],
                llm_service=services["llm_service"],
                initial_greeting=self.worker_config.initial_greeting
            )
            
            try:
                # Initialize agent (connects services to room)
                await agent.initialize()
                
                # Start agent with greeting
                await agent.start()
                
                # Keep agent running until disconnected
                while job_context.room and job_context.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
                    await asyncio.sleep(5)
                    
            finally:
                # Cleanup token refresh task
                task_key = f"{job_context.room_name}:{identity}"
                refresh_task = self._refresh_tasks.get(task_key)
                if refresh_task:
                    refresh_task.cancel()
                    del self._refresh_tasks[task_key]
                    
                # Cleanup agent
                await agent.cleanup()
            
        except Exception as e:
            logger.error(f"Error in agent entrypoint: {e}")
            await job_context.fail(str(e))
            raise
            
    def create_worker_options(self) -> WorkerOptions:
        """Create worker options for running agents."""
        return WorkerOptions(
            ws_url=self.worker_config.ws_url,
            api_key=self.worker_config.api_key,
            api_secret=self.worker_config.api_secret,
            executor_type=self.worker_config.executor_type,
            entrypoint_fnc=self._agent_entrypoint,
            request_fnc=self._job_request_handler,
            dev_mode=self.worker_config.dev_mode
        )
        
    async def prewarm_services(self) -> None:
        """Prewarm services for faster startup."""
        try:
            logger.info("Prewarming services")
            
            # Create dummy context for initialization
            dummy_context = JobContext(
                job_id="prewarm",
                room_name="prewarm",
                url=self.worker_config.ws_url,
                token="",
                identity="prewarm"
            )
            
            # Initialize services
            services = self._create_agent_services(dummy_context)
            
            # Prewarm STT
            await services["stt_service"].initialize()
            
            # Prewarm TTS
            await services["tts_service"].initialize()
            
            # Prewarm LLM
            await services["llm_service"].initialize()
            
            logger.info("Services prewarmed successfully")
            
        except Exception as e:
            logger.error(f"Error prewarming services: {e}")
            raise
            
def create_worker(config: Optional[WorkerConfig] = None) -> WorkerOptions:
    """Create a worker with the given config or from environment."""
    if config is None:
        config = WorkerConfig.from_env()
        
    factory = VoiceWorkerFactory(config)
    
    # Prewarm services if not in dev mode
    if not config.dev_mode:
        asyncio.create_task(factory.prewarm_services())
        
    return factory.create_worker_options()