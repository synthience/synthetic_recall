"""Core voice application implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any

from config.settings import Settings
from interfaces import (
    DatabaseService,
    TTSService,
    STTService,
    KnowledgeService,
    AgentService,
    ObserverService,
    InterpreterService,
    WorkflowService
)
from utils.error_handling import (
    handle_errors,
    ErrorCategory,
    ErrorSeverity,
    ServiceError,
    CommandError,
    KnowledgeError,
    SpeechError
)
from utils.session_manager import SessionManager

logger = logging.getLogger(__name__)


class InterpreterVoiceLucidiaApp:
    """Enhanced Voice App that integrates multiple services through dependency injection."""
    
    def __init__(
        self,
        settings: Settings,
        services: ServiceContainer
    ):
        """Initialize the application with injected services."""
        self.settings = settings
        
        # Store injected services
        self.db = services.db
        self.tts_service = services.tts
        self.stt_service = services.stt
        self.knowledge_service = services.knowledge
        self.agent_service = services.agent
        self.observer_service = services.observer
        self.interpreter_service = services.interpreter
        self.workflow_service = services.workflow
        
        # Session management
        self.current_session_id: Optional[str] = None
        self.session_manager: Optional[SessionManager] = None
        self.running: bool = False
        
        # Voice processing state
        self._current_command: Optional[str] = None
        self._command_start_time: Optional[datetime] = None
        self._silence_duration: float = 0.0
        
    @handle_errors(
        error_category=ErrorCategory.INITIALIZATION,
        reraise=True,
        log_level=ErrorSeverity.CRITICAL
    )
    async def start(self):
        """Start the voice interpreter application."""
        session_data = {
            'name': f'Interpreter Session {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'timestamp': datetime.now().isoformat()
        }
        
        # Initialize session
        try:
            self.current_session_id = self.db.create_game_session(session_data)
            logger.info(f"Created new session: {self.current_session_id}")
            
            # Initialize session manager
            self.session_manager = SessionManager(
                session_id=self.current_session_id,
                max_chunk_duration=self.settings.voice.max_chunk_duration,
                max_chunk_words=self.settings.voice.max_chunk_words,
                max_memory_chunks=self.settings.voice.max_memory_chunks,
                auto_save_threshold=self.settings.voice.auto_save_threshold
            )
        except Exception as e:
            raise ServiceError(
                "Failed to create session",
                service_name="database",
                original_error=e
            )
        
        # Start required services
        await self._start_services()
            
        # Start main voice input loop
        self.running = True
        while self.running:
            try:
                await self._voice_input_loop()
            except Exception as e:
                logger.error(f"Error in voice input loop: {e}")
                await asyncio.sleep(0.1)
                
    @handle_errors(
        error_category=ErrorCategory.COMMAND,
        log_level=ErrorSeverity.WARNING
    )
    async def _voice_input_loop(self):
        """Handle continuous voice input processing."""
        # Listen for voice input
        success, text, confidence = await self.stt_service.listen_for_command(
            timeout=5.0,
            silence_timeout=1.0
        )
        
        if not success or not text:
            self._handle_silence()
            return
            
        # Reset silence tracking
        self._silence_duration = 0.0
        
        # Add to current session
        if self.session_manager:
            completed_chunk = await self.session_manager.add_voice_input(
                text=text,
                confidence=confidence,
                metadata={
                    "type": "voice_command",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Process completed chunk if available
            if completed_chunk:
                await self.process_voice_command(completed_chunk.text)
                
    def _handle_silence(self):
        """Handle silence in voice input."""
        self._silence_duration += 0.1  # Assuming 100ms processing loop
        
        # If we have accumulated command and hit silence threshold
        if (
            self._current_command and
            self._silence_duration >= self.settings.voice.silence_threshold
        ):
            # Process the accumulated command
            asyncio.create_task(self.process_voice_command(self._current_command))
            self._current_command = None
            self._command_start_time = None
            
    @handle_errors(
        error_category=ErrorCategory.COMMAND,
        log_level=ErrorSeverity.ERROR
    )
    async def process_voice_command(self, command: str):
        """Process voice command with context from recent history."""
        logger.info(f"Processing voice command: {command}")
        
        # Get current context including recent history
        context = await self._create_context(command)
        
        # Get knowledge insights
        context = await self._enhance_context_with_knowledge(command, context)
        
        # Process command through available handlers
        await self._process_command_handlers(command, context)
            
    async def _create_context(self, command: str) -> ReasoningContext:
        """Create context with recent session history."""
        try:
            vision_context = await self.observer_service.get_current_context()
        except Exception as e:
            logger.warning(f"Failed to get vision context: {e}")
            vision_context = {}
            
        # Get recent voice history
        recent_history = []
        if self.session_manager:
            recent_history = await self.session_manager.get_recent_context(
                num_chunks=5
            )
            
        return ReasoningContext(
            event_history=[
                *recent_history,
                {
                    "type": "voice_command",
                    "command": command,
                    "timestamp": datetime.now().isoformat(),
                    "vision_context": vision_context
                }
            ],
            system_state={
                "current_topics": ["voice_command", "automation"],
                "last_command_time": datetime.now().isoformat()
            },
            performance_metrics=self.session_manager.get_stats() if self.session_manager else {},
            timestamp=datetime.now()
        )
        
    @handle_errors(
        error_category=ErrorCategory.SERVICE,
        log_level=ErrorSeverity.ERROR
    )
    async def stop(self):
        """Stop the voice interpreter application."""
        self.running = False
        
        # Clean up session
        if self.session_manager:
            await self.session_manager.cleanup()
            
        # Stop all services
        await self._stop_services()
            
    @handle_errors(
        error_category=ErrorCategory.SERVICE,
        log_level=ErrorSeverity.ERROR
    )
    async def _start_services(self):
        """Start all required services."""
        try:
            await self.stt_service.start()
        except Exception as e:
            raise ServiceError(
                "Failed to start STT service",
                service_name="stt",
                original_error=e
            )
            
        try:
            await self.observer_service.start()
        except Exception as e:
            raise ServiceError(
                "Failed to start observer service",
                service_name="observer",
                original_error=e
            )
                
    @handle_errors(
        error_category=ErrorCategory.SERVICE,
        log_level=ErrorSeverity.ERROR
    )
    async def _stop_services(self):
        """Stop all services."""
        services_to_stop = [
            (self.stt_service.stop(), "stt"),
            (self.interpreter_service.cleanup(), "interpreter"),
            (self.workflow_service.cleanup(), "workflow"),
            (self.observer_service.stop(), "observer")
        ]
        
        for service_task, service_name in services_to_stop:
            try:
                await service_task
            except Exception as e:
                raise ServiceError(
                    f"Failed to stop {service_name} service",
                    service_name=service_name,
                    original_error=e
                )
            
    @handle_errors(
        error_category=ErrorCategory.COMMAND,
        log_level=ErrorSeverity.WARNING
    )
    async def _process_command_handlers(self, command: str, context: ReasoningContext):
        """Process command through available handlers."""
        # Try processing with agents first
        try:
            agent_response = await self.agent_service.process_command(command, context)
            if agent_response:
                await self._handle_agent_response(agent_response)
                return
        except Exception as e:
            raise CommandError(
                "Agent processing failed",
                command=command,
                original_error=e
            )
            
        # Try workflow suggestions
        try:
            workflow = await self.workflow_service.suggest_automation(command, context)
            if workflow:
                await self._handle_workflow_suggestion(workflow)
                return
        except Exception as e:
            raise CommandError(
                "Workflow suggestion failed",
                command=command,
                original_error=e
            )
            
        # Fall back to interpreter
        try:
            interpreter_response = await self.interpreter_service.execute_command(
                command,
                context.to_dict()
            )
            await self._handle_interpreter_response(interpreter_response)
        except Exception as e:
            raise CommandError(
                "Interpreter execution failed",
                command=command,
                original_error=e
            )
            
    @handle_errors(
        error_category=ErrorCategory.SPEECH,
        log_level=ErrorSeverity.WARNING
    )
    async def speak(self, text: str):
        """Convert text to speech and play it."""
        try:
            await self.tts_service.speak(text)
        except Exception as e:
            raise SpeechError(
                "Failed to speak text",
                operation="speak",
                original_error=e
            )
            
    async def _handle_agent_response(self, response: Dict[str, Any]):
        """Handle response from an agent."""
        if response.get('speak'):
            await self.speak(response['speak'])
            
    async def _handle_workflow_suggestion(self, workflow: Dict[str, Any]):
        """Handle workflow automation suggestion."""
        if workflow.get('description'):
            await self.speak(f"I can help automate that. {workflow['description']}")
            
    async def _handle_interpreter_response(self, response: Dict[str, Any]):
        """Handle interpreter command response."""
        if response.get('message'):
            await self.speak(response['message'])
