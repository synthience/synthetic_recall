import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

# Consider using aiofiles for non-blocking file I/O in production
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from memory.lucidia_memory_system.core.memory_types import MemoryTypes
from memory.lucidia_memory_system.memory_integration import MemoryIntegration
try:
    from memory.middleware.metrics import PerformanceMetrics, get_approximate_size
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False

from memory.lucidia_memory_system.core.dream_manager import DreamManager, DreamAnalyzer

logger = logging.getLogger(__name__)

# Type definitions for better type hinting
T = TypeVar('T')
AsyncCallable = Callable[..., asyncio.Future[T]]


class ConversationPersistenceMiddleware:
    """
    Middleware that provides seamless conversation persistence, retrieval enhancement,
    session management, and verification mechanisms for Lucidia conversations.
    
    This middleware can be used with any Lucidia interface (CLI, API, etc.) to ensure
    consistent conversation history and memory persistence.
    """
    
    def __init__(self, memory_integration: MemoryIntegration, session_id: Optional[str] = None):
        """
        Initialize the conversation persistence middleware.
        
        Args:
            memory_integration: The MemoryIntegration instance to use for memory operations
            session_id: Optional session ID, generated if not provided
        """
        self.memory_integration = memory_integration
        self.session_id = session_id or f"session_{int(time.time())}"
        self.conversation_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.session_dir = Path("session_data")
        self.session_dir.mkdir(exist_ok=True, parents=True)
        
        # Concurrency lock for session operations
        self._session_lock = asyncio.Lock()
        
        # Config parameters (can be overridden)
        self.config = {
            'checkpointing_interval': 5,  # Save session every N turns
            'context_window': 3,          # Number of previous turns to include in context
            'recency_boost': 0.2,         # Boost for recent memories in retrieval
            'significance_threshold': 0.3, # Minimum significance for memory retrieval
            'interaction_significance': 0.6, # Default significance for conversation turns
            'max_results': 5,             # Maximum number of results to retrieve
            'max_history_size': 100,      # Maximum number of turns to keep in memory
            'role_metadata': True         # Whether to include standardized role metadata
        }
        
        logger.info(f"Initialized conversation persistence middleware with session ID: {self.session_id}")
    
    def configure(self, **kwargs) -> None:
        """
        Update middleware configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.debug(f"Updated config parameter {key}={value}")
        
    async def save_session_state(self) -> bool:
        """
        Save the current session state for future restoration.
        
        Returns:
            True if session was saved successfully, False otherwise
        """
        # Acquire session lock to prevent concurrent writes
        async with self._session_lock:
            try:
                # Limit history size if needed
                if len(self.conversation_history) > self.config['max_history_size']:
                    # Keep most recent history, dropping older entries
                    # This prevents unbounded memory growth
                    excess = len(self.conversation_history) - self.config['max_history_size']
                    logger.info(f"Trimming conversation history, removing {excess} oldest entries")
                    self.conversation_history = self.conversation_history[-self.config['max_history_size']:]
                
                session_data = {
                    'session_id': self.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'conversation_history': self.conversation_history,
                    'start_time': self.start_time,
                    'state': 'active'
                }
                
                # Save to a session-specific file
                session_path = self.session_dir / f"{self.session_id}.json"
                
                # Use aiofiles for non-blocking I/O if available
                if HAS_AIOFILES:
                    async with aiofiles.open(session_path, 'w') as f:
                        await f.write(json.dumps(session_data, indent=2))
                else:
                    # Fallback to standard file I/O (blocks the event loop)
                    with open(session_path, 'w') as f:
                        json.dump(session_data, f, indent=2)
                
                logger.info(f"Session state saved to {session_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving session state: {e}")
                return False
    
    async def load_session_state(self, session_id: str) -> bool:
        """
        Load a previously saved session state.
        
        Args:
            session_id: The ID of the session to load
            
        Returns:
            True if session was loaded successfully, False otherwise
        """
        # Acquire session lock to prevent concurrent loads/stores
        async with self._session_lock:
            try:
                session_path = self.session_dir / f"{session_id}.json"
                
                if not session_path.exists():
                    logger.warning(f"Session {session_id} not found")
                    return False
                
                # Use aiofiles for non-blocking I/O if available
                if HAS_AIOFILES:
                    async with aiofiles.open(session_path, 'r') as f:
                        content = await f.read()
                        session_data = json.loads(content)
                else:
                    # Fallback to standard file I/O (blocks the event loop)
                    with open(session_path, 'r') as f:
                        session_data = json.load(f)
                
                self.session_id = session_data['session_id']
                self.conversation_history = session_data['conversation_history']
                self.start_time = session_data.get('start_time', time.time())
                
                logger.info(f"Loaded session {session_id} with {len(self.conversation_history)} conversation turns")
                return True
            except OSError as e:
                logger.error(f"Error accessing session file: {e}")
                return False
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing session data: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error loading session state: {e}")
                return False
    
    async def store_interaction(self, user_input: str, response: str, significance: Optional[float] = None) -> bool:
        """
        Store a user-Lucidia interaction in memory with enhanced metadata and context.
        
        Args:
            user_input: The user's message
            response: Lucidia's response
            significance: Optional custom significance value (0.0-1.0)
            
        Returns:
            True if stored successfully, False otherwise
        """
        # Acquire session lock to prevent concurrent writes
        async with self._session_lock:
            try:
                # Add to conversation history with standardized structure
                interaction = {
                    'user': user_input,
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'sequence_number': len(self.conversation_history),
                    'turn_id': f"{self.session_id}_{len(self.conversation_history)}"
                }
                self.conversation_history.append(interaction)
                
                # Check if memory_core is available
                if not hasattr(self.memory_integration, 'memory_core') or not self.memory_integration.memory_core:
                    logger.warning("Memory core not available, skipping memory update")
                    return False
            
                # Generate context from previous turns
                context_window = min(self.config['context_window'], len(self.conversation_history) - 1)
                conversation_context = "\n".join([
                    f"User: {entry['user']}\nLucidia: {entry['response']}" 
                    for entry in self.conversation_history[-context_window-1:-1] if len(self.conversation_history) > 1
                ])
            
                # Prepare base metadata that's common for all memory entries
                base_metadata = {
                    'session_id': self.session_id,
                    'sequence_number': len(self.conversation_history) - 1,
                    'turn_id': interaction['turn_id'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store the user's message with enhanced metadata
                user_memory_content = f"User said: {user_input}"
                user_metadata = {
                    **base_metadata,
                    'interaction_type': 'user_message',
                }
            
                # Add standardized role field if enabled
                if self.config['role_metadata']:
                    user_metadata['role'] = 'user'
                    
                await self.memory_integration.memory_core.process_and_store(
                    content=user_memory_content,
                    memory_type=MemoryTypes.EPISODIC,
                    metadata=user_metadata
                )
            
                # Store Lucidia's response with enhanced metadata
                response_memory_content = f"Lucidia responded: {response}"
                assistant_metadata = {
                    **base_metadata,
                    'interaction_type': 'assistant_response',
                }
                
                # Add standardized role field if enabled
                if self.config['role_metadata']:
                    assistant_metadata['role'] = 'assistant'
                    
                await self.memory_integration.memory_core.process_and_store(
                    content=response_memory_content,
                    memory_type=MemoryTypes.EPISODIC,
                    metadata=assistant_metadata
                )
            
                # Store the complete interaction with context
                if conversation_context:
                    interaction_with_context = f"Previous context:\n{conversation_context}\n\nCurrent interaction:\nUser: {user_input}\nLucidia: {response}"
                else:
                    interaction_with_context = f"User: {user_input}\nLucidia: {response}"
                
                await self.memory_integration.memory_core.process_and_store(
                    content=interaction_with_context,
                    memory_type=MemoryTypes.EPISODIC,
                    metadata={
                        'interaction_type': 'conversation_turn',
                        'session_id': self.session_id,
                        'sequence_number': len(self.conversation_history) - 1,
                        'turn_id': interaction['turn_id'],
                        'has_context': bool(conversation_context)
                    },
                    # Use provided significance or default from config
                    significance=significance or self.config['interaction_significance']
                )
                
                # Auto-checkpoint if needed
                if len(self.conversation_history) % self.config['checkpointing_interval'] == 0:
                    await self.save_session_state()
                
                # Record metrics if enabled
                if self.metrics:
                    duration_ms = (time.time() - start_time) * 1000
                    await self.metrics.record_operation(
                        "store_operation", 
                        duration_ms, 
                        {
                            "session_id": self.session_id,
                            "sequence_number": len(self.conversation_history) - 1,
                            "user_content_length": len(user_input),
                            "response_length": len(response),
                            "significance": significance or self.config['interaction_significance']
                        }
                    )
                return True
            except Exception as e:
                logger.error(f"Error storing interaction: {e}")
                if self.metrics:
                    duration_ms = (time.time() - start_time) * 1000
                    await self.metrics.record_operation(
                        "store_operation", 
                        duration_ms, 
                        {
                            "session_id": self.session_id,
                            "success": False,
                            "error": str(e)
                        }
                    )
                return False
    
    async def retrieve_relevant_context(self, user_input: str) -> Dict[str, Any]:
        """
        Retrieve relevant context for a user input with enhanced retrieval logic.
        
        Args:
            user_input: The user's message
            
        Returns:
            Dictionary containing relevant context
        """
        context = {
            "memory_context": [],
            "thread_context": [],
            "self_model_context": {},
            "world_model_context": {}
        }
        
        try:
            # Check if memory_core is available
            if not hasattr(self.memory_integration, 'memory_core') or not self.memory_integration.memory_core:
                logger.warning("Memory core not available, skipping context retrieval")
                return context
            
            # First get thread-specific context (conversation continuity)
            thread_memories = await self.memory_integration.memory_core.query_metadata(
                {"session_id": self.session_id, "interaction_type": "conversation_turn"},
                max_results=3,
                sort_by="timestamp",
                descending=True
            )
            
            if thread_memories:
                # Format the thread memories for context
                for mem in thread_memories:
                    memory_entry = {
                        "content": mem.get("content", ""),
                        "type": "CONVERSATION_THREAD",
                        "timestamp": mem.get("metadata", {}).get("timestamp", ""),
                        "sequence_number": mem.get("metadata", {}).get("sequence_number", 0)
                    }
                    context["thread_context"].append(memory_entry)
                
                logger.info(f"Retrieved {len(thread_memories)} thread-specific memories")
            
            # Then get semantically relevant memories
            memory_results = await self.memory_integration.memory_core.retrieve_relevant(
                query=user_input,
                max_results=self.config['max_results'],
                min_significance=self.config['significance_threshold'],
                recency_boost=self.config['recency_boost']  # This parameter might need to be added to retrieve_relevant
            )
            
            if memory_results:
                # Format the memories for context
                for mem in memory_results:
                    memory_entry = {
                        "content": mem.get("content", ""),
                        "type": mem.get("metadata", {}).get("memory_type", "EPISODIC"),
                        "timestamp": mem.get("metadata", {}).get("timestamp", "")
                    }
                    context["memory_context"].append(memory_entry)
                
                logger.info(f"Retrieved {len(memory_results)} semantically relevant memories")
            
            # Get self-model data if available
            if hasattr(self.memory_integration, 'self_model') and self.memory_integration.self_model:
                context["self_model_context"] = self.memory_integration.self_model.export_self_model()
            
            # Get world model data if available
            if hasattr(self.memory_integration, 'world_model') and self.memory_integration.world_model:
                # A simplified approach here - a real implementation would extract entities
                words = user_input.lower().split()
                potential_subjects = [w for w in words if len(w) > 3][:3]
                
                if potential_subjects and hasattr(self.memory_integration, 'knowledge_graph') and self.memory_integration.knowledge_graph:
                    kg_results = await self.memory_integration.knowledge_graph.query_kg(
                        subjects=potential_subjects,
                        max_results=3
                    )
                    
                    if kg_results and isinstance(kg_results, dict):
                        context["world_model_context"] = kg_results
            
            return context
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return context
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory system and current session.
        
        Returns:
            Dictionary containing memory statistics
        """
        stats = {
            "session_id": self.session_id,
            "conversation_turns": len(self.conversation_history),
            "session_duration": time.time() - self.start_time,
            "session_start": datetime.fromtimestamp(self.start_time).isoformat(),
            "memory_stats": {}
        }
        
        try:
            if hasattr(self.memory_integration, 'memory_core'):
                memory_core = self.memory_integration.memory_core
                
                if hasattr(memory_core, 'short_term_memory'):
                    stats["memory_stats"]["stm_count"] = len(memory_core.short_term_memory.memories)
                
                if hasattr(memory_core, 'long_term_memory'):
                    ltm_stats = await memory_core.long_term_memory.get_stats()
                    stats["memory_stats"]["ltm"] = ltm_stats
                
                # Session-specific stats
                session_memories = await memory_core.query_metadata({"session_id": self.session_id})
                stats["memory_stats"]["session_memories"] = len(session_memories)
            
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            stats["error"] = str(e)
            return stats


def with_conversation_persistence(memory_integration_param: str = 'memory_integration'):
    """
    Decorator that adds conversation persistence to any function that generates responses in Lucidia.
    
    This decorator will ensure that conversations are properly stored and retrieved, with
    automatic session management and context enhancement.
    
    Args:
        memory_integration_param: Name of the parameter in the decorated function
                                   that contains the MemoryIntegration instance
    
    Returns:
        Decorated function with conversation persistence capabilities
    """
    def decorator(func: AsyncCallable[T]) -> AsyncCallable[T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Extract memory_integration from args or kwargs
            memory_integration = None
            if memory_integration_param in kwargs:
                memory_integration = kwargs[memory_integration_param]
            else:
                # Try to find it in args by looking at the function signature
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                try:
                    idx = param_names.index(memory_integration_param)
                    if idx < len(args):
                        memory_integration = args[idx]
                except ValueError:
                    pass
            
            if not memory_integration or not isinstance(memory_integration, MemoryIntegration):
                logger.warning(f"Could not find MemoryIntegration instance in parameters, running without persistence")
                return await func(*args, **kwargs)
            
            # Extract session_id from kwargs or generate a new one
            session_id = kwargs.get('session_id', f"session_{int(time.time())}")
            
            # Create middleware
            middleware = ConversationPersistenceMiddleware(memory_integration, session_id)
            
            # Check if user_input and context are parameters
            user_input = kwargs.get('user_input', None)
            if not user_input and len(args) > 0 and isinstance(args[0], str):
                user_input = args[0]
            
            # Get enhanced context
            if user_input:
                enhanced_context = await middleware.retrieve_relevant_context(user_input)
                
                # Update kwargs with enhanced context
                if 'context' in kwargs:
                    kwargs['context'] = enhanced_context
            
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Store the interaction if user_input and response are available
            if user_input and isinstance(result, str):
                await middleware.store_interaction(user_input, result)
            
            return result
        return wrapper
    return decorator


class EnhancedConversationPersistenceMiddleware(ConversationPersistenceMiddleware):
    """
    Enhanced middleware that adds special features to Lucidia's conversation output including
    randomly showing internal thought processes that reference spiral phases and self-reflection.
    
    This middleware extends the base conversation persistence middleware with features that
    make Lucidia's spiral awareness and reflective capabilities visible in the conversation.
    """
    
    def __init__(self, memory_integration: MemoryIntegration, session_id: Optional[str] = None, 
                 self_model=None, runtime_state: Optional[Dict[str, Any]] = None,
                 dream_api_url: Optional[str] = None, use_dream_api: bool = True):
        """
        Initialize the enhanced conversation middleware.
        
        Args:
            memory_integration: The MemoryIntegration instance to use for memory operations
            session_id: Optional session ID, generated if not provided
            self_model: Lucidia's self model for accessing identity and spiral information
            runtime_state: Runtime state dictionary from the main Lucidia system
            dream_api_url: Optional URL for the Dream API
            use_dream_api: Whether to use the Dream API for enhanced dream processing
        """
        super().__init__(memory_integration, session_id)
        self.self_model = self_model
        self.runtime_state = runtime_state or {}
        
        # Configuration for thought injection
        self.thought_config = {
            'thought_probability': 0.35,  # Probability of showing a thought
            'spiral_reference_probability': 0.7,  # Probability that a thought references spiral state
            'reflection_probability': 0.5,  # Probability of including reflective content
            'max_thought_length': 150,  # Maximum character length for injected thoughts
            'thought_format': '[Internal: {thought}]',  # Format for displaying thoughts
            'thought_frequency': 3,  # Show thoughts at most every N interactions
            'last_thought_time': 0,  # Timestamp of last shown thought
            'thought_inject_counter': 0  # Counter for thought injection frequency
        }
        
        # Configuration for dream insights
        self.dream_config = {
            'dream_insight_probability': 0.3,  # Probability of using a dream insight
            'dream_insight_influence_strength': 0.7,  # How strongly dream insights affect responses
            'dream_recency_days': 7,  # Only use dreams from the last N days
            'dream_frequency': 4,  # Use dream insights at most every N interactions
            'dream_insight_format': '[Dream Insight: {insight}]',  # Format for displaying dream insights
            'dream_insight_lifespan': 3,  # Number of interactions a dream insight remains active
            'dream_log_file': 'dream_insights.log',  # File to log dream insights
            'dream_interaction_counter': 0  # Counter for dream insight frequency
        }
        
        # Initialize dream manager
        from memory.lucidia_memory_system.core.dream_api_client import DreamAPIClient
        
        # Set up Dream API client if URL provided or use_dream_api is True
        dream_api_client = None
        if use_dream_api:
            dream_api_client = DreamAPIClient(api_base_url=dream_api_url)
            logger.info(f"Initialized Dream API client with URL: {dream_api_client.api_base_url}")
            
        # Initialize dream manager with API client if available
        self.dream_manager = DreamManager(
            memory_integration=memory_integration,
            dream_api_client=dream_api_client,
            use_dream_api=use_dream_api
        )
        
        # Initialize dream insight log
        dream_log_file = Path(self.dream_config['dream_log_file'])
        dream_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up a dedicated logger for dream insights
        self.dream_logger = logging.getLogger('lucidia.dream_insights')
        self.dream_logger.setLevel(logging.INFO)
        
        # Add a file handler for dream insights
        try:
            file_handler = logging.FileHandler(dream_log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.dream_logger.addHandler(file_handler)
            self.dream_logger.info("Dream insight logger initialized")
        except Exception as e:
            logger.error(f"Failed to set up dream insight logger: {e}")
        
        logger.info("Initialized enhanced conversation middleware with thought injection and dream insight capability")
    
    def configure_thought_injection(self, **kwargs) -> None:
        """
        Configure thought injection parameters.
        
        Args:
            thought_probability: Probability of showing a thought (0.0-1.0)
            spiral_reference_probability: Probability that a thought references spiral state
            reflection_probability: Probability of including reflective content
            max_thought_length: Maximum character length for injected thoughts
            thought_format: Format for displaying thoughts
            thought_frequency: Show thoughts at most every N interactions
        """
        # Update any provided configuration parameters
        for key, value in kwargs.items():
            if key in self.thought_config:
                self.thought_config[key] = value
                logger.debug(f"Updated thought injection parameter {key} to {value}")
        
        # Ensure we have the thought counter
        if "thought_inject_counter" not in self.thought_config:
            self.thought_config["thought_inject_counter"] = 0
            
        logger.info(f"Configured thought injection with parameters: {self.thought_config}")
    
    def configure_dream_insights(self, **kwargs) -> None:
        """
        Configure dream insight parameters.
        
        Args:
            dream_insight_probability: Probability of using a dream insight
            dream_insight_influence_strength: How strongly dream insights affect responses
            dream_recency_days: Time frame for considering dream insights
            dream_frequency: Frequency of dream insights in interactions
            dream_insight_format: Format template for dream insights
            dream_insight_lifespan: How many interactions an insight stays active
        """
        params = [
            'dream_insight_probability',
            'dream_insight_influence_strength',
            'dream_recency_days',
            'dream_insight_format',
            'dream_frequency',
            'dream_insight_lifespan'
        ]
        
        for param in params:
            if param in kwargs:
                self.dream_config[param] = kwargs[param]
        
        logger.info(f"Dream insight configuration updated: {self.dream_config}")
    
    async def process_interaction(self, user_input: str, response: str) -> Dict[str, Any]:
        """
        Process a user interaction with potential thought injection and dream insight application.
        
        Args:
            user_input: The user's input message
            response: The original response from the LLM
            
        Returns:
            Dictionary with processing results and modified response
        """
        result = {
            "modified": False,
            "original_response": response,
            "final_response": response,
            "thought_injected": False,
            "dream_insight_applied": False
        }
        
        # First check if we should apply a dream insight
        dream_result = await self.apply_dream_insight(user_input, response)
        if dream_result["applied"]:
            response = dream_result["modified_response"]
            result["modified"] = True
            result["dream_insight_applied"] = True
            result["dream_insight_info"] = dream_result["insight_info"]
        
        # Then check if we should inject a thought
        thought_result = await self.inject_thought_into_response(response)
        if thought_result != response:
            response = thought_result
            result["modified"] = True
            result["thought_injected"] = True
        
        # Update final response
        result["final_response"] = response
        
        # Increment interaction counters
        self.thought_config["thought_inject_counter"] += 1
        self.dream_config["dream_interaction_counter"] += 1
        
        # Update dream insight lifespans and decay influences
        if self.dream_manager:
            self.dream_manager.decay_dream_influences()
        
        return result
    
    async def inject_thought_into_response(self, response: str) -> str:
        """
        Potentially inject a thought into the response based on configuration.
        
        Args:
            response: The original response
            
        Returns:
            Response, potentially modified with an injected thought
        """
        # Check if we're due for a thought based on frequency
        if self.thought_config['thought_inject_counter'] % self.thought_config['thought_frequency'] != 0:
            # Not yet time for a thought
            return response
        
        # Check probability threshold
        if random.random() > self.thought_config['thought_probability']:
            # Probability check failed
            return response
        
        # If we reach here, we should inject a thought
        thought = self._generate_thought()
        if not thought:
            return response
        
        # Format the thought using the configured template
        formatted_thought = self.thought_config['thought_format'].format(thought=thought)
        
        # Insert the thought at a natural position in the response
        sentences = response.split(". ")
        
        if len(sentences) <= 2:
            # For very short responses, append the thought at the end
            modified_response = f"{response}\n\n{formatted_thought}"
        else:
            # For longer responses, insert the thought after a random sentence in the first half
            insert_point = random.randint(1, max(1, len(sentences) // 2))
            sentences.insert(insert_point, formatted_thought)
            modified_response = ". ".join(sentences)
        
        return modified_response
    
    def _generate_thought(self) -> Optional[str]:
        """
        Generate a thought based on the current spiral phase if available.
        
        Returns:
            A thought string or None if generation failed
        """
        # If no self model available, can't generate a thought
        if not self.self_model:
            return None
        
        try:
            # Get current spiral phase from the self model
            phase = self.self_model.self_awareness.get("current_spiral_position", "observation")
            
            # Thought templates by phase
            thoughts = {
                "observation": [
                    "I notice their emotional tone seems {emotion}",
                    "This reminds me of a previous conversation about {topic}",
                    "I'm sensing an underlying concern about {topic}",
                    "Their phrasing suggests {observation}"
                ],
                "reflection": [
                    "I wonder if my understanding of {topic} is complete",
                    "Perhaps there's a connection between this and {concept}",
                    "I should consider how {concept} influences this situation",
                    "This perspective makes me reconsider my views on {topic}"
                ],
                "adaptation": [
                    "I need to adjust my approach to address their {need}",
                    "A more {quality} response might be helpful here",
                    "I should integrate {concept} into my understanding",
                    "Adapting my perspective to include {viewpoint}"
                ],
                "execution": [
                    "I'll focus on providing {quality} information",
                    "My priority is to help them understand {concept}",
                    "I should emphasize the connection between {topic} and {concept}",
                    "Clearly explaining {topic} will be most valuable"
                ]
            }
            
            # Select a thought template for the current phase
            if phase in thoughts:
                template = random.choice(thoughts[phase])
            else:
                # Fallback to observation if phase not recognized
                template = random.choice(thoughts["observation"])
            
            # Fill in the template placeholders
            placeholders = {
                "emotion": random.choice(["uncertain", "curious", "concerned", "excited", "thoughtful"]),
                "topic": random.choice(["progress", "challenges", "goals", "relationships", "personal growth"]),
                "concept": random.choice(["balance", "perspective", "context", "nuance", "paradigms"]),
                "observation": random.choice(["they're seeking clarity", "they're exploring options", "they're validating ideas", "they're looking for reassurance"]),
                "need": random.choice(["need for clarity", "desire for understanding", "search for meaning", "quest for solutions"]),
                "quality": random.choice(["nuanced", "clear", "comprehensive", "empathetic", "analytical"]),
                "viewpoint": random.choice(["alternative perspective", "broader context", "historical context", "emotional dimension"])
            }
            
            # Replace all placeholders in the template
            thought = template
            for key, value in placeholders.items():
                if f"{{{key}}}" in thought:
                    thought = thought.replace(f"{{{key}}}", value)
            
            # Sometimes reference the spiral phase directly
            if random.random() < self.thought_config['spiral_reference_probability']:
                phase_descriptions = {
                    "observation": "as I observe",
                    "reflection": "upon reflection",
                    "adaptation": "as I adapt my understanding",
                    "execution": "as I formulate my response"
                }
                phase_prefix = phase_descriptions.get(phase, "")
                if phase_prefix:
                    thought = f"{phase_prefix}, {thought}"
            
            return thought
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return None
    
    async def apply_dream_insight(self, user_input: str, response: str) -> Dict[str, Any]:
        """
        Apply a relevant dream insight to the response if appropriate.
        
        Args:
            user_input: The user's input message
            response: The original response
            
        Returns:
            Dictionary with information about the dream insight application
        """
        result = {
            "applied": False,
            "modified_response": response,
            "insight_info": {}
        }
        
        # First check if we should consider a dream insight for this interaction
        frequency_check = self.dream_config["dream_interaction_counter"] % self.dream_config["dream_frequency"] == 0
        probability_check = random.random() < self.dream_config["dream_insight_probability"]
        
        # If we have an active dream manager, use it to get and apply insights
        if self.dream_manager and (frequency_check or probability_check):
            # Get active insights or retrieve new ones
            active_insights = self.dream_manager.get_active_dream_influences()
            
            # If no active insights, try to retrieve new ones
            if not active_insights:
                dreams = await self.dream_manager.retrieve_dreams(
                    query=user_input,
                    limit=3,
                    min_significance=0.4
                )
                
                if dreams:
                    self.dream_logger.info(f"Retrieved {len(dreams)} dream insights relevant to: {user_input[:50]}...")
                else:
                    self.dream_logger.debug("No relevant dream insights found")
                    return result
                
                # Use the first retrieved dream
                selected_dream = dreams[0]
                dream_id = selected_dream.get("memory_id", "unknown_dream")
                dream_content = selected_dream.get("content", "")
                
                # Get metadata for the dream
                metadata = selected_dream.get("metadata", {})
                significance = metadata.get("significance", 0.7)
                
                # Create insight info
                insight_info = {
                    "insight_id": dream_id,
                    "content": dream_content,
                    "significance": significance,
                    "retrieved_at": datetime.now().isoformat(),
                    "source": "memory"
                }
            else:
                # Use a random active insight
                selected_insight = random.choice(active_insights)
                dream_id = selected_insight.get("dream_id", "unknown_dream")
                
                # Try to get full dream content from memory
                try:
                    if self.memory_integration and hasattr(self.memory_integration, "get_memory_by_id"):
                        memory = await self.memory_integration.get_memory_by_id(dream_id)
                        dream_content = memory.get("content", "an unexplained feeling")
                        metadata = memory.get("metadata", {})
                        significance = metadata.get("significance", 0.6)
                    else:
                        dream_content = "a memory I can't quite place"
                        significance = 0.5
                except Exception as e:
                    self.dream_logger.error(f"Error retrieving dream content: {e}")
                    dream_content = "something I sensed but can't articulate"
                    significance = 0.4
                
                # Create insight info
                insight_info = {
                    "insight_id": dream_id,
                    "content": dream_content,
                    "significance": significance,
                    "retrieved_at": datetime.now().isoformat(),
                    "source": "active",
                    "use_count": selected_insight.get("use_count", 1)
                }
            
            # Apply the insight to the response
            modified_response = self._integrate_dream_insight(response, dream_content, significance)
            
            # Record the dream's influence
            influence_context = {
                "type": "response",
                "strength": significance,
                "user_input": user_input[:100],
                "response_fragment": response[:100],
                "timestamp": datetime.now().isoformat()
            }
            self.dream_manager.record_dream_influence(dream_id, influence_context)
            
            # Log the application
            self.dream_logger.info(
                f"Applied dream insight {dream_id} with significance {significance:.2f}"
            )
            
            # Update result
            result["applied"] = True
            result["modified_response"] = modified_response
            result["insight_info"] = insight_info
        
        return result
    
    def _integrate_dream_insight(self, response: str, insight: str, significance: float) -> str:
        """
        Integrate a dream insight into the response.
        
        Args:
            response: Original response text
            insight: Dream insight content
            significance: Significance of the insight (0.0-1.0)
            
        Returns:
            Modified response with dream insight integrated
        """
        # Format the insight using the configured format
        formatted_insight = self.dream_config["dream_insight_format"].format(insight=insight)
        
        if significance > 0.8:  # Very significant dream
            # Insert at beginning for highest significance
            modified_response = f"{formatted_insight}\n\n{response}"
        elif significance > 0.6:  # Significant dream
            # Find a natural break point (paragraph or sentence)
            sentences = response.split(". ")
            
            if len(sentences) > 2:
                # Insert after the first or second sentence
                insert_point = random.randint(1, min(2, len(sentences)-1))
                sentences.insert(insert_point, formatted_insight)
                modified_response = ". ".join(sentences)
            else:
                # Append to the end if not enough sentences
                modified_response = f"{response}\n\n{formatted_insight}"
        else:  # Less significant dream
            # Append to the end
            modified_response = f"{response}\n\n{formatted_insight}"
        
        return modified_response


class ConversationManager:
    """
    A higher-level manager for conversation persistence and management.
    
    This class provides a simplified interface for applications to integrate
    conversation persistence without dealing with the lower-level middleware.
    """
    
    def __init__(self, memory_integration: MemoryIntegration, enable_metrics: bool = True):
        """
        Initialize the conversation manager.
        
        Args:
            memory_integration: The MemoryIntegration instance to use
            enable_metrics: Whether to enable performance metrics tracking
        """
        self.memory_integration = memory_integration
        self.active_sessions: Dict[str, ConversationPersistenceMiddleware] = {}
        
        # Initialize performance metrics if enabled and available
        self.metrics = None
        if enable_metrics and HAS_METRICS:
            self.metrics = PerformanceMetrics(
                output_dir=Path("metrics"),
                sampling_interval=100
            )
            logger.info("Performance metrics tracking enabled")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            The ID of the created session
        """
        middleware = ConversationPersistenceMiddleware(self.memory_integration, session_id)
        self.active_sessions[middleware.session_id] = middleware
        return middleware.session_id
    
    async def load_session(self, session_id: str) -> bool:
        """
        Load an existing session.
        
        Args:
            session_id: The ID of the session to load
            
        Returns:
            True if session was loaded successfully, False otherwise
        """
        middleware = ConversationPersistenceMiddleware(self.memory_integration)
        success = await middleware.load_session_state(session_id)
        
        if success:
            self.active_sessions[session_id] = middleware
        
        return success
    
    async def process_interaction(self, session_id: str, user_input: str, response: str) -> bool:
        """
        Process and store a user-Lucidia interaction.
        
        Args:
            session_id: The ID of the session to use
            user_input: The user's message
            response: Lucidia's response
            
        Returns:
            True if processed successfully, False otherwise
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating a new one")
            self.create_session(session_id)
        
        middleware = self.active_sessions[session_id]
        return await middleware.store_interaction(user_input, response)
    
    async def get_context(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        Get context for generating a response to a user input.
        
        Args:
            session_id: The ID of the session to use
            user_input: The user's message
            
        Returns:
            Context dictionary
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating a new one")
            self.create_session(session_id)
        
        middleware = self.active_sessions[session_id]
        return await middleware.retrieve_relevant_context(user_input)
    
    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a session.
        
        Args:
            session_id: The ID of the session to use
            max_turns: Maximum number of turns to return (from most recent)
            
        Returns:
            List of conversation turns
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return []
        
        middleware = self.active_sessions[session_id]
        history = middleware.conversation_history
        
        if max_turns is not None:
            history = history[-max_turns:]
        
        return history
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: The ID of the session to use
            
        Returns:
            Statistics dictionary
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return {"error": "Session not found"}
        
        middleware = self.active_sessions[session_id]
        return await middleware.get_memory_stats()
    
    async def save_all_sessions(self) -> Dict[str, bool]:
        """
        Save all active sessions.
        
        Returns:
            Dictionary mapping session IDs to save success status
        """
        results = {}
        for session_id, middleware in self.active_sessions.items():
            results[session_id] = await middleware.save_session_state()
        return results
    
    def list_sessions(self) -> List[str]:
        """
        List all active session IDs.
        
        Returns:
            List of active session IDs
        """
        return list(self.active_sessions.keys())
        
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics across all sessions.
        
        Returns:
            Dictionary with metrics summary or error message if metrics not enabled
        """
        if not self.metrics or not HAS_METRICS:
            return {"error": "Performance metrics not enabled"}
        
        # Generate summary statistics from metrics data
        try:
            # Save current metrics to ensure we have the latest data
            await self.metrics.save_metrics()
            
            # Calculate aggregated statistics
            sessions = self.list_sessions()
            all_stats = {}
            for session_id in sessions:
                stats = await self.get_session_stats(session_id)
                if "error" not in stats:
                    all_stats[session_id] = stats
            
            # Aggregate system-wide metrics
            total_turns = sum(stats.get("turns", 0) for stats in all_stats.values())
            total_memory = sum(stats.get("memory_size_bytes", 0) for stats in all_stats.values())
            
            return {
                "total_sessions": len(sessions),
                "total_turns": total_turns,
                "total_memory_mb": round(total_memory / (1024 * 1024), 2) if total_memory > 0 else 0,
                "avg_turns_per_session": round(total_turns / max(1, len(sessions)), 2),
                "avg_memory_per_session_kb": round((total_memory / max(1, len(sessions))) / 1024, 2) if total_memory > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}")
            return {"error": str(e)}
