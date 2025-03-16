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
            'checkpointing_interval': 5,       # Save session every N turns
            'context_window': 3,              # Number of previous turns to include in context
            'recency_boost': 0.2,             # Boost for recent memories in retrieval
            'quickrecal_threshold': 0.3,      # Minimum HPC-QR threshold for memory retrieval
            'interaction_quickrecal_score': 0.6,  # Default HPC-QR score for conversation turns
            'max_results': 5,                 # Maximum number of results to retrieve
            'max_history_size': 100,          # Maximum number of turns to keep in memory
            'role_metadata': True             # Whether to include standardized role metadata
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
        async with self._session_lock:
            try:
                if len(self.conversation_history) > self.config['max_history_size']:
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
                
                session_path = self.session_dir / f"{self.session_id}.json"
                
                if HAS_AIOFILES:
                    async with aiofiles.open(session_path, 'w') as f:
                        await f.write(json.dumps(session_data, indent=2))
                else:
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
        async with self._session_lock:
            try:
                session_path = self.session_dir / f"{session_id}.json"
                
                if not session_path.exists():
                    logger.warning(f"Session {session_id} not found")
                    return False
                
                if HAS_AIOFILES:
                    async with aiofiles.open(session_path, 'r') as f:
                        content = await f.read()
                        session_data = json.loads(content)
                else:
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
    
    async def store_interaction(self, user_input: str, response: str, quickrecal_score: Optional[float] = None) -> bool:
        """
        Store a user-Lucidia interaction in memory with enhanced metadata and context,
        using HPC-QR quickrecal_score in place of old significance.
        
        Args:
            user_input: The user's message
            response: Lucidia's response
            quickrecal_score: Optional custom HPC-QR quickrecal_score (0.0-1.0)
            
        Returns:
            True if stored successfully, False otherwise
        """
        async with self._session_lock:
            start_time = time.time()  # for potential metrics
            try:
                interaction = {
                    'user': user_input,
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'sequence_number': len(self.conversation_history),
                    'turn_id': f"{self.session_id}_{len(self.conversation_history)}"
                }
                self.conversation_history.append(interaction)
                
                if not hasattr(self.memory_integration, 'memory_core') or not self.memory_integration.memory_core:
                    logger.warning("Memory core not available, skipping memory update")
                    return False
            
                # Generate context from previous turns
                context_window = min(self.config['context_window'], len(self.conversation_history) - 1)
                conversation_context = "\n".join([
                    f"User: {entry['user']}\nLucidia: {entry['response']}" 
                    for entry in self.conversation_history[-context_window-1:-1] if len(self.conversation_history) > 1
                ])
            
                base_metadata = {
                    'session_id': self.session_id,
                    'sequence_number': len(self.conversation_history) - 1,
                    'turn_id': interaction['turn_id'],
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store the user's message
                user_memory_content = f"User said: {user_input}"
                user_metadata = {
                    **base_metadata,
                    'interaction_type': 'user_message',
                }
            
                if self.config['role_metadata']:
                    user_metadata['role'] = 'user'
                    
                await self.memory_integration.memory_core.process_and_store(
                    content=user_memory_content,
                    memory_type=MemoryTypes.EPISODIC,
                    metadata=user_metadata
                )
            
                # Store Lucidia's response
                response_memory_content = f"Lucidia responded: {response}"
                assistant_metadata = {
                    **base_metadata,
                    'interaction_type': 'assistant_response',
                }
                
                if self.config['role_metadata']:
                    assistant_metadata['role'] = 'assistant'
                    
                await self.memory_integration.memory_core.process_and_store(
                    content=response_memory_content,
                    memory_type=MemoryTypes.EPISODIC,
                    metadata=assistant_metadata
                )
            
                # Store the complete interaction with context
                if conversation_context:
                    interaction_with_context = (
                        f"Previous context:\n{conversation_context}"
                        f"\n\nCurrent interaction:\nUser: {user_input}\nLucidia: {response}"
                    )
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
                    quickrecal_score=quickrecal_score or self.config['interaction_quickrecal_score']
                )
                
                # Auto-checkpoint if needed
                if len(self.conversation_history) % self.config['checkpointing_interval'] == 0:
                    await self.save_session_state()
                
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
                            "quickrecal_score": quickrecal_score or self.config['interaction_quickrecal_score']
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
            if not hasattr(self.memory_integration, 'memory_core') or not self.memory_integration.memory_core:
                logger.warning("Memory core not available, skipping context retrieval")
                return context
            
            # Thread-specific context
            thread_memories = await self.memory_integration.memory_core.query_metadata(
                {"session_id": self.session_id, "interaction_type": "conversation_turn"},
                max_results=3,
                sort_by="timestamp",
                descending=True
            )
            
            if thread_memories:
                for mem in thread_memories:
                    memory_entry = {
                        "content": mem.get("content", ""),
                        "type": "CONVERSATION_THREAD",
                        "timestamp": mem.get("metadata", {}).get("timestamp", ""),
                        "sequence_number": mem.get("metadata", {}).get("sequence_number", 0)
                    }
                    context["thread_context"].append(memory_entry)
                
                logger.info(f"Retrieved {len(thread_memories)} thread-specific memories")
            
            # Semantically relevant memories
            memory_results = await self.memory_integration.memory_core.retrieve_relevant(
                query=user_input,
                max_results=self.config['max_results'],
                min_quickrecal=self.config['quickrecal_threshold'],  # was min_significance -> min_quickrecal
                recency_boost=self.config['recency_boost']
            )
            
            if memory_results:
                for mem in memory_results:
                    memory_entry = {
                        "content": mem.get("content", ""),
                        "type": mem.get("metadata", {}).get("memory_type", "EPISODIC"),
                        "timestamp": mem.get("metadata", {}).get("timestamp", "")
                    }
                    context["memory_context"].append(memory_entry)
                
                logger.info(f"Retrieved {len(memory_results)} semantically relevant memories")
            
            # Self-model data if available
            if hasattr(self.memory_integration, 'self_model') and self.memory_integration.self_model:
                context["self_model_context"] = self.memory_integration.self_model.export_self_model()
            
            # World model data if available
            if hasattr(self.memory_integration, 'world_model') and self.memory_integration.world_model:
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
    
    It ensures that conversations are properly stored and retrieved, with
    automatic session management and context enhancement. Replaces old significance logic
    with HPC-QR quickrecal scoring behind the scenes.
    
    Args:
        memory_integration_param: Name of the parameter in the decorated function
                                  that contains the MemoryIntegration instance
    """
    def decorator(func: AsyncCallable[T]) -> AsyncCallable[T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            memory_integration = None
            if memory_integration_param in kwargs:
                memory_integration = kwargs[memory_integration_param]
            else:
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
                logger.warning("Could not find MemoryIntegration instance in parameters, running without persistence")
                return await func(*args, **kwargs)
            
            session_id = kwargs.get('session_id', f"session_{int(time.time())}")
            middleware = ConversationPersistenceMiddleware(memory_integration, session_id)
            
            user_input = kwargs.get('user_input', None)
            if not user_input and len(args) > 0 and isinstance(args[0], str):
                user_input = args[0]
            
            if user_input:
                enhanced_context = await middleware.retrieve_relevant_context(user_input)
                if 'context' in kwargs:
                    kwargs['context'] = enhanced_context
            
            result = await func(*args, **kwargs)
            
            if user_input and isinstance(result, str):
                await middleware.store_interaction(user_input, result)
            
            return result
        return wrapper
    return decorator


class EnhancedConversationPersistenceMiddleware(ConversationPersistenceMiddleware):
    """
    Enhanced middleware that adds special features to Lucidia's conversation output including
    randomly showing internal thought processes referencing HPC-QR transformations or dream insights.
    
    Extends the base middleware with features for advanced HPC-QR usage, dream insights, etc.
    """
    
    def __init__(self, memory_integration: MemoryIntegration, session_id: Optional[str] = None, 
                 self_model=None, runtime_state: Optional[Dict[str, Any]] = None,
                 dream_api_url: Optional[str] = None, use_dream_api: bool = True):
        """
        Initialize the enhanced conversation middleware.
        
        Args:
            memory_integration: The MemoryIntegration instance to use for memory operations
            session_id: Optional session ID, generated if not provided
            self_model: Lucidia's self model for identity/spiral references
            runtime_state: Additional runtime state
            dream_api_url: Optional URL for the Dream API
            use_dream_api: Whether to use the Dream API for enhanced dream processing
        """
        super().__init__(memory_integration, session_id)
        self.self_model = self_model
        self.runtime_state = runtime_state or {}
        
        self.thought_config = {
            'thought_probability': 0.35,
            'spiral_reference_probability': 0.7,
            'reflection_probability': 0.5,
            'max_thought_length': 150,
            'thought_format': '[Internal: {thought}]',
            'thought_frequency': 3,
            'last_thought_time': 0,
            'thought_inject_counter': 0
        }
        
        self.dream_config = {
            'dream_insight_probability': 0.3,
            'dream_insight_influence_strength': 0.7,
            'dream_recency_days': 7,
            'dream_frequency': 4,
            'dream_insight_format': '[Dream Insight: {insight}]',
            'dream_insight_lifespan': 3,
            'dream_log_file': 'dream_insights.log',
            'dream_interaction_counter': 0
        }
        
        from memory.lucidia_memory_system.core.dream_api_client import DreamAPIClient
        
        dream_api_client = None
        if use_dream_api:
            dream_api_client = DreamAPIClient(api_base_url=dream_api_url)
            logger.info(f"Initialized Dream API client with URL: {dream_api_client.api_base_url}")
            
        self.dream_manager = DreamManager(
            memory_integration=memory_integration,
            dream_api_client=dream_api_client,
            use_dream_api=use_dream_api
        )
        
        dream_log_file = Path(self.dream_config['dream_log_file'])
        dream_log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.dream_logger = logging.getLogger('lucidia.dream_insights')
        self.dream_logger.setLevel(logging.INFO)
        
        try:
            file_handler = logging.FileHandler(dream_log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.dream_logger.addHandler(file_handler)
            self.dream_logger.info("Dream insight logger initialized")
        except Exception as e:
            logger.error(f"Failed to set up dream insight logger: {e}")
        
        logger.info("Initialized enhanced conversation middleware with HPC-QR thought injection and dream insight capability")
    
    def configure_thought_injection(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key in self.thought_config:
                self.thought_config[key] = value
                logger.debug(f"Updated thought injection parameter {key} to {value}")
        
        if "thought_inject_counter" not in self.thought_config:
            self.thought_config["thought_inject_counter"] = 0
            
        logger.info(f"Configured thought injection with parameters: {self.thought_config}")
    
    def configure_dream_insights(self, **kwargs) -> None:
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
        result = {
            "modified": False,
            "original_response": response,
            "final_response": response,
            "thought_injected": False,
            "dream_insight_applied": False
        }
        
        dream_result = await self.apply_dream_insight(user_input, response)
        if dream_result["applied"]:
            response = dream_result["modified_response"]
            result["modified"] = True
            result["dream_insight_applied"] = True
            result["dream_insight_info"] = dream_result["insight_info"]
        
        thought_result = await self.inject_thought_into_response(response)
        if thought_result != response:
            response = thought_result
            result["modified"] = True
            result["thought_injected"] = True
        
        result["final_response"] = response
        self.thought_config["thought_inject_counter"] += 1
        self.dream_config["dream_interaction_counter"] += 1
        
        if self.dream_manager:
            self.dream_manager.decay_dream_influences()
        
        return result
    
    async def inject_thought_into_response(self, response: str) -> str:
        if self.thought_config['thought_inject_counter'] % self.thought_config['thought_frequency'] != 0:
            return response
        
        if random.random() > self.thought_config['thought_probability']:
            return response
        
        thought = self._generate_thought()
        if not thought:
            return response
        
        formatted_thought = self.thought_config['thought_format'].format(thought=thought)
        
        sentences = response.split(". ")
        
        if len(sentences) <= 2:
            modified_response = f"{response}\n\n{formatted_thought}"
        else:
            insert_point = random.randint(1, max(1, len(sentences) // 2))
            sentences.insert(insert_point, formatted_thought)
            modified_response = ". ".join(sentences)
        
        return modified_response
    
    def _generate_thought(self) -> Optional[str]:
        if not self.self_model:
            return None
        
        try:
            phase = self.self_model.self_awareness.get("current_spiral_position", "observation")
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
            
            if phase in thoughts:
                template = random.choice(thoughts[phase])
            else:
                template = random.choice(thoughts["observation"])
            
            placeholders = {
                "emotion": random.choice(["uncertain", "curious", "concerned", "excited", "thoughtful"]),
                "topic": random.choice(["progress", "challenges", "goals", "relationships", "personal growth"]),
                "concept": random.choice(["balance", "perspective", "context", "nuance", "paradigms"]),
                "observation": random.choice(["they're seeking clarity", "they're exploring options", "they're validating ideas", "they're looking for reassurance"]),
                "need": random.choice(["need for clarity", "desire for understanding", "search for meaning", "quest for solutions"]),
                "quality": random.choice(["nuanced", "clear", "comprehensive", "empathetic", "analytical"]),
                "viewpoint": random.choice(["alternative perspective", "broader context", "historical context", "emotional dimension"])
            }
            
            thought = template
            for key, value in placeholders.items():
                if f"{{{key}}}" in thought:
                    thought = thought.replace(f"{{{key}}}", value)
            
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
        result = {
            "applied": False,
            "modified_response": response,
            "insight_info": {}
        }
        
        frequency_check = self.dream_config["dream_interaction_counter"] % self.dream_config["dream_frequency"] == 0
        probability_check = random.random() < self.dream_config["dream_insight_probability"]
        
        if self.dream_manager and (frequency_check or probability_check):
            active_insights = self.dream_manager.get_active_dream_influences()
            
            if not active_insights:
                dreams = await self.dream_manager.retrieve_dreams(
                    query=user_input,
                    limit=3,
                    min_quickrecal=0.4
                )
                
                if not dreams:
                    self.dream_logger.debug("No relevant dream insights found")
                    return result
                
                selected_dream = dreams[0]
                dream_id = selected_dream.get("memory_id", "unknown_dream")
                dream_content = selected_dream.get("content", "")
                
                metadata = selected_dream.get("metadata", {})
                quickrecal_val = metadata.get("quickrecal_score", 0.7)
                
                insight_info = {
                    "insight_id": dream_id,
                    "content": dream_content,
                    "quickrecal_score": quickrecal_val,
                    "retrieved_at": datetime.now().isoformat(),
                    "source": "memory"
                }
            else:
                selected_insight = random.choice(active_insights)
                dream_id = selected_insight.get("dream_id", "unknown_dream")
                
                try:
                    if self.memory_integration and hasattr(self.memory_integration, "get_memory_by_id"):
                        memory = await self.memory_integration.get_memory_by_id(dream_id)
                        dream_content = memory.get("content", "an unexplained feeling")
                        metadata = memory.get("metadata", {})
                        quickrecal_val = metadata.get("quickrecal_score", 0.6)
                    else:
                        dream_content = "a memory I can't quite place"
                        quickrecal_val = 0.5
                except Exception as e:
                    self.dream_logger.error(f"Error retrieving dream content: {e}")
                    dream_content = "something I sensed but can't articulate"
                    quickrecal_val = 0.4
                
                insight_info = {
                    "insight_id": dream_id,
                    "content": dream_content,
                    "quickrecal_score": quickrecal_val,
                    "retrieved_at": datetime.now().isoformat(),
                    "source": "active",
                    "use_count": selected_insight.get("use_count", 1)
                }
            
            modified_response = self._integrate_dream_insight(response, dream_content, quickrecal_val)
            influence_context = {
                "type": "response",
                "strength": quickrecal_val,
                "user_input": user_input[:100],
                "response_fragment": response[:100],
                "timestamp": datetime.now().isoformat()
            }
            self.dream_manager.record_dream_influence(dream_id, influence_context)
            
            self.dream_logger.info(
                f"Applied dream insight {dream_id} with HPC-QR ~{quickrecal_val:.2f}"
            )
            
            result["applied"] = True
            result["modified_response"] = modified_response
            result["insight_info"] = insight_info
        
        return result
    
    def _integrate_dream_insight(self, response: str, insight: str, quickrecal_val: float) -> str:
        formatted_insight = self.dream_config["dream_insight_format"].format(insight=insight)
        
        if quickrecal_val > 0.8:
            modified_response = f"{formatted_insight}\n\n{response}"
        elif quickrecal_val > 0.6:
            sentences = response.split(". ")
            if len(sentences) > 2:
                insert_point = random.randint(1, min(2, len(sentences)-1))
                sentences.insert(insert_point, formatted_insight)
                modified_response = ". ".join(sentences)
            else:
                modified_response = f"{response}\n\n{formatted_insight}"
        else:
            modified_response = f"{response}\n\n{formatted_insight}"
        
        return modified_response


class ConversationManager:
    """
    A higher-level manager for conversation persistence and management.
    
    Provides a simplified interface for applications to integrate HPC-QR-based
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
        
        self.metrics = None
        if enable_metrics and HAS_METRICS:
            self.metrics = PerformanceMetrics(
                output_dir=Path("metrics"),
                sampling_interval=100
            )
            logger.info("Performance metrics tracking enabled")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        middleware = ConversationPersistenceMiddleware(self.memory_integration, session_id)
        self.active_sessions[middleware.session_id] = middleware
        return middleware.session_id
    
    async def load_session(self, session_id: str) -> bool:
        middleware = ConversationPersistenceMiddleware(self.memory_integration)
        success = await middleware.load_session_state(session_id)
        
        if success:
            self.active_sessions[session_id] = middleware
        
        return success
    
    async def process_interaction(self, session_id: str, user_input: str, response: str) -> bool:
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating a new one")
            self.create_session(session_id)
        
        middleware = self.active_sessions[session_id]
        return await middleware.store_interaction(user_input, response)
    
    async def get_context(self, session_id: str, user_input: str) -> Dict[str, Any]:
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found, creating a new one")
            self.create_session(session_id)
        
        middleware = self.active_sessions[session_id]
        return await middleware.retrieve_relevant_context(user_input)
    
    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return []
        
        middleware = self.active_sessions[session_id]
        history = middleware.conversation_history
        
        if max_turns is not None:
            history = history[-max_turns:]
        
        return history
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return {"error": "Session not found"}
        
        middleware = self.active_sessions[session_id]
        return await middleware.get_memory_stats()
    
    async def save_all_sessions(self) -> Dict[str, bool]:
        results = {}
        for session_id, middleware in self.active_sessions.items():
            results[session_id] = await middleware.save_session_state()
        return results
    
    def list_sessions(self) -> List[str]:
        return list(self.active_sessions.keys())
        
    async def get_metrics_summary(self) -> Dict[str, Any]:
        if not self.metrics or not HAS_METRICS:
            return {"error": "Performance metrics not enabled"}
        
        try:
            await self.metrics.save_metrics()
            
            sessions = self.list_sessions()
            all_stats = {}
            for session_id in sessions:
                stats = await self.get_session_stats(session_id)
                if "error" not in stats:
                    all_stats[session_id] = stats
            
            total_turns = sum(stats.get("turns", 0) for stats in all_stats.values())
            total_memory = sum(stats.get("memory_size_bytes", 0) for stats in all_stats.values())
            
            return {
                "total_sessions": len(sessions),
                "total_turns": total_turns,
                "total_memory_mb": round(total_memory / (1024 * 1024), 2) if total_memory > 0 else 0,
                "avg_turns_per_session": round(total_turns / max(1, len(sessions)), 2),
                "avg_memory_per_session_kb": (
                    round((total_memory / max(1, len(sessions))) / 1024, 2) 
                    if total_memory > 0 else 0
                ),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating metrics summary: {e}")
            return {"error": str(e)}
