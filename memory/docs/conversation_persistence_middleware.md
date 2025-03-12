# Conversation Persistence Middleware

## Overview

The `ConversationPersistenceMiddleware` is a critical component in Lucidia's memory architecture that ensures seamless multi-turn conversation persistence. It manages session state, conversation history, and memory integration, providing a robust foundation for contextual and coherent AI interactions.

## Features

- **Session Management**: Creates, loads, and maintains conversation sessions
- **Concurrency Handling**: Uses `asyncio.Lock` to prevent race conditions during session operations
- **Asynchronous I/O**: Implements non-blocking file operations with `aiofiles`
- **Memory Size Control**: Limits conversation history to prevent unbounded memory growth
- **Standardized Metadata**: Includes role fields and interaction metadata for better memory organization
- **Context Generation**: Creates rich context from previous interactions for improved responses
- **Error Handling**: Gracefully handles exceptions during storage and retrieval operations

## Architecture

```
┌─────────────────────────────────────┐
│      ConversationPersistenceMiddleware      │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│            MemoryIntegration        │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│              MemoryCore             │
└─────────────────────────────────────┘
```

## Usage

### Basic Integration

```python
from memory.middleware.conversation_persistence import ConversationPersistenceMiddleware
from memory.lucidia_memory_system.memory_integration import MemoryIntegration

# Initialize memory integration
memory_integration = MemoryIntegration()

# Configure middleware
config = {
    'session_dir': './session_data',
    'checkpointing_interval': 5,
    'context_window': 10,
    'interaction_significance': 0.7,
    'role_metadata': True,
    'max_history_size': 100
}

# Create middleware instance
middleware = ConversationPersistenceMiddleware(memory_integration, config)

# Initialize session
await middleware.initialize_session("user_123")

# Store interaction
await middleware.store_interaction("Hello, how are you?", "I'm doing well, thank you!")

# Retrieve context for next turn
context = await middleware.retrieve_relevant_context("Tell me more about yourself")
```

### Using the Decorator

```python
from memory.middleware.conversation_persistence import with_conversation_persistence

@with_conversation_persistence
async def chat_function(user_input, session_id, memory_integration, **kwargs):
    # Process user input
    response = # ... generate response
    return response
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|--------|
| `session_dir` | Directory to store session data | `./session_data` |
| `checkpointing_interval` | Number of interactions before auto-saving session | `5` |
| `context_window` | Number of past interactions to include in context | `10` |
| `interaction_significance` | Default significance value for memories | `0.7` |
| `role_metadata` | Whether to include standardized role fields | `True` |
| `max_history_size` | Maximum number of interactions to keep in memory | `100` |

## Memory Structure

Each interaction generates three types of memory entries:

1. **User Message**: Stores the user's input with metadata
2. **Assistant Response**: Stores Lucidia's response with metadata
3. **Complete Interaction**: Stores the full interaction with context

### Memory Metadata

```json
{
  "session_id": "user_123",
  "sequence_number": 5,
  "turn_id": "user_123_5",
  "timestamp": "2025-03-12T15:30:45.123456",
  "interaction_type": "conversation_turn",
  "role": "user",
  "has_context": true
}
```

## Performance Considerations

- **Concurrency**: The middleware uses locks to prevent race conditions
- **Asynchronous Operations**: File I/O and memory operations are non-blocking
- **Memory Management**: History size limits prevent memory leaks
- **Checkpointing**: Regular state saving ensures persistence without performance impact

## Testing

Tests for the middleware are available in `tests/middleware/test_conversation_persistence.py`. Run with pytest:

```bash
pytest tests/middleware/test_conversation_persistence.py -v
```

## Future Enhancements

- Distributed session management for high-availability deployments
- Compression for large conversation histories
- Selective memory pruning based on significance
- Integration with external persistence layers
