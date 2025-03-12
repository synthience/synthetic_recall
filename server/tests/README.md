# Model Context Tools Tests

This directory contains tests for validating the functionality of the Model Context Protocol (MCP) tools in the Lucidia system.

## Overview

The `test_model_context_tools.py` script validates that the `ModelContextToolProvider` is properly integrated and functioning. It tests all key capabilities:

1. **Self-Model Tools**: Update Lucidia's self-model with new reflections and characteristics
2. **Knowledge Graph Tools**: Update Lucidia's knowledge with new concepts and relationships
3. **Memory Operation Tools**: Store and retrieve memories from the memory system
4. **Dream Management Tools**: Control when and how Lucidia dreams
5. **Spiral Phase Tools**: Manage cognitive development and reflection phases
6. **Parameter Management Tools**: Control system parameters to optimize performance
7. **System Health Check**: Validate comprehensive system health and generate recommendations

## Running the Tests

To run the tests, navigate to the server directory and execute:

```bash
python -m tests.test_model_context_tools
```

## Test Architecture

The tests use mock implementations of the system components to allow testing without dependencies on the full system. This makes it possible to verify the interface and functionality of the MCP tools in isolation.

## Integration with Lucidia

The ModelContextToolProvider has been integrated into the main server architecture in the `initialize_components` function of `dream_api_server.py`. It's initialized with all necessary components (self_model, world_model, knowledge_graph, etc.) and registered alongside the other existing tool providers.

During conversations, Lucidia can now call these MCP tools to:
1. Update its self-model based on interactions and learning
2. Manage dreams and reflections autonomously
3. Perform system health checks and implement recommendations
4. Adjust parameters to optimize performance

## Next Steps

After validating the tests pass, monitor the system logs during live conversations to verify that Lucidia is autonomously calling these tools when appropriate.
