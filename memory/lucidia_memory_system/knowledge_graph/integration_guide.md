# Integration Guide: Lucidia Modular Knowledge Graph

This guide provides step-by-step instructions for integrating the modular knowledge graph architecture into the existing Lucidia system.

## Overview

This integration process will transition the knowledge graph from its current monolithic implementation to a more maintainable, modular architecture that addresses embedding dimension mismatches and provides specialized functionality through distinct modules.

## Steps for Integration

### 1. Directory Preparation

The modular implementation is currently in the `knowlege-graph` directory, but the hyphen causes issues with Python imports. First, rename this directory to use an underscore:

```bash
mv memory/lucidia_memory_system/knowlege-graph memory/lucidia_memory_system/knowledge_graph
```

### 2. Import Path Updates

Update all import statements to use the new directory name throughout the codebase. This includes:

- All modular implementation files
- Integration example script
- Migration utility

### 3. Module System Integration

The modular architecture uses an event-driven system with modules registering and communicating through an EventBus. Integration requires:

1. Initialize the modular knowledge graph with appropriate configuration
2. Configure embedding dimensions to match your current settings (default 768)
3. Register the knowledge graph with your application's dependency system

Example initialization:

```python
from memory.lucidia_memory_system.knowledge_graph.core import LucidiaKnowledgeGraph

# Configuration with embedding dimension settings
config = {
    "embedding": {
        "enable_hyperbolic": True,
        "hyperbolic_curvature": 1.0,
        "embedding_dimension": 768  # Match your system's embedding dimension
    },
    "visualization": {
        "max_nodes": 200
    },
    # Additional module configs here
}

# Initialize the knowledge graph
knowledge_graph = LucidiaKnowledgeGraph(config=config)
knowledge_graph.initialize()
```

### 4. Data Migration

Use the migration utility to transfer existing knowledge graph data to the new modular structure:

```python
from memory.lucidia_memory_system.knowledge_graph.migration_utility import KnowledgeGraphMigrator

# Initialize with the legacy graph instance and configuration
migrator = KnowledgeGraphMigrator(legacy_graph=existing_graph, config=config)

# Run the migration process
stats = await migrator.run_migration()

# Get the new modular graph instance
modular_graph = migrator.modular_graph
```

This utility will handle embedding dimension alignment, ensuring that 384-dimension and 768-dimension vectors can coexist without causing errors.

### 5. API Adaptation

The modular architecture maintains compatibility with the existing API but uses delegation to specialized modules. Update your code to use the new module delegation pattern:

Instead of:
```python
# Legacy approach
result = knowledge_graph.search_by_content(query, threshold=0.7)
```

With the modular architecture:
```python
# Modular approach - internally delegates to specialized module
result = await knowledge_graph.search_nodes(query, threshold=0.7)
```

### 6. Embedding Dimension Compatibility

The modular system includes enhanced embedding handling to address dimension mismatches:

- `_align_vectors_for_comparison`: Aligns vectors of different dimensions for comparison
- `_normalize_embedding`: Handles dimension mismatches with padding or truncation
- `_validate_embedding`: Ensures embeddings contain no NaN or Inf values

When providing embeddings to the system, these will be handled automatically. For manual access to these utilities:

```python
# Get the embedding manager module
embedding_manager = knowledge_graph.module_registry.get_module("embedding_manager")

# Use the alignment utilities
aligned_vec1, aligned_vec2 = embedding_manager._align_vectors_for_comparison(embedding1, embedding2)
```

### 7. Testing the Integration

Use the integration example as a template for testing the modular architecture:

```bash
python -m memory.lucidia_memory_system.knowledge_graph.integration_example
```

This will run a comprehensive demonstration of the modular architecture's capabilities.

### 8. Transitioning Existing Code

Gradually transition your existing code to use the modular architecture:

1. Start with non-critical features or new features
2. Use the core LucidiaKnowledgeGraph API for common operations
3. Access specialized modules when needed for advanced functionality

```python
# Get a specialized module for advanced operations
contradiction_manager = knowledge_graph.module_registry.get_module("contradiction_manager")
result = await contradiction_manager.detect_contradictions(node_id)
```

## Troubleshooting

### Embedding Dimension Errors

If you encounter embedding dimension mismatch errors:

1. Verify your configuration has the correct `embedding_dimension` setting
2. Use the `_align_vectors_for_comparison` utility when manually comparing vectors
3. Check if any embeddings contain NaN or Inf values using `_validate_embedding`

### Module Import Errors

If you encounter module import errors:

1. Verify the directory has been renamed from `knowlege-graph` to `knowledge_graph`
2. Ensure `__init__.py` file is present with proper imports
3. Check that import paths use `knowledge_graph` (with underscore, not hyphen)

## Support and Further Development

The modular architecture supports independent evolution of modules. To extend the system:

1. Create a new module class inheriting from `KnowledgeGraphModule`
2. Register it with the module registry
3. Subscribe to relevant events via the event bus

Example module registration:

```python
from memory.lucidia_memory_system.knowledge_graph.base_module import KnowledgeGraphModule

# Create custom module
class CustomModule(KnowledgeGraphModule):
    # Implement module functionality
    pass

# Register with the system
custom_module = CustomModule(event_bus, module_registry)
module_registry.register_module("custom_module", custom_module)
```
