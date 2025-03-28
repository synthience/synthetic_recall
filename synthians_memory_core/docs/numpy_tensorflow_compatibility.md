# NumPy-TensorFlow Compatibility Solution

## Overview

This document describes the solution implemented to resolve NumPy version incompatibility issues in the Lucidia cognitive system, particularly focusing on the TensorFlow integration in the Titans architecture variants.

## Problem Statement

The system experienced a binary incompatibility error related to NumPy versions:

```
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```

This occurred because:

1. The `fix_numpy.py` script downgraded NumPy to version 1.26.4
2. TensorFlow was being imported during module initialization
3. TensorFlow's import chain loaded NumPy before the downgrade could take effect
4. This created conflicts between the original NumPy version and the downgraded version

## Solution: Lazy Loading Pattern

We implemented a lazy loading pattern for TensorFlow that delays its import until actually needed at runtime, allowing the NumPy downgrade to complete first.

### Implementation Details

#### 1. Lazy Loading Mechanism in `titans_variants.py`

```python
# Global variable to hold the TensorFlow module
_tf = None

def _get_tf():
    """Lazy-load TensorFlow only when needed to avoid early NumPy conflicts"""
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf
```

#### 2. Replacing Direct TensorFlow References

Before:
```python
import tensorflow as tf

def process_input(self, attention_output: tf.Tensor) -> Dict[str, Any]:
    # Function implementation
```

After:
```python
def process_input(self, attention_output) -> Dict[str, Any]:
    tf = _get_tf()  # Only imported when function is called
    # Function implementation
```

#### 3. Type Annotation Modifications

Before:
```python
def calculate_gates_from_attention(self, attention_output: tf.Tensor) -> Tuple[float, float, float]:
```

After:
```python
def calculate_gates_from_attention(self, attention_output) -> Tuple[float, float, float]:
```

## Key Files Modified

1. `titans_variants.py` - Implemented lazy loading for TensorFlow and updated all TensorFlow references
2. `context_cascade_engine.py` - Updated imports to avoid direct TensorFlow loading

## Benefits

1. **Proper Initialization Sequence**: Ensures NumPy is downgraded before TensorFlow tries to use it
2. **Reduced Import Coupling**: Components only import TensorFlow when actually needed
3. **Improved Startup Performance**: Modules can be imported without loading the entire TensorFlow stack

## Usage Guidelines

When working with TensorFlow in the Lucidia system:

1. Always use the `_get_tf()` function instead of directly importing TensorFlow
2. Avoid type annotations that directly reference TensorFlow types
3. Use string literals for type annotations when needed: `def func(x: 'tf.Tensor') -> None:`

## Testing

After implementing the lazy loading pattern, all Titans variants (MAC, MAG, MAL) can be initialized and used without triggering NumPy compatibility errors. The system now starts up cleanly and operates as expected.

## Docker Networking Configuration

When testing the Titans architecture variants in a Docker environment, proper service name resolution is critical. The following solution was implemented to ensure communication between the trainer-server and memory-core containers:

1. **Service Discovery Issue**: Direct communication using service names (e.g., `memory-core:5010`) may not work due to Docker networking configuration.

2. **Solution**: Use the special DNS name `host.docker.internal` which allows containers to access services on the host machine:
   ```
   --memcore-url http://host.docker.internal:5010
   ```

3. **Execution Example**: Run Titans variants with the correct memory core URL:
   ```bash
   docker exec -e TITANS_VARIANT=MAC trainer-server python -m synthians_memory_core.tools.lucidia_think_trace --query "This is a test" --memcore-url "http://host.docker.internal:5010"
   ```

4. **Results**: All three Titans variants (MAC, MAG, MAL) successfully connect to the Memory Core service and complete processing with proper neural memory integration.
