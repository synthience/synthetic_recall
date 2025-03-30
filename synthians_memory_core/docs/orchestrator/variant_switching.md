# Titans Variant Runtime Switching Protocol

## Overview

The Context Cascade Engine (CCE) supports dynamic switching between Titans architecture variants at runtime. This capability is primarily intended for development, experimentation, and testing purposes, allowing developers to compare the behavior of different Titans variants without restarting the system.

## Key Components

### 1. Core Implementation

- **`set_variant()` Method**: Implemented in `ContextCascadeEngine` to handle the safe transition between variants
- **FastAPI Endpoint**: `/set_variant` route exposed through the CCE HTTP API
- **DevMode Protection**: Requires `CCE_DEV_MODE=true` environment variable to enable variant switching
- **Audit Trail**: Complete logging of all variant switches with timestamps and metadata

### 2. Safety Features

- **Processing Lock Check**: Prevents variant switching during active request processing
- **Context Flushing**: Clears the `SequenceContextManager` to prevent cross-variant contamination
- **Processor Reconfiguration**: Rebuilds the attention mechanism and variant processor for the new variant
- **Input Validation**: Validates variant names against the `TitansVariantType` enum

### 3. Neural Memory Considerations

- **State Persistence (Default)**: By default, Neural Memory's internal state (`M` weights, momentum) is preserved when switching variants
- **Optional Reset**: The API supports an optional parameter to reset Neural Memory's state during variant switching

## Usage

### API Endpoint

```http
POST /set_variant
Content-Type: application/json

{
  "variant": "MAC",
  "reset_neural_memory": false
}
```

### Parameters

- **`variant`** (required): The Titans variant to switch to (`"NONE"`, `"MAC"`, `"MAG"`, or `"MAL"`)
- **`reset_neural_memory`** (optional): Whether to reset the Neural Memory state (default: `false`)

### Response

```json
{
  "success": true,
  "variant": "MAC",
  "previous_variant": "NONE",
  "timestamp": "2025-03-30T21:45:00Z",
  "switch_id": "switch_20250330T2145Z",
  "context_flushed": true,
  "context_size_flushed": 12,
  "reconfigured": true,
  "neural_memory_reset": false,
  "error": null,
  "message": "Variant switched successfully with context flush and reconfiguration",
  "status": "switched",
  "dev_mode": true
}
```

## Neural Memory State Handling

### Default Behavior

By default, the Neural Memory state is preserved when switching variants. This allows for studying how different CCE variants affect the same learning process over time.

### When to Reset Neural Memory

Resetting the Neural Memory state (`reset_neural_memory: true`) is recommended when:

1. The previous variant has significantly altered the learning dynamics (e.g., switching from MAL to MAC)
2. You want to start with a clean learning state for comparative analysis
3. You're debugging unexpected behavior that might be related to variant-specific learning patterns

## Audit Trail

All variant switches are logged to:

1. The console logs with detailed information
2. A persistent JSONL file at `logs/variant_switch_log.jsonl`

This audit trail includes:

- Timestamp of the switch
- Previous and new variant types
- Context size that was flushed
- Unique switch ID for tracing
- Reconfiguration status and errors (if any)
- Whether Neural Memory was reset

## Implementation Notes

### Concurrency Considerations

The current implementation is designed for single-worker CCE deployments. In multi-worker scenarios, additional synchronization mechanisms would be needed beyond the current processing lock check.

### Error Handling

If reconfiguration fails during a variant switch:

1. The CCE's `active_variant_type` will be updated
2. The `variant_processor` may remain `None`
3. Subsequent calls to `process_new_input` will effectively run as if the variant is `NONE`
4. The error details are included in the response and the audit log

## Testing Recommendations

When testing variant switching:

1. **Basic Functionality**: Verify all variants can be switched to and from
2. **Concurrent Operation**: Test switching during periods of inactivity
3. **Error Recovery**: Test behavior when reconfiguration or Neural Memory reset fails
4. **State Persistence**: Compare results with and without Neural Memory reset

## Variant Metrics Structure

### Overview

Each Titans variant produces metrics that are included in the response payload. These metrics follow a standardized, nested structure that is crucial for proper integration testing and client interpretation.

### Standard Metrics Format

```json
{
  "variant_output": {
    "variant_type": "MAC",  // The active variant type
    "mac": {                 // Variant-specific metrics in a nested dictionary
      "attended_output_generated": true,
      "fallback_mode": false
      // Other MAC-specific metrics
    }
    // For MAG variant, metrics would be under "mag" key
    // For MAL variant, metrics would be under "mal" key
  }
}
```

### Implementation Details

1. **Metric Isolation**: Each variant's metrics are isolated under their own key (`mac`, `mag`, or `mal`) to prevent namespace collisions.

2. **Top-Level Properties**: Only the `variant_type` is stored at the top level of the `variant_output` object.

3. **Consistent Structure**: All variants follow the same pattern, making client parsing predictable.

### Recent Fixes (March 2025)

The following issues were addressed to ensure consistent metrics structure:

1. **Redundant Metrics**: Fixed an issue where MAC variant was adding the `attended_output_generated` flag both inside the `mac` object and at the top level of `variant_metrics`.

2. **Metrics Propagation**: Corrected the handling of variant metrics in `_process_memory` to prevent direct updates to the top-level `variant_metrics` dictionary.

3. **Standardized Responses**: Ensured that all variant processors produce metrics in the same structured format for consistent API responses.

These changes ensure that integration tests correctly validate the metrics structure and provide a reliable API contract for clients consuming the CCE output.

## Debugging Notes

### Troubleshooting Variant Metrics

If integration tests fail with structure-related issues in the variant metrics, check the following:

1. **Log the Step Context**: Examine the `step_context["variant_metrics"]` structure at different points in the processing pipeline using debug logging:

   ```python
   logger.warning(f"DEBUG: variant_metrics at point X: {step_context['variant_metrics']}")
   ```

2. **Verify Nested Structure**: Ensure all variant-specific metrics are properly nested under their variant key (`mac`, `mag`, or `mal`).

3. **Check for Direct Updates**: Look for code that directly modifies the top-level `variant_metrics` dictionary instead of updating the nested variant dictionary.

4. **Integration Test Expectations**: Verify the assertions in the integration tests to ensure they match the expected structure.

### Common Metrics Issues

1. **Redundant Keys**: Check for metrics being added both at the variant level and at the top level.

2. **Missing Initialization**: Ensure that the variant metrics dictionary is properly initialized with default values for required keys.

3. **Inconsistent Structure**: Verify that all variants follow the same structure pattern, even when errors occur.

4. **Metrics Propagation**: Make sure metrics from post-retrieval processing are correctly merged with pre-update metrics.

### Testing with Docker Compose

When debugging with Docker Compose:

1. Use `docker-compose restart context-cascade-orchestrator` to apply changes without rebuilding.

2. Check container logs with `docker-compose logs -f context-cascade-orchestrator`.

3. For complex issues, run the tests with higher verbosity: `python -m pytest tests/integration/test_variant_switching.py -vv`.

4. Consider using Docker's inspection tools to examine container state: `docker inspect context-cascade-orchestrator`.
