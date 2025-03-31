# Performance-Aware Variant Selection (Phase 5.5)

This document describes the implementation of performance-aware variant selection in the Context Cascade Engine (CCE). This feature allows the system to dynamically switch between different Titans variants based on Neural Memory (NM) performance metrics.

## Overview

The performance-aware variant selection system consists of three main components:

1. **Performance Tracking**: The CCE maintains a history of Neural Memory performance metrics (loss and gradient norm) after each update.
2. **Trend Analysis**: The system analyzes recent performance history to detect trends (increasing or decreasing surprise).
3. **Dynamic Selection**: The `VariantSelector` uses performance metrics and trends to select the optimal variant for the current context.

## Performance Tracking

The CCE tracks Neural Memory performance metrics in a rolling history buffer:

```python
# In ContextCascadeEngine.__init__
self.nm_performance_history = deque(maxlen=20)  # Keep the last 20 update metrics
```

After each successful Neural Memory update, the performance metrics are added to this history:

```python
# After successful NM update
self.nm_performance_history.append({
    "loss": update_resp.get("loss"),
    "grad_norm": update_resp.get("grad_norm"),
    "timestamp": time.time(),
    "variant": self.active_variant_type.value
})
```

## Trend Analysis

Before calling the `VariantSelector`, the CCE calculates average performance metrics and detects trends using linear regression:

```python
# Calculate rolling average and detect trends
avg_loss = sum(p["loss"] for p in perf_history if p.get("loss")) / count
avg_grad_norm = sum(p["grad_norm"] for p in perf_history if p.get("grad_norm")) / count

# Trend analysis using numpy.polyfit
loss_trend = float(np.polyfit(x_norm, y_loss, 1)[0])
grad_trend = float(np.polyfit(x_norm, y_grad, 1)[0])
combined_trend = loss_trend + (grad_trend / 10.0)

# Set trend flags based on slope magnitude
trend_threshold = 0.05  # Minimum slope to consider a genuine trend
nm_performance["trend_increasing"] = combined_trend > trend_threshold
nm_performance["trend_decreasing"] = combined_trend < -trend_threshold
nm_performance["trend_slope"] = combined_trend
```

## Variant Selection Rules

The `VariantSelector` uses the following rules to select the optimal variant:

1. **Priority Order**:
   - LLM guidance (highest priority)
   - Metadata task type
   - Performance metrics
   - Query keywords
   - Default fallback

2. **Performance Rules**:
   - **Increasing Trend**: If surprise is increasing significantly, select **MAG** to adapt learning parameters.
   - **High Surprise**: If average surprise exceeds `high_surprise_threshold`, select **MAG**.
   - **Decreasing Trend** (moderate surprise): If surprise is decreasing in the moderate range, select **MAL** for refinement.
   - **Low Surprise**: If average surprise is below `low_surprise_threshold`, select **NONE** for efficiency.
   - **Default**: For moderate surprise with no clear trend, select **MAC**.

## Thresholds and Configuration

The performance-aware selection uses several configurable thresholds:

```python
# In ContextCascadeEngine.__init__
self.variant_selector = VariantSelector(
    high_surprise_threshold=high_surprise_threshold,  # Default: 0.5
    low_surprise_threshold=low_surprise_threshold     # Default: 0.1
)
```

Other internal thresholds include:
- **Sample Count Threshold**: At least 3 samples required for trend analysis
- **Trend Threshold**: Minimum slope magnitude (0.05) to consider a genuine trend
- **Normalization Factor**: For scaling gradient norms (10.0)

## Integration with CCE

The performance-aware selection is fully integrated into the CCE's processing flow:

```python
# In process_new_input method
# 1. Calculate average NM performance metrics
nm_performance = {
    "avg_loss": avg_loss,
    "avg_grad_norm": avg_grad_norm,
    "sample_count": count,
    "trend_increasing": combined_trend > trend_threshold,
    "trend_decreasing": combined_trend < -trend_threshold,
    "trend_slope": combined_trend
}

# 2. Select optimal variant using VariantSelector
selected_variant, reason, decision_trace = self.variant_selector.select_variant(
    query=content,
    metadata=step_context["metadata"],
    nm_performance=nm_performance,
    llm_variant_hint=llm_advice.get("variant_hint")
)

# 3. Switch variant if needed
if selected_variant != self.active_variant_type:
    logger.info(f"Switching variant from {self.active_variant_type.value} to {selected_variant.value} ({reason})")
    switch_success = await self._switch_variant_internal(selected_variant, reason)
```

## Testing

The performance-aware selection is tested at multiple levels:

1. **Unit Tests** (`test_performance_aware_selection.py`): Tests individual selection rules in isolation.
2. **Integration Tests** (`test_cce_performance_selection.py`): Tests end-to-end performance-based selection with mocked Neural Memory responses.

## Future Enhancements

Potential future enhancements include:

1. **Adaptive Thresholds**: Automatically adjust thresholds based on observed performance patterns.
2. **Continuous Learning**: Use reinforcement learning to optimize variant selection based on outcome metrics.
3. **Hysteresis**: Prevent rapid switching between variants by requiring sustained threshold crossing.
4. **Task-Specific Performance Profiles**: Develop specialized performance profiles for different task types.
