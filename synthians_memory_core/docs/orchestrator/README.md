# Context Cascade Engine Documentation

This directory contains documentation for the Context Cascade Engine (CCE) and its components that orchestrate the cognitive cycle.

## Contents

* [Context Cascade Engine](./cce.md): Overview of the `ContextCascadeEngine`.
* [Titans Variants](./titans_variants.md): Documentation on MAC, MAG, MAL variants.
* [Attention Mechanisms](./attention.md): Details on attention calculations.
* [Sequence Context Management](./sequence_context.md): Documentation on `SequenceContextManager`.
* [Performance-Aware Selection](./performance_aware_selection.md): How variants are dynamically selected.

## Technical Details

* **Variant Flow & Switching**: Processing paths and dynamic selection.
* **TensorFlow Integration**: Lazy loading mechanism.
* **Surprise Feedback Loop**: QuickRecal boost mechanism.
* **Performance Tracking & LLM Guidance**: Integration details.
* **History Management**: Context usage for attention.
* **Phase 5.9 Interaction**: Provides enhanced metrics via `/metrics/recent_cce_responses` for dashboard consumption.

## Phase 5.9 Enhancements

The Context Cascade Engine has been enhanced in Phase 5.9 to provide more detailed metadata about its decision-making process:

1. **Enhanced Metrics Response**: The `/metrics/recent_cce_responses` endpoint now includes detailed information about:
   * Variant selection reasoning with trace information
   * Performance metrics that influenced the decision
   * LLM guidance details including confidence levels and adjustments
   * Attention focus mechanisms used
   
2. **Configuration Exposure**: Runtime configuration can potentially be exposed via the Memory Core API proxy at `/config/runtime/cce`.

3. **Dashboard Integration**: These enhancements support the visualization of CCE behavior in the upcoming Synthians Cognitive Dashboard.

The enhanced metrics response structure now includes:

```json
{
  "timestamp": "...",
  "status": "completed",
  "memory_id": "mem_abc",
  "variant_output": { /* ... variant specific metrics ... */ },
  "variant_selection": {
    "selected": "MAG",
    "reason": "Performance (High Surprise 0.65 -> MAG)",
    "trace": ["Input metrics: ...", ...],
    "perf_metrics_used": {"avg_loss": 0.65, ...}
  },
  "llm_advice_used": {
    "raw_advice": { /* Optional raw */ },
    "adjusted_advice": { /* Advice after confidence adjustment */ },
    "confidence_level": 0.95,
    "adjustment_reason": "High confidence...",
    "boost_modifier_applied": 0.1,
    "tags_added": ["quantum"],
    "variant_hint_followed": true,
    "attention_focus_used": "relevance"
  },
  "neural_memory_update": { /* ... loss, grad_norm ... */ },
  "quickrecal_feedback": { /* ... boost applied ... */ }
}
```

This enhanced data enables deeper understanding of why the CCE makes specific decisions, how LLM guidance influences processing, and what factors contribute to variant selection.

Refer to the main [Architecture](../ARCHITECTURE.md) and [Component Guide](../COMPONENT_GUIDE.md) for system context.
