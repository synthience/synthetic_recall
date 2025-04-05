# Neural Memory Server Documentation

This directory provides documentation for the `synthians_trainer_server` package, the adaptive associative memory component.

## Contents

* [Neural Memory Module](./neural_memory.md): Describes the core `NeuralMemoryModule` (TensorFlow).
* [Metrics & Diagnostics](./metrics_store.md): Explains `MetricsStore` and diagnostic endpoints like `/diagnose_emoloop`.
* [Surprise Detection](./surprise_detector.md): Details on surprise calculation.

## Phase 5.9 Interaction

In Phase 5.9, the Neural Memory Server maintains its core functionality without major internal changes, but participates in the enhanced observability ecosystem through:

1. **Performance Metrics**: Provides performance metrics (loss, grad_norm) via the `/update_memory` endpoint, which are used by the Context Cascade Engine for variant selection and by the Memory Core for QuickRecal boosting.

2. **Diagnostic Data**: Exposes diagnostic information via the `/diagnose_emoloop` endpoint, which can be consumed by the dashboard to visualize emotional loop performance.

3. **Configuration Exposure**: Runtime configuration potentially exposed via `/config/runtime/neural-memory` (proxied by the Memory Core API) to provide visibility into Neural Memory settings.

The diagnostics data provided by the `/diagnose_emoloop` endpoint includes:
- Average loss and gradient norm over time
- Gate value distribution
- Learning statistics (updates processed, timings)
- Memory update metrics

This data enables dashboard users to understand the adaptive behavior of the Neural Memory system and how it influences the overall cognitive process.

Refer to the main [Architecture](../ARCHITECTURE.md) and [Component Guide](../COMPONENT_GUIDE.md) for system context.
