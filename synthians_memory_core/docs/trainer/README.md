# Neural Memory Server Documentation

This directory provides documentation for the `synthians_trainer_server` package, which implements the adaptive associative memory component of the Synthians architecture.

## Contents

*   [Neural Memory Module](./neural_memory.md): Describes the core TensorFlow/Keras model (`NeuralMemoryModule`) implementing test-time learning based on the Titans paper, including its internal structure (Memory MLP `M`, projections, gates) and update mechanisms.
*   [Metrics & Diagnostics](./metrics_store.md): Explains the `MetricsStore` used for collecting operational statistics and the diagnostic endpoints provided by the server.

Refer to the main [Architecture](../ARCHITECTURE.md) and [Component Guide](../COMPONENT_GUIDE.md) for how this server fits into the overall system and interacts with the Context Cascade Engine.
