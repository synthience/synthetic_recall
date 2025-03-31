# Context Cascade Engine Documentation

This directory contains documentation for the Context Cascade Engine (CCE) and its components that orchestrate the cognitive cycle.

## Contents

* [Context Cascade Engine](./cce.md): **(Placeholder)** Overview of the `ContextCascadeEngine` class that implements the refactored cognitive flow.
* [Titans Variants](./titans_variants.md): Documentation on the MAC, MAG, and MAL variants from the Titans paper and their implementation in the CCE.
* [Attention Mechanisms](./attention.md): Details on how attention is calculated and applied in the different variant implementations.
* [Sequence Context Management](./sequence_context.md): **(Placeholder)** Documentation on the `SequenceContextManager` that maintains history for attention operations.
* [Performance-Aware Selection](./performance_aware_selection.md): Documentation on how the system dynamically selects variants based on Neural Memory performance metrics and trend analysis.

## Technical Details

* **Variant Flow**: Different processing paths for MAC (post-retrieval attention), MAG (gated update), and MAL (value modification).
* **TensorFlow Integration**: How lazy loading of TensorFlow avoids NumPy version conflicts.
* **Surprise Feedback Loop**: How loss and gradient norm from Neural Memory are converted into QuickRecal score boosts in Memory Core.
* **Performance Tracking**: How Neural Memory performance metrics (loss, gradient norm) are tracked and analyzed for trend detection.
* **Dynamic Variant Selection**: How the `VariantSelector` uses performance metrics, trends, and other factors to select the optimal variant.
* **History Management**: How the sequence context of embeddings, keys, values, and outputs is maintained and used for attention calculations.
