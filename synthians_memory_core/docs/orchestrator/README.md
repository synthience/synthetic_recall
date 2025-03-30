# Context Cascade Engine Documentation

This directory contains documentation for the Context Cascade Engine (CCE) and its components that orchestrate the cognitive cycle.

## Contents

* [Context Cascade Engine](./cce.md): **(Placeholder)** Overview of the `ContextCascadeEngine` class that implements the refactored cognitive flow.
* [Titans Variants](./titans_variants.md): Documentation on the MAC, MAG, and MAL variants from the Titans paper and their implementation in the CCE.
* [Attention Mechanisms](./attention.md): Details on how attention is calculated and applied in the different variant implementations.
* [Sequence Context Management](./sequence_context.md): **(Placeholder)** Documentation on the `SequenceContextManager` that maintains history for attention operations.

## Technical Details

* **Variant Flow**: Different processing paths for MAC (post-retrieval attention), MAG (gated update), and MAL (value modification).
* **TensorFlow Integration**: How lazy loading of TensorFlow avoids NumPy version conflicts.
* **Surprise Feedback Loop**: How loss and gradient norm from Neural Memory are converted into QuickRecal score boosts in Memory Core.
* **History Management**: How the sequence context of embeddings, keys, values, and outputs is maintained and used for attention calculations.
