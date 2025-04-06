# Synthians Cognitive Architecture - Documentation

Welcome to the documentation for the Synthians Cognitive Architecture, a system designed to emulate aspects of human memory and cognition.

## Overview

This documentation provides comprehensive details on the system's architecture, its core components (Memory Core, Neural Memory, Context Cascade Engine), the underlying APIs, and usage guidelines.

**Key Concepts (Post Phase 5.9 Backend Implementation):**

*   **Bi-Hemispheric Model:** Interaction between episodic/declarative memory (Memory Core) and adaptive associative memory (Neural Memory).
*   **Memory Assemblies:** Stable, persistent groups of related memories with composite embeddings, enabling merge operations and enhancing contextual retrieval. Includes merge lineage tracking (`merged_from`).
*   **QuickRecal:** Dynamic relevance score for memories, influenced by factors like recency, emotion, and surprise feedback.
*   **Surprise Feedback:** Neural Memory signals novelty (loss, grad_norm) to boost corresponding memory relevance in the Core.
*   **Performance-Aware Adaptation (Phase 5+):** System dynamically selects optimal processing variants (MAC, MAG, MAL) based on performance and context.
*   **Vector Index Reliability:** Robust FAISS (`IndexIDMap`) integration with diagnostics, consistency checks, and graceful handling of failures via a retry queue.
*   **Explainability & Diagnostics (Phase 5.9):** Backend logic and APIs implemented to provide insights into assembly activation, merge history, lineage, runtime configuration, and activation stats. Controlled via `ENABLE_EXPLAINABILITY` flag.
*   **Asynchronous Processing:** Built with `asyncio` for efficient I/O.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** High-level overview, principles, Bi-Hemispheric model, Assembly integration, Explainability/Diagnostics layer.
*   **[Component Guide](./COMPONENT_GUIDE.md):** Detailed breakdown of Memory Core, Neural Memory, CCE, Explainability/Metrics Modules, Tools, Testing.
*   **[API Reference & Client Usage](./api/README.md):** HTTP APIs and Python client library.
    *   [API Reference](./api/API_REFERENCE.md)
    *   [Client Usage Guide](./api/client_usage.md)
*   **[Guides](./guides/README.md):** Setup, development, configuration, tooling, **Dashboard Specification**.
*   **[Architecture Changes](./architechture-changes.md):** Log of significant architectural decisions.
*   **[Changelog](./CHANGELOG.md):** Chronological list of changes.

## Getting Started

1.  Review the **[Architecture](./ARCHITECTURE.md)**.
2.  Explore the **[Component Guide](./COMPONENT_GUIDE.md)**.
3.  Consult the **[API Reference & Client Usage](./api/README.md)**.
4.  See the **[Guides](./guides/README.md)** for setup/development.

*This documentation is actively maintained alongside the codebase.*
