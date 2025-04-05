# Synthians Cognitive Architecture - Documentation

Welcome to the documentation for the Synthians Cognitive Architecture, a system designed to emulate aspects of human memory and cognition through interacting, specialized components.

## Overview

This documentation provides comprehensive details on the system's architecture, its core components (Memory Core, Neural Memory, Context Cascade Engine), the underlying APIs, usage guidelines, and development practices.

**Key Concepts (Current & Upcoming):**

*   **Bi-Hemispheric Model:** Interaction between structured, indexed memory (Memory Core) and adaptive associative sequence memory (Neural Memory).
*   **Memory Assemblies:** Stable, persistent groups of related memories with composite embeddings, enabling merge operations and enhancing contextual retrieval.
*   **QuickRecal:** Dynamic relevance score for memories, influenced by factors like recency, emotion, and surprise feedback.
*   **Surprise Feedback:** Neural Memory signals novelty (loss, grad_norm) to boost corresponding memory relevance in the Core.
*   **Performance-Aware Adaptation (Phase 5+):** The Context Cascade Engine (CCE) dynamically selects optimal processing variants (MAC, MAG, MAL) based on performance, context, and LLM guidance.
*   **Explainability & Diagnostics (Planned for Phase 5.9):** Backend APIs and logic to provide insights into memory activation, assembly merging, lineage, and runtime configuration.
*   **Vector Index Reliability:** Robust FAISS (`IndexIDMap`) integration with diagnostics, consistency checks, and graceful handling of failures.
*   **Asynchronous Processing:** Built with `asyncio` for efficient I/O.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** High-level overview, principles, Bi-Hemispheric model, Assembly integration, Explainability layer concept.
*   **[Component Guide](./COMPONENT_GUIDE.md):** Detailed breakdown of Memory Core, Neural Memory, CCE, Explainability/Metrics Modules, Tools, Testing.
*   **[API Reference & Client Usage](./api/README.md):** HTTP APIs and Python client library.
    *   [API Reference](./api/API_REFERENCE.md)
    *   [Client Usage Guide](./api/client_usage.md)
*   **[Guides](./guides/README.md):** Setup, development, configuration, tooling, **Dashboard Specification**.
*   **[Architecture Changes](./architechture-changes.md):** Log of significant architectural decisions.
*   **[Changelog](./CHANGELOG.md):** Chronological list of changes.

## Getting Started

1.  Review the **[Architecture](./ARCHITECTURE.md)**.
2.  Explore the **[Component Guide](./COMPONENT_GUIDE.md)**, including the upcoming Explainability/Diagnostics sections.
3.  Consult the **[API Reference & Client Usage](./api/README.md)** for backend endpoints.
4.  Review the **[Guides](./guides/README.md)**, especially the **[Dashboard Specification](./guides/DASHBOARD_SPECIFICATION.md)** for the next phase.

*This documentation reflects both the current system state and upcoming features in Phase 5.9. Features marked as "planned" or "upcoming" are not yet implemented.*
