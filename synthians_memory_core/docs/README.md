# Synthians Cognitive Architecture - Documentation

Welcome to the documentation for the Synthians Cognitive Architecture, a system designed to emulate aspects of human memory and cognition.

## Overview

This documentation provides comprehensive details on the system's architecture, its core components (Memory Core, Neural Memory, Context Cascade Engine), the underlying APIs, and usage guidelines.

**Key Concepts:**

*   **Bi-Hemispheric Model:** The system loosely models the interaction between episodic/declarative memory (Memory Core - The Archive) and procedural/associative memory (Neural Memory - The Associator).
*   **QuickRecal:** A dynamic relevance score for memories, influenced by factors like recency, emotion, and surprise.
*   **Surprise Feedback:** The Neural Memory provides signals (loss, gradient norm) indicating how surprising new input is, which boosts the QuickRecal score of corresponding memories in the Core.
*   **Performance-Aware Adaptation:** The system dynamically selects optimal processing variants based on Neural Memory performance metrics and trend analysis.
*   **Asynchronous Processing:** Built with `asyncio` for efficient handling of I/O-bound operations.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** High-level overview of the system's design, principles, and the Bi-Hemispheric model.
*   **[Component Guide](./COMPONENT_GUIDE.md):** Detailed breakdown of each major component (Memory Core, Neural Memory, CCE, Tools, Testing) and their roles.
*   **[API Reference & Client Usage](./api/README.md):** Documentation for the HTTP APIs and the Python client library.
    *   [API Reference](./api/API_REFERENCE.md)
    *   [Client Usage Guide](./api/client_usage.md)
*   **[Guides](./guides/README.md):** Practical guides for setup, development, and specific use cases.
*   **[Architecture Changes](./architechture-changes.md):** Log of significant architectural decisions and evolution.
*   **[Changelog](./CHANGELOG.md):** Chronological list of changes and updates to the codebase.

## Getting Started

1.  Review the **[Architecture](./ARCHITECTURE.md)** to understand the core concepts.
2.  Explore the **[Component Guide](./COMPONENT_GUIDE.md)** for details on individual parts.
3.  If interacting programmatically, consult the **[API Reference & Client Usage](./api/README.md)**.
4.  For setup and development workflows, see the **[Guides](./guides/README.md)**.

---

*This documentation is actively maintained alongside the codebase.*
