# Synthians Cognitive Dashboard - Documentation

Welcome to the documentation for the Synthians Cognitive Dashboard project.

## Overview

This project implements a web-based user interface for monitoring, inspecting, and interacting with the Synthians Cognitive Architecture services (Memory Core, Neural Memory, CCE). It aims to provide real-time visibility into system health, performance metrics, internal states (like memory assemblies and merge history), configuration, and includes placeholders for future interactive features like live logging and chat.

**Phase Context:** This documentation describes the dashboard structure as planned for integration with the **Phase 5.9 backend features** (Explainability & Diagnostics). Many UI components are present, but the connections to the specific Phase 5.9 backend APIs are **TODO** items.

## Key Features (Planned & Partially Implemented)

*   **Service Status Monitoring:** Health, uptime, version for MC, NM, CCE.
*   **Core Metrics Display:** Memory/assembly counts, vector index stats, NM performance (loss/grad), CCE variant selection.
*   **Assembly Inspector:** Browse assemblies, view details, members, metadata, and **planned explainability views (lineage, activation, merge)**.
*   **Configuration Viewer:** Display **sanitized** runtime configurations from services.
*   **Diagnostics Views:** Display **merge log history**.
*   **(Placeholders):** Real-time Log Streaming, Interactive Chat, Admin Actions.

## Technology Stack

*   **Frontend:** React (Vite), TypeScript, Tailwind CSS, Shadcn UI
*   **Routing:** Wouter
*   **State Management:** TanStack Query (Server State), Zustand (Client State - e.g., polling)
*   **Charting:** Recharts
*   **Backend (Dashboard Proxy):** Express.js (Node.js), TypeScript, Axios (for proxying)
*   **(Optional for Dev):** In-memory storage for mocking alerts.

## Navigation

*   **[Architecture](./ARCHITECTURE.md):** Dashboard's internal architecture (Client, Proxy Backend).
*   **[Project Structure](./PROJECT_STRUCTURE.md):** File tree and component overview.
*   **[Data Flow & API](./DATA_FLOW_API.md):** How data is fetched via the proxy backend, with **TODOs** for Phase 5.9 integration.
*   **[Development Guide](./DEVELOPMENT_GUIDE.md):** Setup, running, adding features, best practices.
*   **[UI Components](./UI_COMPONENTS.md):** Overview of key layout and dashboard-specific components.

## Getting Started

Refer to the **[Development Guide](./DEVELOPMENT_GUIDE.md)** for setup and running instructions.