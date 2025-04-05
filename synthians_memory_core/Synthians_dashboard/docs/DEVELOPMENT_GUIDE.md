# Dashboard Development Guide

This document provides an overview of the development process and key components of the Synthians Cognitive Dashboard.

## Getting Started

### Prerequisites

- Node.js 20 or higher
- npm 9 or higher
- Access to backend services: Memory Core, Neural Memory, and Cognitive Core Engine (CCE)

### Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/Synthians/cognitive-dashboard.git

# Navigate to the project directory
cd cognitive-dashboard

# Install dependencies
npm install
```

### Running the Dashboard

```bash
# Start the development server
npm run dev
```

This will start the development server using `tsx server/index.ts`, which properly integrates the Vite frontend with the Express backend proxy. The dashboard will be available at http://localhost:5000.

## Development Workflow

1. The frontend code is in the `client/` directory
2. The backend proxy is in the `server/` directory
3. Shared TypeScript interfaces are in the `shared/` directory
4. The backend proxy forwards requests to the actual services (Memory Core, Neural Memory, CCE)

### Key Configuration

Backend service URLs are configured in `server/config.ts`.

# Dashboard UI Components

This document provides an overview of the key React components used in the dashboard.

## Layout Components (`client/src/components/layout/`)

*   **`DashboardShell.tsx`:** The main application wrapper. Includes the `Sidebar` and `TopBar` and renders the main page content as children. Handles mobile sidebar toggling.
*   **`Sidebar.tsx`:** The left-hand navigation menu. Contains `NavLink` components for routing. Uses `wouter`'s `useLocation` to highlight the active page.
*   **`TopBar.tsx`:** The header bar. Includes a mobile sidebar toggle, search bar (placeholder), manual refresh button (`RefreshButton`), polling rate selector, and basic service status links.
*   **`ServiceStatus.tsx`:** A small component displaying the health status (Healthy, Unhealthy, etc.) of a backend service with a colored dot and text.

## UI Primitives (`client/src/components/ui/`)

*   These are standard components generated from **Shadcn UI**. They provide the building blocks for the interface (Buttons, Cards, Tables, Forms, Toasts, etc.). Refer to the Shadcn UI documentation for usage details.
*   Key components used extensively: `Card`, `Button`, `Table`, `Badge`, `Skeleton`, `Tabs`, `Select`, `Input`, `ScrollArea`, `Progress`, `Alert`, `Toast`, `Collapsible`.

## Dashboard Specific Components (`client/src/components/dashboard/`)

These components are tailored for displaying specific types of data within the dashboard views.

*   **`OverviewCard.tsx`:** Displays a summary card for a specific service (MC, NM, CCE), showing health status and key metrics.
*   **`MetricsChart.tsx`:** A reusable line chart component (using `Recharts`) for displaying time-series data (e.g., NM Loss/Grad Norm). Includes time range selection.
*   **`CCEChart.tsx`:** A specialized bar chart (using `Recharts`) for visualizing CCE variant distribution over time.
*   **`AssemblyTable.tsx`:** Displays a list of assemblies in a table format, including name, member count, update time, and **sync status**. Links to the detail view.
*   **`SystemArchitecture.tsx`:** Renders a static SVG-based diagram showing the high-level interaction between MC, NM, and CCE.
*   **`DiagnosticAlerts.tsx`:** Displays a list of recent alerts (currently mocked via `server/storage.ts`).

### Phase 5.9 Explainability Components:

*   **`ActivationExplanationView.tsx`:** Displays the detailed explanation for why a memory did or did not activate within an assembly. Renders data from the `useExplainActivation` hook.
*   **`MergeExplanationView.tsx`:** Displays the details of how an assembly was formed via a merge, including source assemblies, timestamp, and cleanup status. Renders data from the `useExplainMerge` hook.
*   **`LineageView.tsx`:** Displays the merge ancestry of an assembly in a list or tree-like format. Renders data from the `useAssemblyLineage` hook.
*   **`MergeLogView.tsx`:** Displays recent merge events fetched from the `/diagnostics/merge_log` endpoint via the `useMergeLog` hook. Correlates merge and cleanup events.

## API and Data Fetching

The dashboard uses TanStack Query (React Query) for data fetching, caching and state management. All API hooks are defined in `client/src/lib/api.ts`.

### Key API Hooks

**Core Service Status:**
- `useMemoryCoreHealth`, `useNeuralMemoryHealth`, `useCCEHealth`: Basic health checks
- `useMemoryCoreStats`, `useNeuralMemoryStatus`, `useCCEStatus`: Detailed status information

**Memory & Assemblies:**
- `useAssemblies`: Fetch all assemblies
- `useAssembly`: Fetch details for a specific assembly

**Phase 5.9 Explainability:**
- `useExplainActivation`: Get explanation of memory activation in an assembly
- `useExplainMerge`: Get explanation of how an assembly was formed by merge
- `useAssemblyLineage`: Get the merge ancestry history of an assembly
- `useMergeLog`: Get the recent merge events log
- `useRuntimeConfig`: Get the runtime configuration of a service

### Feature Detection

The dashboard uses the `FeaturesContext` to detect and manage feature flags from the backend. Currently, it checks for the `ENABLE_EXPLAINABILITY` flag from Memory Core to determine whether to enable the Phase 5.9 explainability features in the UI.