# Synthians Cognitive Dashboard - Architecture

This document outlines the architecture of the Synthians Cognitive Dashboard, describing the key components, data flow, and design decisions.

## System Overview

The Synthians Cognitive Dashboard follows a standard client-server pattern, but with a specific twist: the "server" component acts primarily as a **Backend-For-Frontend (BFF) proxy**.

## Architecture Diagram

```mermaid
graph LR
    subgraph Browser
        A[React Frontend App]
    end

    subgraph Dashboard Server (Node.js/Express)
        B(Backend Proxy Server)
        B -- Serves --> A
        B -- API Proxy --> C[Memory Core API]
        B -- API Proxy --> D[Neural Memory API]
        B -- API Proxy --> E[CCE API]
    end

    subgraph Synthians Core Services
        C(Memory Core Service)
        D(Neural Memory Service)
        E(Context Cascade Engine)
    end

    A -- HTTP Request /api/... --> B


The Synthians Cognitive Dashboard is a web-based monitoring and management interface for the Synthians AI system. It provides real-time visibility and control for the three core services that make up the Synthians Cognitive Architecture:

1. **Memory Core** - Manages the storage and retrieval of episodic and semantic memories
2. **Neural Memory** - Handles vector embedding generation and memory association
3. **Context Cascade Engine (CCE)** - Orchestrates information flow between components

The dashboard follows a client-server architecture, with a React frontend and an Express.js backend that proxies requests to the underlying services.

## Architecture Diagram

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Memory Core   │         │  Neural Memory  │         │       CCE       │
│    Service      │         │    Service      │         │    Service      │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         │                           │                           │
         └───────────────────┬───────────────────┬──────────────┘
                             │                   │
                     ┌───────┴───────┐   ┌───────┴───────┐
                     │               │   │               │
                     │ Express.js    │◄──┤ WebSocket     │
                     │ Backend       │   │ Server        │
                     │               │   │               │
                     └───────┬───────┘   └───────────────┘
                             │                   
                     ┌───────┴───────┐           
                     │               │           
                     │  React        │           
                     │  Frontend     │           
                     │               │           
                     └───────────────┘           
```

## Components

### Frontend

The frontend is a React application built with TypeScript and organized into the following main directories:

- **/components** - Reusable UI components
  - **/dashboard** - Dashboard-specific components like metric charts
  - **/layout** - Layout components like sidebar and topbar
  - **/ui** - Generic UI components (built on shadcn/ui)
- **/hooks** - Custom React hooks
- **/lib** - Utilities and API clients
- **/pages** - Page components corresponding to routes

Key frontend technologies include:

- **React** - Component-based UI library
- **TypeScript** - Type-safe JavaScript
- **TailwindCSS** - Utility-first CSS framework
- **Shadcn UI** - Component primitives
- **TanStack Query** - Data fetching and caching
- **Zustand** - State management
- **Recharts** - Data visualization
- **Wouter** - Routing

#### Component Structure

The dashboard follows a hierarchical component structure:

1. **App.tsx** - The root component that sets up routing and providers
2. **DashboardShell** - Provides the application layout with sidebar and topbar
   - **Sidebar** - Navigation menu with links to different sections
   - **TopBar** - Header with search, refresh controls, and status indicators
3. **Page Components** - Main content areas for each route
4. **UI Components** - Reusable elements like buttons, cards, and toasts

#### JSX Runtime and React Imports

The application uses React 18's automatic JSX runtime transformation, but explicit React imports are still required in all components that use React features like hooks or JSX. The Vite configuration is set up to handle path aliases and proper JSX transformation.

### Backend

The backend is an Express.js application that serves the React frontend and provides API routes. The server has several key responsibilities:

1. **Proxy API Requests** - Forward requests to the underlying Synthians services
2. **Error Handling** - Provide consistent error responses
3. **Data Transformation** - Format data for the frontend
4. **Authentication** - Manage user sessions (future implementation)

Key backend technologies include:

- **Express.js** - Web server framework
- **TypeScript** - Type-safe JavaScript
- **cors** - Cross-origin resource sharing

### Data Storage

The dashboard itself does not maintain persistent data storage but relies on the core services for data. In a development environment, it can use in-memory storage to simulate the services.

## Data Flow

1. **User Interaction** - User interacts with the React frontend
2. **API Request** - Frontend sends a request to the Express backend
3. **Service Proxying** - Backend forwards the request to the appropriate service
4. **Service Response** - Service processes the request and sends a response
5. **UI Update** - Frontend updates the UI based on the response

For real-time updates, the dashboard uses a combination of:

1. **Polling** - Regular API requests on a configurable interval
2. **WebSockets** - For log streaming and immediate notifications (future implementation)

## Key Design Decisions

### 1. Separation of Concerns

The dashboard is designed with a clear separation between different aspects of the application:

- **UI Components** - Presentation logic
- **API Hooks** - Data fetching logic
- **State Management** - Application state
- **Routing** - Navigation logic

This makes the codebase easier to maintain and test.

### 2. Type Safety

TypeScript is used throughout the application to ensure type safety and provide better developer experience. Shared schemas are defined in a central location to ensure consistency between frontend and backend.

### 3. Responsive Design

The dashboard is designed to be responsive and work well on different screen sizes. It uses a combination of:

- Responsive grid layouts
- Collapsible sidebar
- Adaptive components

### 4. Error Handling

The application has a comprehensive error handling strategy:

- API errors are caught and displayed to the user
- Network failures are gracefully handled
- Type errors are caught at compile time

### 5. Performance Optimization

Several strategies are used to optimize performance:

- Query caching with TanStack Query
- Memoization of expensive calculations
- Lazy loading of components
- Efficient rendering with React

## Future Architectural Considerations

1. **Authentication and Authorization** - Adding user accounts and role-based access control
2. **WebSocket Integration** - For real-time updates across all services
3. **Service Worker** - For offline support and caching
4. **Analytics** - For tracking usage patterns and performance metrics

## Development Guidelines

When extending the architecture, consider the following guidelines:

1. **Maintain Type Safety** - Add proper types for all new code
2. **Follow Component Structure** - Keep components small and focused
3. **Consistent State Management** - Use the existing state management patterns
4. **Document Changes** - Update this document and others when making architectural changesComponents
1. Frontend Client (/client)
Framework: React (using Vite for development and bundling).

Language: TypeScript.

UI Library: Shadcn UI built upon Radix UI and Tailwind CSS.

Routing: Wouter handles client-side navigation (/, /memory-core, /assemblies/:id, etc.).

State Management:

TanStack Query (@tanstack/react-query): Manages server state, caching, background refresh, and request status (loading, error) for data fetched from the dashboard's backend proxy (/api/...). Hooks are defined in client/src/lib/api.ts.

Zustand: Used for simple global client state, primarily the data polling interval (client/src/lib/store.ts).

Core Structure:

main.tsx: Entry point, sets up QueryClientProvider.

App.tsx: Defines routes using <Switch> and renders the main DashboardShell.

components/layout/: Contains DashboardShell, Sidebar, TopBar.

components/ui/: Contains Shadcn UI components.

components/dashboard/: Contains reusable components specific to this dashboard's views (e.g., OverviewCard, MetricsChart, ActivationExplanationView).

pages/: Contains top-level components for each route.

lib/: Utilities, API hooks (api.ts), Zustand store (store.ts), query client setup.

2. Backend Proxy Server (/server)
Framework: Express.js (running via Node.js).

Language: TypeScript (using tsx or compiled JS for execution).

Primary Role:

Serve Frontend: In development (via Vite middleware) and production (serving static build from /dist/public), it serves the index.html and associated assets.

API Proxying: All requests starting with /api/ are intercepted. The server determines the target backend service (MC, NM, CCE) based on the path (e.g., /api/memory-core/* proxies to MEMORY_CORE_URL). It uses axios to forward the request (method, query params, body) to the appropriate internal service URL (configured via environment variables). It then forwards the response (or error) back to the frontend client. This avoids CORS issues and hides the internal service URLs from the browser.

(Development/Mocking): Can include mock handlers for endpoints if backend services are unavailable (e.g., /api/alerts uses server/storage.ts).

Key Files:

server.mjs / server/index.ts: Entry point, sets up Express app, middleware, and Vite integration (dev only).

server/routes.ts: (CRITICAL) Defines the proxy routes. Needs significant updates for Phase 5.9.

server/vite.ts: Helper for Vite middleware integration.

server/storage.ts: Simple in-memory storage for mock data (e.g., alerts).

3. Synthians Core Services (External)
These are the independent backend services (Memory Core, Neural Memory, CCE) running, potentially in Docker containers.

The dashboard proxy needs the correct URLs (e.g., http://memory-core:5010) configured via environment variables (MEMORY_CORE_URL, etc.) to reach them.

Design Decisions
BFF Proxy: Simplifies frontend development by providing a single API endpoint (/api/...) and handling CORS. It can also potentially aggregate or cache data in the future.

TanStack Query: Provides robust caching, background refresh, and request state management, simplifying data fetching logic in components.

Shadcn UI & Tailwind: Offers a flexible and consistent design system based on unstyled primitives.

TypeScript: Enforces type safety across the frontend, backend proxy, and shared schemas.

Polling: Simple mechanism for periodic data refresh, managed by Zustand and TanStack Query invalidation. Real-time updates via WebSockets are a future enhancement.