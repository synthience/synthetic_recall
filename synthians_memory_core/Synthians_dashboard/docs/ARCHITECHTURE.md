# Synthians Cognitive Dashboard - Architecture

This document outlines the architecture of the Synthians Cognitive Dashboard, describing the key components, data flow, and design decisions.

## System Overview

The Synthians Cognitive Dashboard is a web-based monitoring and management interface for the Synthians AI system. It provides real-time visibility and control for the three core services that make up the Synthians Cognitive Architecture:

1. **Memory Core** - Manages the storage and retrieval of episodic and semantic memories
2. **Neural Memory** - Handles vector embedding generation and memory association
3. **Controlled Context Exchange (CCE)** - Orchestrates information flow between components

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
4. **Document Changes** - Update this document and others when making architectural changes