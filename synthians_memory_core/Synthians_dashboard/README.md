# Synthians Cognitive Architecture Development Dashboard

The Synthians Cognitive Architecture Development Dashboard is a comprehensive web-based monitoring and management interface for the Synthians AI system. This dashboard provides real-time visibility, diagnostic capabilities, and interactive interfaces for core components of the cognitive architecture.

![Synthians Development Dashboard](./docs/images/dashboard-preview.png)

## 🧠 Features

- **System Overview**: Real-time monitoring of core services health and performance metrics
- **Component Dashboards**: Detailed views for Memory Core, Neural Memory, and Controlled Context Exchange (CCE)
- **Assembly Inspector**: Browse and analyze memory assemblies and their relationships
- **Real-time Logs**: Stream and filter logs from all system components
- **Admin Controls**: Maintenance and configuration actions for system components
- **Interactive Chat**: Directly engage with the Synthians AI through a chat interface
- **Configuration Management**: View and modify system configuration parameters
- **LLM Guidance Monitoring**: Track interactions with external LLM services

## 💻 Tech Stack

- **Frontend**: React, TypeScript, TailwindCSS, Shadcn UI
- **State Management**: TanStack Query, Zustand
- **Data Visualization**: Recharts
- **Backend**: Express.js, TypeScript
- **API Integration**: REST APIs to Synthians core services

## 🚀 Getting Started

### Prerequisites

- Node.js 20.x or higher
- npm 9.x or higher
- Access to the Synthians Cognitive Architecture services (Memory Core, Neural Memory, CCE)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/synthians/cognitive-dashboard.git
cd cognitive-dashboard
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit the `.env` file to include the addresses of your Synthians Cognitive Architecture services:

```
MEMORY_CORE_URL=http://localhost:8080
NEURAL_MEMORY_URL=http://localhost:8081
CCE_URL=http://localhost:8082
```

4. Start the development server:
```bash
npm run dev
```

The dashboard will be available at `http://localhost:5000`

## 🏗️ Architecture

The Synthians Cognitive Dashboard follows a client-server architecture:

### Client

The client is a React application with the following key features:
- **Component-based Structure**: Modular components for different dashboard elements
- **Real-time Data**: Uses TanStack Query for efficient data fetching and caching
- **Responsive Design**: Mobile-friendly layout with adaptive components

### Server

The server is an Express.js application that:
- Serves the frontend application
- Proxies API requests to the core Synthians services
- Provides authentication and session management
- Handles data transformation and aggregation

### Core Services

The dashboard integrates with three primary services:

1. **Memory Core**: Manages episodic and semantic memory storage and retrieval
2. **Neural Memory**: Handles vector embedding generation and maintenance
3. **CCE (Controlled Context Exchange)**: Orchestrates information flow between components

## 📁 Project Structure

```
/client             # Frontend application
  /src
    /components     # Reusable UI components
    /hooks          # Custom React hooks
    /lib            # Utilities and API clients
    /pages          # Page components
/server             # Backend Express server
  /routes           # API route handlers
  /storage          # Storage interfaces
/shared             # Shared TypeScript schemas
/docs               # Documentation
```

## 🧩 Core Components

### Memory Core

The Memory Core dashboard provides visibility into:
- Memory storage statistics
- Vector index health
- Assembly metrics and status
- Memory retrieval performance

### Neural Memory

The Neural Memory dashboard displays:
- Training status and metrics
- Emotional loop diagnostics
- Vector embedding quality metrics
- Runtime configuration

### CCE (Controlled Context Exchange)

The CCE dashboard shows:
- Active variant information
- Response metrics
- LLM guidance statistics
- Performance indicators

## 🔧 Development

### Adding New Features

1. For frontend changes:
   - Add components to `client/src/components`
   - Add pages to `client/src/pages`
   - Update routing in `client/src/App.tsx`

2. For backend changes:
   - Add API routes to `server/routes.ts`
   - Update storage interfaces in `server/storage.ts`
   - Add schema definitions to `shared/schema.ts`

### Code Style

- Follow TypeScript best practices
- Use functional components with hooks for React code
- Add comprehensive comments for complex logic
- Include type definitions for all functions and variables

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

- [Shadcn UI](https://ui.shadcn.com/) for component primitives
- [TailwindCSS](https://tailwindcss.com/) for styling
- [TanStack Query](https://tanstack.com/query) for data fetching
- [Recharts](https://recharts.org/) for visualization components