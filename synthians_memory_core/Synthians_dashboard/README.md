# Synthians Cognitive Architecture Development Dashboard

The Synthians Cognitive Architecture Development Dashboard is a comprehensive web-based monitoring and management interface for the Synthians AI system. This dashboard provides real-time visibility, diagnostic capabilities, and interactive interfaces for core components of the cognitive architecture.

![Synthians Development Dashboard](./docs/images/dashboard-preview.png)

## 🧠 Features

- **System Overview**: Real-time monitoring of core services health and performance metrics
- **Component Dashboards**: Detailed views for Memory Core, Neural Memory, and Controlled Context Exchange (CCE)
- **Assembly Inspector**: Browse and analyze memory assemblies and their relationships
- **Memory Lineage & Explainability**: Visualize assembly merge history and understand activation mechanisms (Phase 5.9)
- **Merge Diagnostics**: Track merge operations with detailed logs and cleanup status (Phase 5.9)
- **Runtime Configuration**: View active configuration parameters for all services
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

4. Start the development server:
```bash
npm run dev
```

This will start the development server using `tsx server/index.ts`, which properly integrates the Vite frontend with the Express backend proxy.

5. Access the dashboard at [http://localhost:5000](http://localhost:5000)

## 🔄 Architecture

The dashboard follows a client-server architecture:

- **Client**: React application with TypeScript for type safety
- **Server**: Express.js backend that proxies requests to the Synthians services
- **Shared**: Common TypeScript interfaces used by both client and server

### Directory Structure

```
├── client/            # Frontend React application
│   ├── src/           # Source code
│   │   ├── components/  # UI components
│   │   ├── contexts/    # React contexts
│   │   ├── hooks/       # Custom React hooks
│   │   ├── lib/         # Utility functions and API clients
│   │   ├── pages/       # Page components
│   │   └── App.tsx      # Main application component
├── server/            # Backend Express server
│   ├── routes.ts      # API routes definition
│   ├── config.ts      # Server configuration
│   └── index.ts       # Server entry point
├── shared/            # Shared TypeScript types
│   └── schema.ts      # Type definitions for API responses
└── docs/              # Documentation
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Architecture Overview](./docs/ARCHITECHTURE.md)
- [Development Guide](./docs/DEVELOPMENT_GUIDE.md)
- [API Reference](./docs/API_REFERENCE.md)
- [Project Structure](./docs/PROJECT_STRUCTURE.md)
- [Change Log](./docs/CHANGELOG.md)

## 🧪 Phase 5.9 Explainability Features

The dashboard integrates the Phase 5.9 explainability features from the Memory Core service:

- **Assembly Lineage**: Visualize the ancestry of memory assemblies through the merge history
- **Merge Explanation**: Understand how assemblies were formed, including similarity scores and cleanup status
- **Activation Explanation**: See why specific memories did or did not activate within an assembly
- **Merge Log**: View comprehensive logs of recent merge operations across the system
- **Runtime Configuration**: Inspect active configuration parameters affecting system behavior

These features can be toggled via the `ENABLE_EXPLAINABILITY` flag in the Memory Core configuration.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.