# Synthians Cognitive Dashboard - Quick Start Guide

This guide will help you quickly set up the Synthians Cognitive Dashboard for local development.

## Prerequisites

- Node.js 20.x or higher
- npm 9.x or higher
- Git

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/synthians/cognitive-dashboard.git
cd cognitive-dashboard
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment

Create a `.env` file in the root directory with the following variables:

```
# Core Service URLs
MEMORY_CORE_URL=http://localhost:8080
NEURAL_MEMORY_URL=http://localhost:8081
CCE_URL=http://localhost:8082

# Development Settings
NODE_ENV=development
PORT=5000
```

### 4. Start the Development Server

```bash
npm run dev
```

This will start both the Express backend server and the frontend development server. The application will be available at `http://localhost:5000`.

## Project Structure

```
/client             # Frontend React application
  /src
    /components     # UI components
    /hooks          # Custom React hooks
    /lib            # Utilities and API clients
    /pages          # Page components
/server             # Express backend
  /routes.ts        # API routes
  /storage.ts       # Storage interfaces
/shared             # Shared TypeScript schemas
```

## Key Development Workflows

### Adding a New Dashboard Page

1. Create a new page component in `client/src/pages/`
2. Add the route in `client/src/App.tsx`
3. Add a sidebar navigation link in `client/src/components/layout/Sidebar.tsx`

Example page component:

```tsx
import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function NewFeature() {
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">New Feature</h2>
          <p className="text-sm text-gray-400">
            Description of the new feature
          </p>
        </div>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Feature Details</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Feature content goes here */}
        </CardContent>
      </Card>
    </>
  );
}
```

### Adding a New API Endpoint

1. Add the endpoint in `server/routes.ts`
2. Create a client-side API hook in `client/src/lib/api.ts`

Example API endpoint:

```typescript
// In server/routes.ts
app.get("/api/new-feature/data", (req, res) => {
  // Implementation
  res.json({ data: { /* your data */ } });
});

// In client/src/lib/api.ts
export const useNewFeatureData = () => {
  return useQuery({
    queryKey: ["/api/new-feature/data"],
    staleTime: 30000
  });
};
```

### Working with Mock Data During Development

While developing, you may need to work with mock data before connecting to real services:

1. Create mock handlers in the server routes
2. Use consistent data structures based on the shared schema

Example mock implementation:

```typescript
// In server/routes.ts
app.get("/api/memory-core/assemblies", (req, res) => {
  // Return mock data following the Assembly schema
  res.json({
    data: [
      {
        id: "assembly-1",
        name: "Test Assembly",
        description: "A test assembly for development",
        member_count: 42,
        keywords: ["test", "development"],
        tags: ["important"],
        topics: ["testing"],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        memory_ids: ["mem-1", "mem-2"]
      }
    ]
  });
});
```

## Troubleshooting

### API Connection Issues

If you're having trouble connecting to the core services:

1. Check that the environment variables are correctly set
2. Verify that the services are running on the expected ports
3. Check for CORS issues in the browser dev tools

### Build Errors

If you encounter build errors:

1. Check for TypeScript errors
2. Ensure all dependencies are installed
3. Clear the node_modules folder and reinstall

```bash
rm -rf node_modules
npm install
```

## Next Steps

After setting up your development environment, you might want to:

1. Explore the existing codebase to understand the architecture
2. Check out the open issues for potential contributions
3. Run the test suite to ensure everything is working correctly

For more detailed information, refer to the main [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md) files.