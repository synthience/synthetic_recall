# Synthians Cognitive Dashboard Implementation Guide

**Note: This implementation guide is for the planned Phase 5.9.1 dashboard. The backend APIs needed for many dashboard features are not yet implemented in Phase 5.8 and will be added in Phase 5.9.**

This guide provides practical advice for implementing the Synthians Cognitive Dashboard based on the [specification](./DASHBOARD_SPECIFICATION.md).

## Development Environment Setup

1. **Node.js and npm**: Ensure you're using Node.js 18+ and npm 9+
2. **Project Structure**:
   ```
   Synthians_dashboard/
   ├── client/            # React frontend
   │   ├── src/
   │   │   ├── components/
   │   │   ├── pages/
   │   │   ├── lib/
   │   │   └── ...
   ├── server/            # Express.js backend proxy
   │   ├── routes.ts
   │   ├── storage.ts
   │   └── ...
   ├── shared/            # Shared types and utilities
   │   └── ...
   └── package.json
   ```
3. **Dependencies**: Install core dependencies including React, React Router, TanStack Query, Tailwind CSS, and Shadcn UI

## Performance Considerations

Different endpoints have different update frequencies. Relying on fixed-interval polling for all data sources (`/stats`, `/assemblies`, `/merge_log`, `/diagnostics_emoloop`, `/metrics/recent_cce_responses`) can lead to either outdated information or excessive backend load.

**Recommended Approaches:**
- **Leverage TanStack Query:** Use its caching (`staleTime`, `cacheTime`) effectively. Set longer `staleTime` for endpoints like `/config/runtime` that change infrequently.
- **Selective Invalidation:** Implement logic where certain actions (e.g., a successful merge reported in `/stats`) trigger targeted invalidation of specific queries (like `/assemblies` or `/diagnostics/merge_log`).
- **Adaptive Polling:** Consider adjusting polling rates based on system activity indicators found in `/stats` (e.g., poll `/merge_log` more often if recent merges are detected).
- **Component-Level Fetching:** Fetch detailed data (like `/explain_*`) only when a specific component (e.g., Assembly Inspector) is mounted or interacted with, not globally.

## Error Handling

Robust error handling is critical for a monitoring dashboard. The system might experience temporary API unavailability, partial failures, or missing features.

**Recommended Approaches:**
- **Graceful Degradation:** Show partial data when some endpoints fail but others succeed.
- **Tiered Error Handling:** 
  - **Level 1:** Component-level fallbacks (e.g., "Data unavailable")
  - **Level 2:** Page-level fallbacks (e.g., alternative views)
  - **Level 3:** Application-level fallbacks (e.g., status dashboard mode)
- **Exponential Backoff:** When API calls fail, implement exponential backoff for retries.
- **Error Boundaries:** Use React error boundaries to prevent entire UI crashes.

## Handling Feature Flags

The Synthians system uses feature flags to enable/disable certain components. These must be respected in the dashboard UI.

Frontend UI elements for explainability being visible, but the backend feature flag `ENABLE_EXPLAINABILITY` is `false`, leading to 403/404 errors, creates a confusing user experience.

**Recommended Approaches:**
- The dashboard should fetch the runtime config (`/config/runtime/memory-core`) on load and use the `enable_explainability` value to conditionally render UI elements.
- Disable buttons or add informative tooltips for features that are not available based on backend configuration.

## Implementation Examples

### Service Status Component

```tsx
import { useMemoryCoreHealth, useNeuralMemoryHealth, useCCEHealth } from '@/lib/api';

const ServiceStatus = () => {
  const mcHealth = useMemoryCoreHealth();
  const nmHealth = useNeuralMemoryHealth();
  const cceHealth = useCCEHealth();
  
  // Loading states
  if (mcHealth.isLoading || nmHealth.isLoading || cceHealth.isLoading) {
    return <StatusSkeleton />;
  }
  
  // Error states - show partial data if available
  const hasErrors = mcHealth.isError || nmHealth.isError || cceHealth.isError;
  
  return (
    <div className="grid grid-cols-3 gap-4">
      <StatusCard 
        service="Memory Core" 
        status={mcHealth.data?.status || "unknown"} 
        error={mcHealth.error}
        metrics={mcHealth.data?.memory_count ? 
                `${mcHealth.data.memory_count} memories` : undefined}
      />
      <StatusCard 
        service="Neural Memory" 
        status={nmHealth.data?.status || "unknown"} 
        error={nmHealth.error}
      />
      <StatusCard 
        service="CCE" 
        status={cceHealth.data?.status || "unknown"} 
        error={cceHealth.error}
        metrics={cceHealth.data?.active_variant ? 
                `Active: ${cceHealth.data.active_variant}` : undefined}
      />
      {hasErrors && (
        <div className="col-span-3 bg-amber-50 p-4 rounded-md text-amber-800">
          Some services are experiencing issues. Check service status above.
        </div>
      )}
    </div>
  );
};
```

### Feature Flag Aware Component

For conditionally rendering explainability features:

```tsx
import { useMemoryCoreConfig } from '@/lib/api';

const ExplainabilityFeatures = ({ assemblyId }) => {
  // Get runtime config to check if explainability is enabled
  const { data, isLoading, isError } = useMemoryCoreConfig();
  
  // Determine if explainability is enabled
  const enabled = data?.success &&
    data?.config?.enable_explainability === true;
  
  if (isLoading) return <Spinner />;
  
  if (isError || !enabled) {
    return (
      <div className="p-4 border rounded-md bg-gray-50">
        <h3 className="font-medium">Explainability Features</h3>
        <p className="text-gray-500">
          {isError ? 
            "Unable to determine if explainability features are available." :
            "Explainability features are currently disabled on the server."}
        </p>
      </div>
    );
  }
  
  // Render actual explainability UI when enabled
  return (
    <div className="p-4 border rounded-md">
      <h3 className="font-medium">Explainability Features</h3>
      <div className="flex gap-3 mt-4">
        <Button onClick={() => handleExplainActivation(assemblyId)}>
          Explain Activation
        </Button>
        <Button onClick={() => handleExplainMerge(assemblyId)}>
          Explain Merge
        </Button>
        <Button onClick={() => handleViewLineage(assemblyId)}>
          View Lineage
        </Button>
      </div>
      {/* Results display */}
    </div>
  );
};
```

### Handling Async Operations

```tsx
const AssemblyInspector = ({ assemblyId }) => {
  // Core assembly data
  const { data: assembly, isLoading, isError } = useAssembly(assemblyId);
  
  // Explanation queries (not auto-fetched)
  const explainActivation = useExplainActivation(assemblyId);
  const explainMerge = useExplainMerge(assemblyId);
  const lineage = useAssemblyLineage(assemblyId);
  
  // Handle activation explanation
  const handleExplainActivation = async (memoryId) => {
    // Refetch with the specific memoryId parameter
    await explainActivation.refetch({ queryKey: [
      'memory-core', 'assemblies', assemblyId, 'explain_activation', 
      { memory_id: memoryId }
    ]});
  };
  
  // Results display with loading/error states
  return (
    <div>
      {/* Assembly basic details */}
      {isLoading && <Spinner />}
      {isError && <ErrorMessage error={error} />}
      {assembly && (
        <AssemblyDetails assembly={assembly.assembly} />
      )}
      
      {/* Explainability section */}
      <ExplainabilityFeatures 
        assemblyId={assemblyId} 
        onExplainActivation={handleExplainActivation}
        activationResult={explainActivation.data}
        activationLoading={explainActivation.isLoading}
        activationError={explainActivation.error}
        /* Other props */
      />
    </div>
  );
};
```

## Best Practices

1. **Data Fetching Principles**:
   - Collocate data fetching with the components that need the data
   - Use TanStack Query's caching and stale-time effectively
   - Implement loading and error states for all data fetching

2. **UI Performance**:
   - Virtualize long lists (especially for `/assemblies` or `/merge_log`)
   - Use skeleton loaders for initial data fetch
   - Consider windowing for data-heavy charts and graphs

3. **Testing**:
   - Write unit tests for UI components with mock data
   - Test error and loading states
   - Use MSW (Mock Service Worker) to test API integration

4. **Accessibility**:
   - Ensure proper contrast for charts and visualizations
   - Add keyboard navigation for interactive elements
   - Use proper ARIA labels for complex components

The Synthians Cognitive Dashboard will be a powerful tool for understanding and monitoring the Synthians Cognitive Architecture once implemented in Phase 5.9.1. This guide provides a foundation for implementing the dashboard in a robust, maintainable way that respects the system's architecture and constraints.