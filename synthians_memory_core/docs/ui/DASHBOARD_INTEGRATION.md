# Synthians Dashboard Integration Guide (Phase 5.9)

**Version:** 1.0.0  
**Date:** April 2025

This document provides guidance for frontend developers integrating the Synthians Cognitive Dashboard with the Phase 5.9 API endpoints for explainability and diagnostics.

## Overview

The Synthians Cognitive Dashboard is a React-based frontend that provides a visual interface for interacting with the Synthians Memory Core, Neural Memory Server, and Context Cascade Engine. Phase 5.9 extends the dashboard to visualize explainability and diagnostics data provided by the new API endpoints.

## Prerequisites

- Node.js 16.x or higher
- Access to the Synthians API server (running locally or remotely)
- The `ENABLE_EXPLAINABILITY` flag must be set to `true` in the Memory Core configuration to access these features

## Key Components and Integration Points

### 1. API Integration Layer

The API client code in `client/src/lib/api.ts` has been updated to include the new explainability and diagnostics endpoints. Use these hooks to fetch data from the API:

```typescript
// Explainability hooks
const { data, isLoading, isError, error } = useExplainActivation(assemblyId, memoryId);
const { data, isLoading, isError, error } = useExplainMerge(assemblyId);
const { data, isLoading, isError, error } = useAssemblyLineage(assemblyId, maxDepth);

// Diagnostics hooks
const { data, isLoading, isError, error } = useMergeLog(limit);
const { data, isLoading, isError, error } = useRuntimeConfig(serviceName);
const { data, isLoading, isError, error } = useMemoryCoreStats();
const { data, isLoading, isError, error } = useCheckIndexIntegrity();
```

Remember that data from these hooks is nested - the actual payload is at `data.data`. Refer to the API schemas in `shared/schema.ts` for the specific data structure of each response.

### 2. Feature Flag Integration

All explainability and diagnostics features should respect the `ENABLE_EXPLAINABILITY` feature flag. Use the `useFeatures` hook from the Features context to check if these features are enabled:

```typescript
import { useFeatures } from '../contexts/FeaturesContext';

function MyComponent() {
  const { explainabilityEnabled } = useFeatures();
  
  return (
    <div>
      {explainabilityEnabled ? (
        <ExplainabilityFeature />
      ) : (
        <FeatureDisabledMessage />
      )}
    </div>
  );
}
```

### 3. UI Components

#### Assembly Inspector

The Assembly Inspector component (`components/dashboard/assembly-inspector.tsx`) has been updated to include three new tabs:

1. **Activation Tab**: Displays a list of memories in the assembly with the option to explain why each memory is part of the assembly.
   - Use the `useExplainActivation` hook to fetch activation explanations for selected memories.
   - Implement a "loading" state while waiting for the explanation data.
   - Display the similarity scores and threshold information in a visually intuitive way.

2. **Merge Tab**: Shows how this assembly was formed (if it was created by a merge operation).
   - Use the `useExplainMerge` hook to fetch merge information.
   - Display source assemblies with links to their detail pages.
   - Show merge similarity scores and cleanup status.

3. **Lineage Tab**: Visualizes the ancestry of the assembly through its merge history.
   - Use the `useAssemblyLineage` hook to fetch lineage data.
   - Implement a tree or graph visualization to show the ancestry relationships.
   - Include depth controls to limit or expand the visualization.

#### Diagnostics Dashboard

The Diagnostics Dashboard component (`components/dashboard/diagnostics-dashboard.tsx`) is a new component that displays system-wide diagnostics information:

1. **Merge Log Tab**: Displays a table of recent merge operations.
   - Use the `useMergeLog` hook to fetch the merge log data.
   - Implement sorting and filtering controls.
   - Show merge statuses with appropriate visual indicators (green for completed, red for failed, etc.).

2. **Vector Index Health Tab**: Shows the health of the vector index.
   - Use the `useCheckIndexIntegrity` hook to fetch index health data.
   - Display a summary card with overall health status.
   - Show detailed diagnostics in expandable sections.

3. **Runtime Configuration Tab**: Displays the current runtime configuration.
   - Use the `useRuntimeConfig` hook to fetch configuration data.
   - Implement a service selector to switch between different service configurations.
   - Display configuration values in a readable format.

#### Memory Core Overview

The Memory Core Overview component (`components/dashboard/overview-card.tsx`) has been updated to include activation statistics:

- Use the `useMemoryCoreStats` hook to fetch statistics data.
- Display the top activated assemblies in a chart or table.
- Show total memory and assembly counts with appropriate visualizations.

### 4. Error Handling

Implement robust error handling for API requests. The API returns standardized error responses that you should handle appropriately:

```typescript
function AssemblyLineageView({ assemblyId }) {
  const { data, isLoading, isError, error } = useAssemblyLineage(assemblyId);
  
  if (isLoading) {
    return <Skeleton />;
  }
  
  if (isError) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error?.message || 'Failed to load lineage data'}
        </AlertDescription>
      </Alert>
    );
  }
  
  if (!data?.data) {
    return <div>No lineage data available</div>;
  }
  
  return <LineageTree data={data.data.lineage} />;
}
```

### 5. Loading States

Implement loading states using the Skeleton component for all components that display data from API requests:

```typescript
function MergeLogView() {
  const { data, isLoading } = useMergeLog();
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }
  
  // Render actual data
  return <MergeLogTable entries={data?.data?.entries || []} />;
}
```

## Example Integration: Assembly Inspector

Here's a complete example of how to integrate the explainability features in the Assembly Inspector component:

```typescript
import { useParams } from 'react-router-dom';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Alert, AlertDescription, AlertTitle } from '../ui/alert';
import { Skeleton } from '../ui/skeleton';
import { useFeatures } from '../../contexts/FeaturesContext';
import {
  useAssembly,
  useExplainMerge,
  useAssemblyLineage,
  useExplainActivation
} from '../../lib/api';

export function AssemblyInspector() {
  const { id } = useParams<{ id: string }>();
  const { explainabilityEnabled } = useFeatures();
  const { data: assemblyData, isLoading: assemblyLoading } = useAssembly(id);
  
  // Fetch merge explanation data (disabled by default, triggered by user)
  const {
    data: mergeData,
    isLoading: mergeLoading,
    isError: mergeError,
    error: mergeErrorData,
    refetch: refetchMerge
  } = useExplainMerge(id, { enabled: false });
  
  // Fetch lineage data (disabled by default, triggered by user)
  const {
    data: lineageData,
    isLoading: lineageLoading,
    isError: lineageError,
    error: lineageErrorData,
    refetch: refetchLineage
  } = useAssemblyLineage(id, { enabled: false });
  
  const [selectedMemoryId, setSelectedMemoryId] = useState(null);
  
  // Fetch activation explanation for selected memory
  const {
    data: activationData,
    isLoading: activationLoading,
    isError: activationError,
    error: activationErrorData,
    refetch: refetchActivation
  } = useExplainActivation(id, selectedMemoryId, { enabled: false });
  
  const handleExplainMerge = () => {
    refetchMerge();
  };
  
  const handleExplainLineage = () => {
    refetchLineage();
  };
  
  const handleExplainActivation = (memoryId) => {
    setSelectedMemoryId(memoryId);
    refetchActivation();
  };
  
  if (assemblyLoading) {
    return <Skeleton className="h-96 w-full" />;
  }
  
  const assembly = assemblyData?.data;
  if (!assembly) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Assembly not found</AlertDescription>
      </Alert>
    );
  }
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Assembly: {assembly.name || assembly.id}</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="details">
          <TabsList>
            <TabsTrigger value="details">Details</TabsTrigger>
            <TabsTrigger value="memories">Memories</TabsTrigger>
            {explainabilityEnabled && (
              <>
                <TabsTrigger value="activation">Activation</TabsTrigger>
                <TabsTrigger value="merge">Merge</TabsTrigger>
                <TabsTrigger value="lineage">Lineage</TabsTrigger>
              </>
            )}
          </TabsList>
          
          <TabsContent value="details">
            {/* Basic assembly details here */}
          </TabsContent>
          
          <TabsContent value="memories">
            {/* List of memories in the assembly */}
          </TabsContent>
          
          {explainabilityEnabled && (
            <>
              <TabsContent value="activation">
                <h3>Activation Explanation</h3>
                <p>Select a memory to explain why it's part of this assembly:</p>
                <ul>
                  {assembly.memory_ids.map(memoryId => (
                    <li key={memoryId}>
                      {memoryId}
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleExplainActivation(memoryId)}
                      >
                        Explain
                      </Button>
                    </li>
                  ))}
                </ul>
                
                {activationLoading && <Skeleton className="h-40 w-full" />}
                
                {activationError && (
                  <Alert variant="destructive">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                      {activationErrorData?.message || 'Failed to load activation data'}
                    </AlertDescription>
                  </Alert>
                )}
                
                {activationData?.data && (
                  <div className="mt-4 p-4 border rounded">
                    <h4>Explanation</h4>
                    <p>
                      <strong>Memory ID:</strong> {activationData.data.memory_id}
                    </p>
                    <p>
                      <strong>Similarity:</strong> {activationData.data.calculated_similarity.toFixed(4)}
                    </p>
                    <p>
                      <strong>Threshold:</strong> {activationData.data.activation_threshold.toFixed(4)}
                    </p>
                    <p>
                      <strong>Result:</strong> {activationData.data.passed_threshold ? 'Activated' : 'Not Activated'}
                    </p>
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="merge">
                <h3>Merge History</h3>
                <Button 
                  onClick={handleExplainMerge}
                  disabled={mergeLoading}
                >
                  {mergeLoading ? 'Loading...' : 'Explain Merge'}
                </Button>
                
                {mergeLoading && <Skeleton className="h-40 w-full" />}
                
                {mergeError && (
                  <Alert variant="destructive">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                      {mergeErrorData?.message || 'Failed to load merge data'}
                    </AlertDescription>
                  </Alert>
                )}
                
                {mergeData?.data && mergeData.data.is_merged ? (
                  <div className="mt-4 p-4 border rounded">
                    <h4>Merge Details</h4>
                    <p><strong>Merged At:</strong> {new Date(mergeData.data.merge_timestamp).toLocaleString()}</p>
                    <p><strong>Similarity:</strong> {mergeData.data.similarity_at_merge.toFixed(4)}</p>
                    <p><strong>Threshold:</strong> {mergeData.data.merge_threshold.toFixed(4)}</p>
                    <p><strong>Cleanup Status:</strong> {mergeData.data.cleanup_status}</p>
                    <h5>Source Assemblies:</h5>
                    <ul>
                      {mergeData.data.source_assemblies.map(source => (
                        <li key={source.id}>
                          <a href={`/assemblies/${source.id}`}>
                            {source.name || source.id}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : mergeData?.data ? (
                  <div className="mt-4 p-4 border rounded">
                    <p>This assembly was not created by a merge operation.</p>
                  </div>
                ) : null}
              </TabsContent>
              
              <TabsContent value="lineage">
                <h3>Assembly Lineage</h3>
                <Button 
                  onClick={handleExplainLineage}
                  disabled={lineageLoading}
                >
                  {lineageLoading ? 'Loading...' : 'Trace Lineage'}
                </Button>
                
                {lineageLoading && <Skeleton className="h-60 w-full" />}
                
                {lineageError && (
                  <Alert variant="destructive">
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>
                      {lineageErrorData?.message || 'Failed to load lineage data'}
                    </AlertDescription>
                  </Alert>
                )}
                
                {lineageData?.data?.lineage && (
                  <div className="mt-4">
                    <h4>Ancestry Tree</h4>
                    <ul className="pl-4">
                      {lineageData.data.lineage.map(entry => (
                        <li key={entry.assembly_id} style={{ marginLeft: `${entry.depth * 20}px` }}>
                          <div className={`p-2 border rounded ${entry.status !== 'normal' ? 'bg-yellow-50' : ''}`}>
                            <strong>
                              {entry.name || entry.assembly_id}
                              {entry.status !== 'normal' && ` (${entry.status})`}
                            </strong>
                            <div className="text-sm">
                              <p>Depth: {entry.depth}</p>
                              {entry.created_at && <p>Created: {new Date(entry.created_at).toLocaleString()}</p>}
                              {entry.memory_count && <p>Memories: {entry.memory_count}</p>}
                            </div>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </TabsContent>
            </>
          )}
        </Tabs>
      </CardContent>
    </Card>
  );
}
```

## Testing and Debugging

1. **Feature Flag Testing**: Test the UI with both `ENABLE_EXPLAINABILITY=true` and `ENABLE_EXPLAINABILITY=false` to ensure proper conditional rendering.

2. **API Response Handling**: Use the browser's network inspector to verify API responses and debug any data formatting issues.

3. **Performance Considerations**: Monitor performance when displaying large lineage trees or merge logs. Implement pagination for large datasets.

4. **Error Scenarios**: Test how the UI handles various error scenarios, including 403 errors when the feature flag is disabled, 404 errors for non-existent resources, and 500 errors for server issues.

## Future Enhancements

### Planned for Phase 6.0

1. **Interactive Visualizations**: Enhanced interactive visualizations for lineage trees and assembly relationships.

2. **Real-time Updates**: WebSocket integration for real-time updates to merge logs and activation statistics.

3. **Advanced Filtering**: More sophisticated filtering and search capabilities for diagnostics data.

4. **Export Functionality**: Ability to export diagnostics and explainability data for offline analysis.

## Reference

- [API Reference](../api/API_REFERENCE.md) - Documentation for all API endpoints
- [Error Handling](../api/API_ERRORS.md) - Documentation for API error responses
- [Phase 5.9 Models](../api/phase_5_9_models.md) - Detailed data models for Phase 5.9 API endpoints
