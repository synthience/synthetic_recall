import React, { useState } from "react";
import { OverviewCard } from "@/components/dashboard/OverviewCard";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { SystemArchitecture } from "@/components/dashboard/SystemArchitecture";
import { DiagnosticAlerts } from "@/components/dashboard/DiagnosticAlerts";
import { CCEChart } from "@/components/dashboard/CCEChart";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  useMemoryCoreHealth,
  useNeuralMemoryHealth,
  useCCEHealth,
  useMemoryCoreStats,
  useAssemblies,
  useNeuralMemoryDiagnostics,
  useRecentCCEResponses,
  useAlerts,
} from "@/lib/api";
import { ServiceStatus, ApiResponse } from "@shared/schema"; 
import { useFeatures } from "@/contexts/FeaturesContext";
import { formatDuration } from "@/lib/utils";

export default function Overview() {
  const [timeRange, setTimeRange] = useState<string>("12h");
  const { explainabilityEnabled, isLoading: featuresAreLoading } = useFeatures();
  
  // Fetch all the required data
  const memoryCoreHealth = useMemoryCoreHealth({ enabled: !featuresAreLoading });
  const neuralMemoryHealth = useNeuralMemoryHealth({ enabled: !featuresAreLoading });
  const cceHealth = useCCEHealth({ enabled: !featuresAreLoading });
  const memoryCoreStats = useMemoryCoreStats({ enabled: !featuresAreLoading });
  const assemblies = useAssemblies({ enabled: !featuresAreLoading });
  const neuralMemoryDiagnostics = useNeuralMemoryDiagnostics(timeRange, { enabled: !featuresAreLoading });
  const recentCCEResponses = useRecentCCEResponses({ enabled: !featuresAreLoading });
  const alerts = useAlerts({ enabled: !featuresAreLoading });
  
  // Prepare data for Memory Core status card - properly unwrap nested data with optional chaining
  const memoryCoreService: ServiceStatus | null = memoryCoreHealth.data?.success && memoryCoreHealth.data.data ? {
    name: "Memory Core",
    status: ["ok", "healthy"].includes((memoryCoreHealth.data.data.status || "").toLowerCase()) 
           ? "Healthy" 
           : "Unhealthy",
    url: "/api/memory-core/health",
    uptime: memoryCoreHealth.data.data.uptime || 
           (memoryCoreHealth.data.data.uptime_seconds ? formatDuration(memoryCoreHealth.data.data.uptime_seconds) : "Unknown"),
    version: memoryCoreHealth.data.data.version || "Unknown"
  } : null;
  
  // Memory Core metrics with specific focus on pending updates - carefully handle nested properties with optional chaining
  const memoryCoreMetrics = memoryCoreStats.data?.success && memoryCoreStats.data.data ? {
    "Memories (Cache)": memoryCoreStats.data.data.core_stats?.memory_count_cache?.toLocaleString() ?? 'N/A',
    "Assemblies (Cache)": memoryCoreStats.data.data.core_stats?.assembly_count_cache?.toLocaleString() ?? 'N/A',
    "Pending Updates": memoryCoreStats.data.data.core_stats?.pending_vector_updates?.toLocaleString() ?? 'N/A',
  } : null;
  
  // Prepare data for Neural Memory status card - properly unwrap nested data with optional chaining
  const neuralMemoryService: ServiceStatus | null = neuralMemoryHealth.data?.success && neuralMemoryHealth.data.data ? {
    name: "Neural Memory",
    status: ["ok", "healthy"].includes((neuralMemoryHealth.data.data.status || "").toLowerCase()) 
           ? "Healthy" 
           : "Unhealthy",
    url: "/api/neural-memory/health",
    uptime: neuralMemoryHealth.data.data.uptime || 
           (neuralMemoryHealth.data.data.uptime_seconds ? formatDuration(neuralMemoryHealth.data.data.uptime_seconds) : "Unknown"),
    version: neuralMemoryHealth.data.data.version || "Unknown"
  } : null;
  
  // Neural Memory metrics focusing on diagnostics and metrics - carefully handle nested properties with optional chaining
  const neuralMemoryMetrics = neuralMemoryDiagnostics.data?.success && neuralMemoryDiagnostics.data.data ? {
    "Loss": (neuralMemoryDiagnostics.data.data.avg_loss ?? 0).toFixed(4),
    "Grad Norm": (neuralMemoryDiagnostics.data.data.avg_grad_norm ?? 0).toFixed(4),
    "Emotional Entropy": (neuralMemoryDiagnostics.data.data.emotional_loop?.entropy ?? 0).toFixed(2),
  } : null;
  
  // Prepare data for CCE status card - properly unwrap nested data with optional chaining
  const cceService: ServiceStatus | null = cceHealth.data?.success && cceHealth.data.data ? {
    name: "Context Cascade Engine",
    status: ["ok", "healthy"].includes((cceHealth.data.data.status || "").toLowerCase()) 
           ? "Healthy" 
           : "Unhealthy",
    url: "/api/cce/health",
    uptime: cceHealth.data.data.uptime || 
           (cceHealth.data.data.uptime_seconds ? formatDuration(cceHealth.data.data.uptime_seconds) : "Unknown"),
    version: cceHealth.data.data.version || "Unknown",
    details: cceHealth.data.data.status === "processing" ? "Currently processing requests" : undefined
  } : null;
  
  // CCE metrics - carefully handle nested properties with optional chaining
  const cceMetrics = recentCCEResponses.data?.success && recentCCEResponses.data.data?.recent_responses?.length ? {
    "Active Variant": recentCCEResponses.data.data.recent_responses[0]?.variant_selection?.selected_variant || "Unknown"
  } : null;
  
  // Helper function to get effective error state and message
  const getEffectiveError = (queryResult: { data?: ApiResponse<any>; isLoading: boolean; isError: boolean; error: Error | null }) => {
    const apiError = queryResult.data ? !queryResult.data.success : false;
    const apiErrorMessage = queryResult.data && apiError ? (queryResult.data.error || queryResult.data.message) : null;
    const queryErrorMessage = queryResult.isError ? queryResult.error?.message : null;
    
    return {
      isEffectivelyError: apiError || queryResult.isError,
      effectiveErrorMessage: apiErrorMessage || queryErrorMessage || (apiError || queryResult.isError ? 'Unknown error' : null),
    };
  };

  // Calculate effective errors for core services
  const memoryCoreHealthError = getEffectiveError(memoryCoreHealth);
  const neuralMemoryHealthError = getEffectiveError(neuralMemoryHealth);
  const cceHealthError = getEffectiveError(cceHealth);
  const memoryCoreStatsError = getEffectiveError(memoryCoreStats);  
  const neuralMemoryDiagnosticsError = getEffectiveError(neuralMemoryDiagnostics);
  const recentCCEResponsesError = getEffectiveError(recentCCEResponses);
  const assembliesError = getEffectiveError(assemblies);

  return (
    <div className="container mx-auto py-4 space-y-6">
      <h1 className="text-2xl font-bold mb-4">Synthians Cognitive Dashboard</h1>
      
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white mb-1">System Overview</h2>
        <p className="text-sm text-gray-400">At-a-glance status of all core services</p>
      </div>

      {/* Service Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <OverviewCard
          title="Memory Core"
          icon="database"
          service={memoryCoreService}
          metrics={memoryCoreMetrics}
          isLoading={memoryCoreHealth.isLoading || memoryCoreStats.isLoading}
          isError={memoryCoreHealthError.isEffectivelyError || memoryCoreStatsError.isEffectivelyError}
          error={memoryCoreHealth.error || memoryCoreStats.error}
          errorMessage={memoryCoreHealthError.effectiveErrorMessage || memoryCoreStatsError.effectiveErrorMessage}
        />
        
        <OverviewCard
          title="Neural Memory"
          icon="brain"
          service={neuralMemoryService}
          metrics={neuralMemoryMetrics}
          isLoading={neuralMemoryHealth.isLoading || neuralMemoryDiagnostics.isLoading}
          isError={neuralMemoryHealthError.isEffectivelyError || neuralMemoryDiagnosticsError.isEffectivelyError}
          error={neuralMemoryHealth.error || neuralMemoryDiagnostics.error}
          errorMessage={neuralMemoryHealthError.effectiveErrorMessage || neuralMemoryDiagnosticsError.effectiveErrorMessage}
        />
        
        <OverviewCard
          title="Context Cascade Engine"
          icon="cogs"
          service={cceService}
          metrics={cceMetrics}
          isLoading={cceHealth.isLoading || recentCCEResponses.isLoading}
          isError={cceHealthError.isEffectivelyError || recentCCEResponsesError.isEffectivelyError}
          error={cceHealth.error || recentCCEResponses.error}
          errorMessage={cceHealthError.effectiveErrorMessage || recentCCEResponsesError.effectiveErrorMessage}
        />
      </div>
      
      {/* System Architecture */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-white mb-4">System Architecture</h2>
        <SystemArchitecture />
      </div>
      
      {/* Recent Assemblies */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-white mb-4">Recent Assemblies</h2>
        {assemblies.isLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : assembliesError.isEffectivelyError ? (
          <Alert variant="destructive">
            <AlertTitle>Failed to load assemblies</AlertTitle>
            <AlertDescription>
              {assembliesError.effectiveErrorMessage || "Could not retrieve assemblies data."}
            </AlertDescription>
          </Alert>
        ) : (!assemblies.data?.data || assemblies.data.data.length === 0) ? (
          <Alert>
            <AlertTitle>No assemblies found</AlertTitle>
            <AlertDescription>
              There are currently no memory assemblies in the system.
            </AlertDescription>
          </Alert>
        ) : (
          <AssemblyTable 
            assemblies={assemblies.data?.data || []} 
            isLoading={assemblies.isLoading}
            isError={assembliesError.isEffectivelyError}
            errorMessage={assembliesError.effectiveErrorMessage}
          />
        )}
      </div>

      {/* Only show diagnostics sections if explainability is enabled */}
      {explainabilityEnabled && (
        <>
          {/* Diagnostic Data */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
              <h2 className="text-xl font-semibold text-white mb-4">Neural Memory Training</h2>
              {neuralMemoryDiagnostics.isLoading ? (
                <Skeleton className="h-64 w-full" />
              ) : neuralMemoryDiagnosticsError.isEffectivelyError ? (
                <Alert variant="destructive">
                  <AlertTitle>Failed to load neural memory diagnostics</AlertTitle>
                  <AlertDescription>
                    {neuralMemoryDiagnosticsError.effectiveErrorMessage || "Could not retrieve neural memory diagnostic data."}
                  </AlertDescription>
                </Alert>
              ) : (
                <MetricsChart 
                  title="Neural Memory Metrics"
                  data={neuralMemoryDiagnostics.data?.data?.history || []}
                  dataKeys={[
                    { key: 'loss', color: '#4f46e5', name: 'Loss' },
                    { key: 'grad_norm', color: '#10b981', name: 'Gradient Norm' }
                  ]}
                  isLoading={neuralMemoryDiagnostics.isLoading}
                  timeRange={timeRange}
                  onTimeRangeChange={setTimeRange}
                />
              )}
            </div>
            
            <div>
              <h2 className="text-xl font-semibold text-white mb-4">CCE Variant Usage</h2>
              {recentCCEResponses.isLoading ? (
                <Skeleton className="h-64 w-full" />
              ) : recentCCEResponsesError.isEffectivelyError ? (
                <Alert variant="destructive">
                  <AlertTitle>Failed to load CCE response data</AlertTitle>
                  <AlertDescription>
                    {recentCCEResponsesError.effectiveErrorMessage || "Could not retrieve CCE variant data."}
                  </AlertDescription>
                </Alert>
              ) : (
                <CCEChart 
                  data={recentCCEResponses.data?.data?.recent_responses || []}
                  isLoading={recentCCEResponses.isLoading}
                  title="CCE Variant Usage"
                />
              )}
            </div>
          </div>
          
          {/* Diagnostic Alerts */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-white mb-4">System Alerts</h2>
            {alerts.isLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : getEffectiveError(alerts).isEffectivelyError ? (
              <Alert variant="destructive">
                <AlertTitle>Failed to load alerts</AlertTitle>
                <AlertDescription>
                  {getEffectiveError(alerts).effectiveErrorMessage || "Could not retrieve system alerts."}
                </AlertDescription>
              </Alert>
            ) : (
              <DiagnosticAlerts 
                alerts={alerts.data?.data || []}
                isLoading={alerts.isLoading}  
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
