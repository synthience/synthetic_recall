import React, { useState } from "react";
import { OverviewCard } from "@/components/dashboard/OverviewCard";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { SystemArchitecture } from "@/components/dashboard/SystemArchitecture";
import { DiagnosticAlerts } from "@/components/dashboard/DiagnosticAlerts";
import { CCEChart } from "@/components/dashboard/CCEChart";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
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
import { ServiceStatus } from "@shared/schema";
import { useFeatures } from "@/contexts/FeaturesContext";
import { formatDuration } from "@/lib/utils"; // Import the new function

export default function Overview() {
  const [timeRange, setTimeRange] = useState<string>("12h");
  const { explainabilityEnabled } = useFeatures();
  
  // Fetch all the required data
  const memoryCoreHealth = useMemoryCoreHealth();
  const neuralMemoryHealth = useNeuralMemoryHealth();
  const cceHealth = useCCEHealth();
  const memoryCoreStats = useMemoryCoreStats();
  const assemblies = useAssemblies();
  const neuralMemoryDiagnostics = useNeuralMemoryDiagnostics(timeRange);
  const recentCCEResponses = useRecentCCEResponses();
  const alerts = useAlerts();
  
  // Prepare data for Memory Core status card
  const memoryCoreService: ServiceStatus | null = memoryCoreHealth.data?.success && memoryCoreHealth.data.data ? {
    name: "Memory Core",
    status: ["ok", "healthy"].includes(memoryCoreHealth.data.data.status?.toLowerCase()) ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    // Handle both uptime formats (string or number)
    uptime: memoryCoreHealth.data.data.uptime || 
           (memoryCoreHealth.data.data.uptime_seconds ? formatDuration(memoryCoreHealth.data.data.uptime_seconds) : "Unknown"),
    version: memoryCoreHealth.data.data.version || "Unknown"
  } : null;
  
  // Adjust Memory Core metrics access if needed (assuming stats endpoint *does* follow {success, data} structure)
  const memoryCoreMetrics = memoryCoreStats.data?.success && memoryCoreStats.data.data ? {
    "Total Memories": memoryCoreStats.data.data.core_stats.total_memories?.toLocaleString() ?? 'N/A',
    "Total Assemblies": memoryCoreStats.data.data.core_stats.total_assemblies?.toLocaleString() ?? 'N/A'
  } : null;
  
  // Prepare data for Neural Memory status card
  const neuralMemoryService: ServiceStatus | null = neuralMemoryHealth.data?.success && neuralMemoryHealth.data.data ? {
    name: "Neural Memory",
    status: ["ok", "healthy"].includes(neuralMemoryHealth.data.data.status?.toLowerCase()) ? "Healthy" : "Unhealthy",
    url: "/api/neural-memory/health",
    // Handle both uptime formats (Neural Memory returns string, Memory Core returns number)
    uptime: neuralMemoryHealth.data.data.uptime || 
           (neuralMemoryHealth.data.data.uptime_seconds ? formatDuration(neuralMemoryHealth.data.data.uptime_seconds) : "Unknown"),
    version: neuralMemoryHealth.data.data.version || "Unknown"
  } : null;
  
  // Adjust Neural Memory metrics access (assuming diagnostics endpoint follows {success, data} structure)
  const neuralMemoryMetrics = neuralMemoryDiagnostics.data?.success && neuralMemoryDiagnostics.data.data ? {
    "Avg. Loss": neuralMemoryDiagnostics.data.data.avg_loss?.toFixed(4) ?? '--',
    "Grad Norm": neuralMemoryDiagnostics.data.data.avg_grad_norm?.toFixed(4) ?? '--'
  } : null;
  
  // Prepare data for CCE status card
  // Note: CCE health endpoint might return different structure based on logs (e.g., 'detail')
  // We prioritize the schema structure first.
  const cceService: ServiceStatus | null = cceHealth.data?.success && cceHealth.data.data ? {
    name: "Context Cascade Engine",
    status: ["ok", "healthy"].includes(cceHealth.data.data.status?.toLowerCase()) ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    // Handle both uptime formats (string or number)
    uptime: cceHealth.data.data.uptime || 
           (cceHealth.data.data.uptime_seconds ? formatDuration(cceHealth.data.data.uptime_seconds) : "Unknown"),
    version: cceHealth.data.data.version || "Unknown",
    details: cceHealth.data.data.error || undefined // Use error field if status is not healthy
  } : null;
  
  // Adjust CCE metrics access (assuming recent_cce_responses follows {success, data} structure)
  const cceMetrics = recentCCEResponses.data?.success && recentCCEResponses.data.data?.recent_responses?.length ? {
    "Active Variant": recentCCEResponses.data.data.recent_responses[0]?.variant_selection?.selected_variant || "Unknown"
  } : null;
  
  // Prepare data for Neural Memory chart
  const prepareNeuralMemoryChartData = () => {
    if (!neuralMemoryDiagnostics.data?.data?.history) {
      return [];
    }
    
    return neuralMemoryDiagnostics.data.data.history.map((item) => ({
      timestamp: item.timestamp,
      loss: item.loss,
      grad_norm: item.grad_norm
    }));
  };
  
  const neuralMemoryChartData = prepareNeuralMemoryChartData();
  
  // Function to calculate min/max values from history data
  const calculateMinMaxLoss = () => {
    if (!neuralMemoryDiagnostics.data?.data?.history || neuralMemoryDiagnostics.data.data.history.length === 0) {
      return { min: "--", max: "--" };
    }
    
    const lossValues = neuralMemoryDiagnostics.data.data.history.map(item => item.loss);
    const min = Math.min(...lossValues).toFixed(4);
    const max = Math.max(...lossValues).toFixed(4);
    
    return { min, max };
  };
  
  const { min: minLoss, max: maxLoss } = calculateMinMaxLoss();
  
  // Prepare assemblies data
  const recentAssemblies = assemblies.data?.data || [];
  
  // Check for service-wide errors
  const hasServiceErrors = memoryCoreHealth.isError || neuralMemoryHealth.isError || cceHealth.isError;
  
  // Loading and error handling
  const isLoading = memoryCoreHealth.isLoading || neuralMemoryHealth.isLoading || cceHealth.isLoading || 
                  memoryCoreStats.isLoading || assemblies.isLoading || neuralMemoryDiagnostics.isLoading || 
                  recentCCEResponses.isLoading || alerts.isLoading;

  const hasError = memoryCoreHealth.isError || neuralMemoryHealth.isError || cceHealth.isError || 
                 memoryCoreStats.isError || assemblies.isError || neuralMemoryDiagnostics.isError || 
                 recentCCEResponses.isError || alerts.isError;

  const errorMessages = [];
  if (memoryCoreHealth.error) errorMessages.push(`Memory Core health check failed: ${memoryCoreHealth.error.message}`);
  if (neuralMemoryHealth.error) errorMessages.push(`Neural Memory health check failed: ${neuralMemoryHealth.error.message}`);
  if (cceHealth.error) errorMessages.push(`CCE health check failed: ${cceHealth.error.message}`);
  if (memoryCoreStats.error) errorMessages.push(`Memory Core stats retrieval failed: ${memoryCoreStats.error.message}`);
  if (assemblies.error) errorMessages.push(`Assemblies retrieval failed: ${assemblies.error.message}`);
  if (neuralMemoryDiagnostics.error) errorMessages.push(`Neural Memory diagnostics retrieval failed: ${neuralMemoryDiagnostics.error.message}`);
  if (recentCCEResponses.error) errorMessages.push(`Recent CCE responses retrieval failed: ${recentCCEResponses.error.message}`);
  if (alerts.error) errorMessages.push(`Alerts retrieval failed: ${alerts.error.message}`);

  return (
    <div className="container mx-auto py-4 space-y-6">
      <h1 className="text-2xl font-bold mb-4">Synthians Cognitive Dashboard</h1>
      
      {hasError && (
        <Alert variant="destructive" className="mb-4">
          <AlertTitle>Error retrieving dashboard data</AlertTitle>
          <AlertDescription>
            <ul className="list-disc pl-4">
              {errorMessages.map((msg, idx) => (
                <li key={idx}>{msg}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}
      
      {isLoading && (
        <div className="text-lg text-gray-500 mb-4">Loading...</div>
      )}
      
      {!isLoading && !hasError && (
        <>
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-white mb-1">System Overview</h2>
            <p className="text-sm text-gray-400">At-a-glance status of all core services</p>
          </div>

          {/* Service-wide error alert */}
          {hasServiceErrors && (
            <Alert variant="destructive" className="mb-6">
              <AlertTitle>Service Health Check Failed</AlertTitle>
              <AlertDescription>
                One or more services are experiencing connectivity issues. Check network connectivity or service logs.
              </AlertDescription>
            </Alert>
          )}

          {/* Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <OverviewCard
              title="Memory Core"
              icon="database"
              service={memoryCoreService}
              metrics={memoryCoreMetrics}
              isLoading={memoryCoreHealth.isLoading || memoryCoreStats.isLoading}
              isError={memoryCoreHealth.isError || memoryCoreStats.isError}
              error={memoryCoreHealth.error || memoryCoreStats.error}
            />
            
            <OverviewCard
              title="Neural Memory"
              icon="brain"
              service={neuralMemoryService}
              metrics={neuralMemoryMetrics}
              isLoading={neuralMemoryHealth.isLoading || neuralMemoryDiagnostics.isLoading}
              isError={neuralMemoryHealth.isError || neuralMemoryDiagnostics.isError}
              error={neuralMemoryHealth.error || neuralMemoryDiagnostics.error}
            />
            
            <OverviewCard
              title="Context Cascade Engine"
              icon="sitemap"
              service={cceService}
              metrics={cceMetrics}
              isLoading={cceHealth.isLoading || recentCCEResponses.isLoading}
              isError={cceHealth.isError || recentCCEResponses.isError}
              error={cceHealth.error || recentCCEResponses.error}
            />
          </div>

          {/* Performance Metrics */}
          <div className="mb-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <MetricsChart
                title="Neural Memory - Training Loss"
                data={neuralMemoryChartData}
                dataKeys={[
                  { key: "loss", color: "#FF008C", name: "Avg. Loss" },
                  { key: "grad_norm", color: "#1EE4FF", name: "Grad Norm" }
                ]}
                isLoading={neuralMemoryDiagnostics.isLoading}
                isError={neuralMemoryDiagnostics.isError}
                error={neuralMemoryDiagnostics.error}
                timeRange={timeRange}
                onTimeRangeChange={setTimeRange}
                summary={[
                  { label: "Current", value: neuralMemoryDiagnostics.data?.data?.avg_loss?.toFixed(4) || "--", color: "text-primary" },
                  { label: "Min", value: minLoss, color: "text-secondary" },
                  { label: "Max", value: maxLoss, color: "text-destructive" }
                ]}
              />
              
              <CCEChart
                title="CCE - Variant Selection"
                data={recentCCEResponses.data?.data?.recent_responses || []}
                isLoading={recentCCEResponses.isLoading}
                isError={recentCCEResponses.isError}
                error={recentCCEResponses.error}
              />
            </div>
          </div>
          
          {/* Assemblies and Diagnostics */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <AssemblyTable
              title="Last Updated Assemblies"
              assemblies={recentAssemblies.slice(0, 5)}
              isLoading={assemblies.isLoading}
              isError={assemblies.isError}
              error={assemblies.error}
              showFilters={false}
            />
            
            {explainabilityEnabled && (
              <DiagnosticAlerts
                alerts={alerts.data?.data || []}
                isLoading={alerts.isLoading}
                isError={alerts.isError}
                error={alerts.error}
              />
            )}
          </div>
          
          {/* System Architecture */}
          {explainabilityEnabled && (
            <div className="mb-8">
              <SystemArchitecture />
            </div>
          )}
        </>
      )}
    </div>
  );
}
