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
  useAlerts
} from "@/lib/api";
import { ServiceStatus } from "@shared/schema";
import { useFeatures } from "@/contexts/FeaturesContext";

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
  const memoryCoreService: ServiceStatus | null = memoryCoreHealth.data?.data ? {
    name: "Memory Core",
    status: memoryCoreHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    uptime: memoryCoreHealth.data.data.uptime || "Unknown",
    version: memoryCoreHealth.data.data.version || "Unknown"
  } : null;
  
  const memoryCoreMetrics = memoryCoreStats.data?.data ? {
    "Total Memories": memoryCoreStats.data.data.core_stats.total_memories.toLocaleString(),
    "Total Assemblies": memoryCoreStats.data.data.core_stats.total_assemblies.toLocaleString()
  } : null;
  
  // Prepare data for Neural Memory status card
  const neuralMemoryService: ServiceStatus | null = neuralMemoryHealth.data?.data ? {
    name: "Neural Memory",
    status: neuralMemoryHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/neural-memory/health",
    uptime: neuralMemoryHealth.data.data.uptime || "Unknown",
    version: neuralMemoryHealth.data.data.version || "Unknown"
  } : null;
  
  const neuralMemoryMetrics = neuralMemoryDiagnostics.data?.data ? {
    "Avg. Loss": neuralMemoryDiagnostics.data.data.avg_loss.toFixed(4),
    "Grad Norm": neuralMemoryDiagnostics.data.data.avg_grad_norm.toFixed(4)
  } : null;
  
  // Prepare data for CCE status card
  const cceService: ServiceStatus | null = cceHealth.data?.data ? {
    name: "Context Cascade Engine",
    status: cceHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    uptime: cceHealth.data.data.uptime || "Unknown",
    version: cceHealth.data.data.version || "Unknown"
  } : null;
  
  const cceMetrics = recentCCEResponses.data?.data?.recent_responses?.length ? {
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
  
  return (
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
  );
}
