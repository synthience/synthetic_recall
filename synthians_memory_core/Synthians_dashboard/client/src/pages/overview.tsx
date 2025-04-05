import React, { useState } from "react";
import { OverviewCard } from "@/components/dashboard/OverviewCard";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { SystemArchitecture } from "@/components/dashboard/SystemArchitecture";
import { DiagnosticAlerts } from "@/components/dashboard/DiagnosticAlerts";
import { CCEChart } from "@/components/dashboard/CCEChart";
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

export default function Overview() {
  const [timeRange, setTimeRange] = useState<string>("12h");
  
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
  const memoryCoreService = memoryCoreHealth.data?.data ? {
    name: "Memory Core",
    status: memoryCoreHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    uptime: memoryCoreHealth.data.data.uptime || "Unknown",
    version: memoryCoreHealth.data.data.version || "Unknown"
  } : null;
  
  const memoryCoreMetrics = memoryCoreStats.data?.data ? {
    "Total Memories": memoryCoreStats.data.data.total_memories.toLocaleString(),
    "Total Assemblies": memoryCoreStats.data.data.total_assemblies.toLocaleString()
  } : null;
  
  // Prepare data for Neural Memory status card
  const neuralMemoryService = neuralMemoryHealth.data?.data ? {
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
  const cceService = cceHealth.data?.data ? {
    name: "Context Cascade Engine",
    status: cceHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    uptime: cceHealth.data.data.uptime || "Unknown",
    version: cceHealth.data.data.version || "Unknown"
  } : null;
  
  const cceMetrics = recentCCEResponses.data?.data?.recent_responses ? {
    "Active Titan Variant": recentCCEResponses.data.data.recent_responses[0]?.variant_output?.variant_type || "Unknown"
  } : null;
  
  // Prepare data for Neural Memory chart
  const prepareNeuralMemoryChartData = () => {
    const emptyData = Array(12).fill(0).map((_, i) => ({
      timestamp: new Date(Date.now() - i * 3600 * 1000).toISOString(),
      loss: Math.random() * 0.05 + 0.02, // Placeholder values when no real data
      grad_norm: Math.random() * 0.2 + 0.7
    }));
    
    if (!neuralMemoryDiagnostics.data?.data?.history) {
      return emptyData;
    }
    
    return neuralMemoryDiagnostics.data.data.history.map((item: any) => ({
      timestamp: item.timestamp,
      loss: item.loss,
      grad_norm: item.grad_norm
    }));
  };
  
  const neuralMemoryChartData = prepareNeuralMemoryChartData();
  
  // Prepare assemblies data
  const recentAssemblies = assemblies.data?.data?.slice(0, 5) || null;
  
  return (
    <>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white mb-1">System Overview</h2>
        <p className="text-sm text-gray-400">At-a-glance status of all core services</p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <OverviewCard
          title="Memory Core"
          icon="database"
          service={memoryCoreService}
          metrics={memoryCoreMetrics}
          isLoading={memoryCoreHealth.isLoading || memoryCoreStats.isLoading}
        />
        
        <OverviewCard
          title="Neural Memory"
          icon="brain"
          service={neuralMemoryService}
          metrics={neuralMemoryMetrics}
          isLoading={neuralMemoryHealth.isLoading || neuralMemoryDiagnostics.isLoading}
        />
        
        <OverviewCard
          title="Context Cascade Engine"
          icon="sitemap"
          service={cceService}
          metrics={cceMetrics}
          isLoading={cceHealth.isLoading || recentCCEResponses.isLoading}
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
            timeRange={timeRange}
            onTimeRangeChange={setTimeRange}
            summary={[
              { label: "Current", value: neuralMemoryDiagnostics.data?.data?.avg_loss.toFixed(4) || "--", color: "text-primary" },
              { label: "Min (12h)", value: "0.0341", color: "text-secondary" },
              { label: "Max (12h)", value: "0.0729", color: "text-yellow-400" }
            ]}
          />
          
          <CCEChart
            title="CCE - Variant Selection"
            data={recentCCEResponses.data?.data?.recent_responses || []}
            isLoading={recentCCEResponses.isLoading}
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="mb-8">
        <AssemblyTable
          assemblies={recentAssemblies}
          isLoading={assemblies.isLoading}
          title="Last Updated Assemblies"
        />
      </div>

      {/* System Architecture */}
      <div className="mb-8">
        <SystemArchitecture />
      </div>

      {/* Diagnostic Alerts */}
      <DiagnosticAlerts
        alerts={alerts.data?.data || null}
        isLoading={alerts.isLoading}
      />
    </>
  );
}
