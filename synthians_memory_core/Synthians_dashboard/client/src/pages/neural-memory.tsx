import React, { useState } from "react";
import { useNeuralMemoryHealth, useNeuralMemoryStatus, useNeuralMemoryDiagnostics } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { MetricsChart } from "@/components/dashboard/MetricsChart";
import { usePollingStore } from "@/lib/store";
import { ServiceStatus as ServiceStatusType } from "@shared/schema";
import { useFeatures } from "@/contexts/FeaturesContext";
import { formatDuration } from "@/lib/utils";

export default function NeuralMemory() {
  const { refreshAllData } = usePollingStore();
  const [timeWindow, setTimeWindow] = useState("12h");
  const { explainabilityEnabled } = useFeatures();
  
  // Fetch Neural Memory data
  const neuralMemoryHealth = useNeuralMemoryHealth();
  const neuralMemoryStatus = useNeuralMemoryStatus();
  const neuralMemoryDiagnostics = useNeuralMemoryDiagnostics(timeWindow);
  
  // Prepare service status object safely
  const serviceStatus: ServiceStatusType | null = neuralMemoryHealth.data?.success 
    ? {
        name: "Neural Memory",
        status: ["ok", "healthy"].includes((neuralMemoryHealth.data.data?.status || "").toLowerCase()) ? "Healthy" : "Unhealthy",
        url: "/api/neural-memory/health",
        uptime: neuralMemoryHealth.data.data?.uptime || (neuralMemoryHealth.data.data?.uptime_seconds ? formatDuration(neuralMemoryHealth.data.data.uptime_seconds) : "Unknown"),
        version: neuralMemoryHealth.data.data?.version || "Unknown"
      } 
    : null;
  
  // Prepare chart data with robust null checking
  const prepareChartData = () => {
    // Add checks for data structure before accessing history
    // **CRITICAL:** Access nested data safely
    const history = neuralMemoryDiagnostics.data?.data?.history;
    if (!history || !Array.isArray(history)) { 
      // Reduce console noise, only log if it's unexpected (e.g., not loading)
      if (!neuralMemoryDiagnostics.isLoading) {
        console.warn("[prepareChartData] History data is missing or not an array.");
      }
      return []; // Return empty array if history is not available or invalid
    }
    
    return history.map((item: any) => ({
      timestamp: item.timestamp,
      // Add nullish coalescing for safety
      loss: item.loss ?? null, // Keep null if missing, chart can handle gaps
      grad_norm: item.grad_norm ?? null,
      qr_boost: item.qr_boost ?? null,
    })).filter(item => item.timestamp); // Ensure timestamp exists
  };
  
  const chartData = prepareChartData();
  
  // Determine if any metrics are in warning/critical state - safe access
  const isGradNormHigh = 
    (neuralMemoryDiagnostics.data?.data?.avg_grad_norm ?? 0) > 0.8;
    
  // Check for loading and error states
  const isLoading = neuralMemoryHealth.isLoading || neuralMemoryStatus.isLoading || neuralMemoryDiagnostics.isLoading;
  const hasError = neuralMemoryHealth.isError || neuralMemoryStatus.isError || neuralMemoryDiagnostics.isError;
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Neural Memory Dashboard</h2>
          <p className="text-sm text-gray-400">
            Detailed monitoring of the <code className="text-primary">NeuralMemoryModule</code>
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} isLoading={isLoading} />
      </div>
      
      {/* Status Card */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between">
            <CardTitle>Service Status</CardTitle>
            {neuralMemoryHealth.isLoading ? (
              <Skeleton className="h-5 w-20" />
            ) : neuralMemoryHealth.isError ? (
              <Badge variant="destructive">
                <i className="fas fa-exclamation-circle mr-1"></i>
                Error
              </Badge>
            ) : serviceStatus ? (
              <ServiceStatus service={serviceStatus} />
            ) : (
              <Badge variant="destructive">
                <i className="fas fa-times-circle mr-1"></i>
                Unreachable
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {neuralMemoryHealth.isError ? (
            <Alert variant="destructive" className="mb-4">
              <AlertTitle>Failed to connect to Neural Memory</AlertTitle>
              <AlertDescription>
                {neuralMemoryHealth.error?.message || "Unable to fetch service health information. Please verify the Neural Memory service is running."}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
              <div>
                <p className="text-sm text-gray-500 mb-1">Connection</p>
                {neuralMemoryHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus ? (
                  <p className="text-lg">{serviceStatus.url}</p>
                ) : (
                  <p className="text-red-500">Unreachable</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Uptime</p>
                {neuralMemoryHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus?.uptime ? (
                  <p className="text-lg">{serviceStatus.uptime}</p>
                ) : (
                  <p className="text-gray-400">Unknown</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">TensorFlow Version</p>
                {neuralMemoryHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : neuralMemoryHealth.data?.data?.tensorflow_version ? (
                  <p className="text-lg">{neuralMemoryHealth.data.data.tensorflow_version}</p>
                ) : (
                  <p className="text-gray-400">Unknown</p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Main Tabs */}
      <Tabs defaultValue="diagnostics" className="mb-6">
        <TabsList className="mb-4">
          <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
          <TabsTrigger value="configuration">Configuration</TabsTrigger>
          {explainabilityEnabled && (
            <TabsTrigger value="metrics">Detailed Metrics</TabsTrigger>
          )}
        </TabsList>
        
        <TabsContent value="diagnostics" className="mt-4">
          {neuralMemoryDiagnostics.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load Neural Memory diagnostics</AlertTitle>
              <AlertDescription>
                {neuralMemoryDiagnostics.error?.message || "There was an error fetching diagnostic data. Please try again."}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Performance Metrics */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  {neuralMemoryDiagnostics.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : neuralMemoryDiagnostics.data?.data ? ( // **CRITICAL:** Check if data.data exists
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-1">
                          <p className="text-sm text-gray-500">Average Loss</p>
                          <p className="text-sm font-mono">
                            {(neuralMemoryDiagnostics.data.data.avg_loss ?? 0).toFixed(5)}
                          </p>
                        </div>
                        <Progress 
                          value={Math.min((neuralMemoryDiagnostics.data.data.avg_loss ?? 0) * 100, 100)} 
                          className="h-2" 
                        />
                      </div>
                      <div>
                        <div className="flex justify-between mb-1">
                          <p className="text-sm text-gray-500">Gradient Norm</p>
                          <p className={`text-sm font-mono ${isGradNormHigh ? "text-amber-500" : ""}`}>
                            {(neuralMemoryDiagnostics.data.data.avg_grad_norm ?? 0).toFixed(5)}
                            {isGradNormHigh && <span className="ml-2 text-amber-500">⚠</span>}
                          </p>
                        </div>
                        <Progress 
                          value={Math.min((neuralMemoryDiagnostics.data.data.avg_grad_norm ?? 0) * 100, 100)} 
                          className={isGradNormHigh ? "h-2 bg-amber-900/20" : "h-2"} 
                        />
                      </div>
                      <div>
                        <div className="flex justify-between mb-1">
                          <p className="text-sm text-gray-500">QR Boost</p>
                          <p className="text-sm font-mono">
                            {(neuralMemoryDiagnostics.data.data.avg_qr_boost ?? 0).toFixed(5)}
                          </p>
                        </div>
                        <Progress 
                          value={Math.min((neuralMemoryDiagnostics.data.data.avg_qr_boost ?? 0) * 100, 100)} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-400 text-center py-8">No diagnostic data available</p> // State when data is missing/empty
                  )}
                </CardContent>
              </Card>
              
              {/* Emotional Loop */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Emotional Loop</CardTitle>
                </CardHeader>
                <CardContent>
                  {neuralMemoryDiagnostics.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : neuralMemoryDiagnostics.data?.data?.emotional_loop ? ( // Check nested loop data
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm text-gray-500 mb-1">Dominant Emotions</p>
                        <div className="flex flex-wrap gap-1">
                          {Array.isArray(neuralMemoryDiagnostics.data.data.emotional_loop.dominant_emotions) ? 
                           neuralMemoryDiagnostics.data.data.emotional_loop.dominant_emotions.map((emotion: string, index: number) => (
                            <Badge key={index} variant="secondary">{emotion}</Badge>
                          )) : (
                            <span className="text-xs text-gray-400">N/A</span>
                          )}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500 mb-1">Entropy</p>
                        <p className="text-lg">
                          {(neuralMemoryDiagnostics.data.data.emotional_loop.entropy ?? 0).toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500 mb-1">Bias Index</p>
                        <p className="text-lg">
                          {(neuralMemoryDiagnostics.data.data.emotional_loop.bias_index ?? 0).toFixed(3)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500 mb-1">Match Rate</p>
                        <div className="flex justify-between mb-1">
                          <span></span>
                          <p className="text-sm font-mono">
                            {((neuralMemoryDiagnostics.data.data.emotional_loop.match_rate ?? 0) * 100).toFixed(1)}%
                          </p>
                        </div>
                        <Progress 
                          value={(neuralMemoryDiagnostics.data.data.emotional_loop.match_rate ?? 0) * 100} 
                          className="h-2" 
                        />
                      </div>
                    </div>
                  ) : (
                    <p className="text-gray-400 text-center py-8">No emotional loop data available</p>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="configuration" className="mt-4">
          {neuralMemoryStatus.isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          ) : neuralMemoryStatus.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load configuration</AlertTitle>
              <AlertDescription>
                {neuralMemoryStatus.error?.message || "An error occurred while fetching Neural Memory configuration."}
              </AlertDescription>
            </Alert>
          ) : neuralMemoryStatus.data ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Initialization Status</p>
                <p className="text-lg">
                  {neuralMemoryStatus.data.data?.initialized ? (
                    <span className="text-green-400">Initialized</span>
                  ) : (
                    <span className="text-yellow-400">Not Initialized</span>
                  )}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Dimensions</p>
                <p className="text-lg font-mono">
                  {neuralMemoryStatus.data.data?.config?.dimensions || "Unknown"}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Hidden Size</p>
                <p className="text-lg font-mono">
                  {neuralMemoryStatus.data.data?.config?.hidden_size || "Unknown"}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-4 text-gray-400">
              <p>No configuration data available</p>
            </div>
          )}
        </TabsContent>
        
        {explainabilityEnabled && (
          <TabsContent value="metrics" className="mt-4">
            {neuralMemoryDiagnostics.isLoading ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <Skeleton className="h-8 w-full" />
                    <Skeleton className="h-40 w-full" />
                    <Skeleton className="h-8 w-full" />
                  </div>
                </CardContent>
              </Card>
            ) : neuralMemoryDiagnostics.isError ? (
              <Alert variant="destructive">
                <AlertTitle>Failed to load metrics data</AlertTitle>
                <AlertDescription>
                  {neuralMemoryDiagnostics.error?.message || "An error occurred while fetching Neural Memory metrics data."}
                </AlertDescription>
              </Alert>
            ) : neuralMemoryDiagnostics.data?.data ? (
              <div className="grid grid-cols-1 gap-6">
                <MetricsChart
                  title="Neural Memory Performance"
                  data={chartData} // Already prepared safely
                  dataKeys={[
                    { key: "loss", color: "#FF008C", name: "Loss" },
                    { key: "grad_norm", color: "#1EE4FF", name: "Gradient Norm" },
                    { key: "qr_boost", color: "#FF3EE8", name: "QR Boost" }
                  ]}
                  isLoading={neuralMemoryDiagnostics.isLoading}
                  isError={neuralMemoryDiagnostics.isError}
                  error={neuralMemoryDiagnostics.error}
                  timeRange={timeWindow}
                  onTimeRangeChange={setTimeWindow}
                  summary={[
                    { 
                      label: "Avg. Loss", 
                      value: (neuralMemoryDiagnostics.data.data.avg_loss ?? 0).toFixed(4), 
                      color: "text-primary" 
                    },
                    { 
                      label: "Avg. Grad Norm", 
                      value: (neuralMemoryDiagnostics.data.data.avg_grad_norm ?? 0).toFixed(4),
                      color: isGradNormHigh ? "text-destructive" : "text-secondary"
                    },
                    { 
                      label: "Avg. QR Boost", 
                      value: (neuralMemoryDiagnostics.data.data.avg_qr_boost ?? 0).toFixed(4),
                      color: "text-primary" 
                    }
                  ]}
                />
                
                {/* Recommendations section */}
                {Array.isArray(neuralMemoryDiagnostics.data?.data?.recommendations) && 
                 neuralMemoryDiagnostics.data.data.recommendations.length > 0 && (
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle>System Recommendations</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="list-disc list-inside space-y-2">
                        {neuralMemoryDiagnostics.data.data.recommendations.map((rec: string, idx: number) => (
                          <li key={idx} className="text-sm">{rec || "Invalid recommendation entry"}</li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
                
                {/* Alerts section */}
                {Array.isArray(neuralMemoryDiagnostics.data?.data?.alerts) && 
                 neuralMemoryDiagnostics.data.data.alerts.length > 0 && (
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle>System Alerts</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {neuralMemoryDiagnostics.data.data.alerts.map((alert: string, idx: number) => (
                          <Alert key={idx} variant="destructive">
                            <AlertDescription>{alert || "Invalid alert entry"}</AlertDescription>
                          </Alert>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <p>No metrics data available</p> // Message when data.data is missing/empty
              </div>
            )}
          </TabsContent>
        )}
      </Tabs>
    </>
  );
}
