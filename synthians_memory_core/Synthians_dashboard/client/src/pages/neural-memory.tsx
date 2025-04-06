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

export default function NeuralMemory() {
  const { refreshAllData } = usePollingStore();
  const [timeWindow, setTimeWindow] = useState("12h");
  const { explainabilityEnabled } = useFeatures();
  
  // Fetch Neural Memory data
  const neuralMemoryHealth = useNeuralMemoryHealth();
  const neuralMemoryStatus = useNeuralMemoryStatus();
  const neuralMemoryDiagnostics = useNeuralMemoryDiagnostics(timeWindow);
  
  // Prepare service status object
  const serviceStatus = neuralMemoryHealth.data?.data ? {
    name: "Neural Memory",
    status: neuralMemoryHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/neural-memory/health",
    uptime: neuralMemoryHealth.data.data.uptime || "Unknown",
    version: neuralMemoryHealth.data.data.version || "Unknown"
  } as ServiceStatusType : null;
  
  // Prepare chart data
  const prepareChartData = () => {
    if (!neuralMemoryDiagnostics.data?.data?.history) {
      return [];
    }
    
    return neuralMemoryDiagnostics.data.data.history.map((item: any) => ({
      timestamp: item.timestamp,
      loss: item.loss,
      grad_norm: item.grad_norm,
      qr_boost: item.qr_boost
    }));
  };
  
  const chartData = prepareChartData();
  
  // Determine if any metrics are in warning/critical state
  const isGradNormHigh = 
    (neuralMemoryDiagnostics.data?.data?.avg_grad_norm ?? 0) > 0.8;
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Neural Memory Dashboard</h2>
          <p className="text-sm text-gray-400">
            Detailed monitoring of the <code className="text-primary">NeuralMemoryModule</code>
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
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
                <p className="text-sm text-gray-500 mb-1">Version</p>
                {neuralMemoryHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus?.version ? (
                  <p className="text-lg">{serviceStatus.version}</p>
                ) : (
                  <p className="text-gray-400">Unknown</p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Configuration Overview */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Neural Memory Configuration</CardTitle>
        </CardHeader>
        <CardContent>
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
                  {neuralMemoryStatus.data.initialized ? (
                    <span className="text-green-400">Initialized</span>
                  ) : (
                    <span className="text-yellow-400">Not Initialized</span>
                  )}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Dimensions</p>
                <p className="text-lg font-mono">
                  {neuralMemoryStatus.data.config?.dimensions || "Unknown"}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Hidden Size</p>
                <p className="text-lg font-mono">
                  {neuralMemoryStatus.data.config?.hidden_size || "Unknown"}
                </p>
              </div>
            </div>
          ) : (
            <div className="text-center py-4 text-gray-400">
              <p>No configuration data available</p>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Warning if high grad norm */}
      {isGradNormHigh && neuralMemoryDiagnostics.data?.data && !neuralMemoryDiagnostics.isError && (
        <Alert variant="destructive" className="mb-6">
          <AlertTitle className="flex items-center">
            <i className="fas fa-exclamation-circle mr-2"></i>
            High Gradient Norm Detected
          </AlertTitle>
          <AlertDescription>
            The gradient norm of {neuralMemoryDiagnostics.data.data.avg_grad_norm.toFixed(4)} exceeds the recommended threshold of 0.7500.
          </AlertDescription>
        </Alert>
      )}
      
      {/* Tabs for Performance Metrics */}
      <Tabs defaultValue="performance" className="mb-6">
        <TabsList>
          <TabsTrigger value="performance">Performance Metrics</TabsTrigger>
          <TabsTrigger value="emotional">Emotional Loop</TabsTrigger>
          {explainabilityEnabled && (
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          )}
        </TabsList>
        
        <TabsContent value="performance" className="mt-4">
          <div className="grid grid-cols-1 gap-6">
            <MetricsChart
              title="Neural Memory Performance"
              data={chartData}
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
                  value: neuralMemoryDiagnostics.data?.data?.avg_loss?.toFixed(4) || "--", 
                  color: "text-primary" 
                },
                { 
                  label: "Avg. Grad Norm", 
                  value: neuralMemoryDiagnostics.data?.data?.avg_grad_norm?.toFixed(4) || "--",
                  color: isGradNormHigh ? "text-destructive" : "text-secondary"
                },
                { 
                  label: "Avg. QR Boost", 
                  value: neuralMemoryDiagnostics.data?.data?.avg_qr_boost?.toFixed(4) || "--",
                  color: "text-primary" 
                }
              ]}
            />
          </div>
        </TabsContent>
        
        <TabsContent value="emotional" className="mt-4">
          {neuralMemoryDiagnostics.isLoading ? (
            <Card>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-24 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              </CardContent>
            </Card>
          ) : neuralMemoryDiagnostics.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load emotional loop data</AlertTitle>
              <AlertDescription>
                {neuralMemoryDiagnostics.error?.message || "An error occurred while fetching emotional loop diagnostics."}
              </AlertDescription>
            </Alert>
          ) : neuralMemoryDiagnostics.data?.data?.emotional_loop ? (
            <Card>
              <CardContent className="pt-6">
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Emotional Entropy</p>
                      <p className="text-lg font-mono">
                        {neuralMemoryDiagnostics.data.data.emotional_loop.entropy.toFixed(4)}
                      </p>
                      <Progress 
                        value={neuralMemoryDiagnostics.data.data.emotional_loop.entropy * 100} 
                        className="h-1.5 mt-2" 
                      />
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Bias Index</p>
                      <p className="text-lg font-mono">
                        {neuralMemoryDiagnostics.data.data.emotional_loop.bias_index.toFixed(4)}
                      </p>
                      <Progress 
                        value={neuralMemoryDiagnostics.data.data.emotional_loop.bias_index * 100} 
                        className="h-1.5 mt-2" 
                      />
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Match Rate</p>
                      <p className="text-lg font-mono">
                        {(neuralMemoryDiagnostics.data.data.emotional_loop.match_rate * 100).toFixed(2)}%
                      </p>
                      <Progress 
                        value={neuralMemoryDiagnostics.data.data.emotional_loop.match_rate * 100} 
                        className="h-1.5 mt-2" 
                      />
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-sm text-gray-500 mb-1">Dominant Emotions</p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {neuralMemoryDiagnostics.data.data.emotional_loop.dominant_emotions.map((emotion: string, idx: number) => (
                        <Badge key={idx} variant="outline" className="bg-primary/5">
                          {emotion}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="text-center py-8 text-gray-400">
              <p>No emotional loop data available</p>
            </div>
          )}
        </TabsContent>
        
        {explainabilityEnabled && (
          <TabsContent value="recommendations" className="mt-4">
            {neuralMemoryDiagnostics.isLoading ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <Skeleton className="h-8 w-full" />
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                </CardContent>
              </Card>
            ) : neuralMemoryDiagnostics.isError ? (
              <Alert variant="destructive">
                <AlertTitle>Failed to load recommendations</AlertTitle>
                <AlertDescription>
                  {neuralMemoryDiagnostics.error?.message || "An error occurred while fetching Neural Memory recommendations."}
                </AlertDescription>
              </Alert>
            ) : neuralMemoryDiagnostics.data?.data?.recommendations && neuralMemoryDiagnostics.data.data.recommendations.length > 0 ? (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    {neuralMemoryDiagnostics.data.data.recommendations.map((recommendation: string, idx: number) => (
                      <Alert key={idx} className="bg-primary/5 border-primary/20">
                        <div className="flex">
                          <i className="fas fa-lightbulb text-secondary mt-1 mr-2"></i>
                          <AlertDescription className="text-primary-foreground">
                            {recommendation}
                          </AlertDescription>
                        </div>
                      </Alert>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <p>No recommendations available</p>
              </div>
            )}
          </TabsContent>
        )}
      </Tabs>
    </>
  );
}
