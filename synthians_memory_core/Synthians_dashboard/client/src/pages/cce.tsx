import React, { useState } from "react";
import { useCCEHealth, useCCEStatus, useRecentCCEResponses } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { ServiceStatus as ServiceStatusComponent } from "@/components/layout/ServiceStatus";
import { CCEChart } from "@/components/dashboard/CCEChart";
import { usePollingStore } from "@/lib/store";
import { ServiceStatus, CCEResponse } from "@shared/schema";
import { useFeatures } from "@/contexts/FeaturesContext";
import { formatDuration, formatTimeAgo } from "@/lib/utils";

export default function CCE() {
  const { refreshAllData } = usePollingStore();
  const [activeTab, setActiveTab] = useState("overview");
  const { explainabilityEnabled } = useFeatures();
  
  // Fetch CCE data
  const cceHealth = useCCEHealth();
  const cceStatus = useCCEStatus();
  const recentCCEResponses = useRecentCCEResponses();
  
  // Prepare service status object
  const serviceStatus: ServiceStatus | null = cceHealth.data?.data ? {
    name: "Context Cascade Engine",
    // Access status from nested data property and use robust check
    status: ["ok", "healthy"].includes((cceHealth.data?.data?.status || "").toLowerCase()) ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    // Handle both uptime formats (string or number)
    uptime: cceHealth.data?.data?.uptime || 
           (cceHealth.data?.data?.uptime_seconds ? formatDuration(cceHealth.data?.data?.uptime_seconds) : "Unknown"),
    version: cceHealth.data?.data?.version || "Unknown"
  } : null;
  
  // Get active variant from the most recent response
  const activeVariant = recentCCEResponses.data?.data?.recent_responses?.[0]?.variant_output?.variant_type || "Unknown";
  
  // Filter recent responses with errors
  const errorResponses = recentCCEResponses.data?.data?.recent_responses?.filter(
    (response: CCEResponse) => response.status === "error"
  ) || [];
  
  // Get variant selections for display
  const variantSelections = recentCCEResponses.data?.data?.recent_responses?.filter(
    (response: CCEResponse) => response.variant_selection
  ).slice(0, 10) || [];
  
  // Get responses with LLM guidance
  const llmGuidanceResponses = recentCCEResponses.data?.data?.recent_responses?.filter(
    (response: CCEResponse) => response.llm_advice_used
  ).slice(0, 5) || [];
  
  // Format timestamp
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };
  
  // Check for loading and error states
  const isLoading = cceHealth.isLoading || cceStatus.isLoading || recentCCEResponses.isLoading;
  const hasError = cceHealth.isError || cceStatus.isError || recentCCEResponses.isError;

  // Safely access memory stats
  const memoryUsage = cceStatus.data?.data?.memory_stats?.used_mb;
  const avgResponseTime = recentCCEResponses.data?.data?.avg_response_time_ms;
  const recentResponses = recentCCEResponses.data?.data?.recent_responses || [];
  const responsesCount = recentResponses.length;

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">CCE Dashboard</h2>
          <p className="text-sm text-gray-400">
            Monitoring the <code className="text-primary">Context Cascade Engine</code> and variant selection
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} isLoading={isLoading} />
      </div>
      
      {/* Status Card */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between">
            <CardTitle>Service Status</CardTitle>
            {cceHealth.isLoading ? (
              <Skeleton className="h-5 w-20" />
            ) : cceHealth.isError ? (
              <Badge variant="destructive">
                <i className="fas fa-exclamation-circle mr-1"></i>
                Error
              </Badge>
            ) : serviceStatus ? (
              <ServiceStatusComponent service={serviceStatus} />
            ) : (
              <Badge variant="destructive">
                <i className="fas fa-times-circle mr-1"></i>
                Unreachable
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {cceHealth.isError ? (
            <Alert variant="destructive" className="mb-4">
              <AlertTitle>Failed to connect to Context Cascade Engine</AlertTitle>
              <AlertDescription>
                {cceHealth.error?.message || "Unable to fetch service health information. Please verify the CCE service is running."}
              </AlertDescription>
            </Alert>
          ) : cceHealth.isLoading ? (
            <div className="space-y-4">
              <div className="flex justify-between">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-32" />
              </div>
              <div className="flex justify-between">
                <Skeleton className="h-4 w-28" />
                <Skeleton className="h-4 w-40" />
              </div>
              <div className="flex justify-between">
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-36" />
              </div>
            </div>
          ) : serviceStatus ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Version</span>
                <span className="text-sm font-mono">{serviceStatus.version}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Uptime</span>
                <span className="text-sm font-mono">{serviceStatus.uptime}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-500">Active Variant</span>
                <span className="text-sm font-mono">
                  {recentCCEResponses.isLoading ? (
                    <Skeleton className="h-4 w-20 inline-block" />
                  ) : (
                    activeVariant
                  )}
                </span>
              </div>
              {memoryUsage !== undefined && (
                <div className="flex justify-between">
                  <span className="text-sm text-gray-500">Memory Usage</span>
                  <span className="text-sm font-mono">
                    {memoryUsage.toFixed(1)} MB
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-4 text-gray-400">
              <p>No status information available</p>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-6">
        <TabsList>
          <TabsTrigger value="overview">
            Overview
          </TabsTrigger>
          <TabsTrigger value="responses">
            Recent Responses
          </TabsTrigger>
          {explainabilityEnabled && (
            <TabsTrigger value="diagnostics">
              Diagnostics
            </TabsTrigger>
          )}
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Latest CCE Activity</CardTitle>
              </CardHeader>
              <CardContent>
                {recentCCEResponses.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                ) : recentCCEResponses.isError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Failed to load recent responses</AlertTitle>
                    <AlertDescription>
                      {recentCCEResponses.error?.message || "An error occurred while fetching recent CCE responses."}
                    </AlertDescription>
                  </Alert>
                ) : responsesCount > 0 ? (
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Latest Response</p>
                      <div className="bg-card border rounded-md p-3">
                        <div className="flex justify-between mb-2">
                          <Badge variant="outline">
                            {recentResponses[0].variant_output?.variant_type || "Unknown"}
                          </Badge>
                          <span className="text-xs text-gray-500">
                            {formatTime(recentResponses[0].timestamp)}
                          </span>
                        </div>
                        <p className="text-sm truncate">
                          {recentResponses[0].input?.substring(0, 60)}...
                        </p>
                        {recentResponses[0].status === "error" && (
                          <Alert variant="destructive" className="mt-2">
                            <AlertDescription>
                              {recentResponses[0].error || "Unknown error"}
                            </AlertDescription>
                          </Alert>
                        )}
                      </div>
                    </div>

                    <div>
                      <p className="text-sm text-gray-500 mb-1">Errors ({errorResponses.length})</p>
                      {errorResponses.length > 0 ? (
                        <div className="text-sm text-destructive">
                          {errorResponses.length} error(s) in the last {responsesCount} responses
                        </div>
                      ) : (
                        <div className="text-sm text-primary">No errors detected</div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-4 text-gray-400">
                    <p>No recent responses available</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Variant Usage</CardTitle>
              </CardHeader>
              <CardContent>
                {recentCCEResponses.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-32 w-full" />
                    <Skeleton className="h-4 w-2/3" />
                  </div>
                ) : recentCCEResponses.isError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Failed to load variant data</AlertTitle>
                    <AlertDescription>
                      {recentCCEResponses.error?.message || "An error occurred while fetching variant usage data."}
                    </AlertDescription>
                  </Alert>
                ) : responsesCount > 0 ? (
                  <div>
                    <div className="h-32 mb-4 text-center text-gray-400">
                      <CCEChart 
                        data={variantSelections} 
                        isLoading={recentCCEResponses.isLoading}
                        isError={recentCCEResponses.isError}
                        error={recentCCEResponses.error}
                        title="Variant Distribution"
                      />
                    </div>
                    <div className="text-sm text-gray-500">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <p className="text-xs">Most Used Variant:</p>
                          <p className="font-mono">
                            {/* Logic to determine most used variant */}
                            {activeVariant}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs">Avg. Response Time:</p>
                          <p className="font-mono">
                            {avgResponseTime !== undefined ? 
                              `${avgResponseTime.toFixed(2)} ms` : "N/A"}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-4 text-gray-400">
                    <p>No variant data available</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Responses Tab */}
        <TabsContent value="responses" className="mt-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle>Recent CCE Responses</CardTitle>
            </CardHeader>
            <CardContent>
              {recentCCEResponses.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-32 w-full" />
                </div>
              ) : recentCCEResponses.isError ? (
                <Alert variant="destructive">
                  <AlertTitle>Failed to load recent responses</AlertTitle>
                  <AlertDescription>
                    {recentCCEResponses.error?.message || "An error occurred while fetching recent CCE responses."}
                  </AlertDescription>
                </Alert>
              ) : responsesCount > 0 ? (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Time</TableHead>
                        <TableHead>Input</TableHead>
                        <TableHead>Variant</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {recentResponses.map((response: CCEResponse, index: number) => (
                        <TableRow key={index}>
                          <TableCell className="whitespace-nowrap">
                            {formatTime(response.timestamp)}
                          </TableCell>
                          <TableCell className="max-w-xs truncate">
                            {response.input?.substring(0, 50) || "N/A"}...
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">
                              {response.variant_output?.variant_type || "Unknown"}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {response.status === "error" ? (
                              <Badge variant="destructive">Error</Badge>
                            ) : (
                              <Badge variant="default">Success</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <p>No recent responses available</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Diagnostics Tab - Only show if explainabilityEnabled is true */}
        {explainabilityEnabled && (
          <TabsContent value="diagnostics" className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>LLM Guidance</CardTitle>
                </CardHeader>
                <CardContent>
                  {recentCCEResponses.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-8 w-full" />
                      <Skeleton className="h-40 w-full" />
                    </div>
                  ) : recentCCEResponses.isError ? (
                    <Alert variant="destructive">
                      <AlertTitle>Failed to load LLM guidance data</AlertTitle>
                      <AlertDescription>
                        {recentCCEResponses.error?.message || "An error occurred while fetching LLM guidance data."}
                      </AlertDescription>
                    </Alert>
                  ) : llmGuidanceResponses.length > 0 ? (
                    <div className="space-y-4">
                      {llmGuidanceResponses.map((response: CCEResponse, idx: number) => (
                        <div key={idx} className="border rounded-md p-3">
                          <div className="flex justify-between mb-2">
                            <Badge variant="outline">
                              {response.variant_output?.variant_type || "Unknown"}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {formatTime(response.timestamp)}
                            </span>
                          </div>
                          <p className="text-sm mb-2 font-semibold">LLM Advice:</p>
                          <p className="text-sm bg-primary/5 p-2 rounded">
                            {response.llm_advice || "No specific advice recorded"}
                          </p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      <p>No LLM guidance data available</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Variant Selection History</CardTitle>
                </CardHeader>
                <CardContent>
                  {recentCCEResponses.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-8 w-full" />
                      <Skeleton className="h-40 w-full" />
                    </div>
                  ) : recentCCEResponses.isError ? (
                    <Alert variant="destructive">
                      <AlertTitle>Failed to load variant selection data</AlertTitle>
                      <AlertDescription>
                        {recentCCEResponses.error?.message || "An error occurred while fetching variant selection data."}
                      </AlertDescription>
                    </Alert>
                  ) : variantSelections.length > 0 ? (
                    <div className="space-y-4">
                      {variantSelections.map((response: CCEResponse, idx: number) => (
                        <div key={idx} className="border rounded-md p-3">
                          <div className="flex justify-between mb-2">
                            <Badge variant="outline">
                              {response.variant_output?.variant_type || "Unknown"}
                            </Badge>
                            <span className="text-xs text-gray-500">
                              {formatTime(response.timestamp)}
                            </span>
                          </div>
                          <div className="text-sm">
                            <p className="mb-1"><span className="font-semibold">Input:</span> {response.input?.substring(0, 40)}...</p>
                            <p><span className="font-semibold">Reason:</span> {response.variant_selection?.reason || "No reason provided"}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      <p>No variant selection data available</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        )}
      </Tabs>
    </>
  );
}
