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
    status: cceHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/cce/health",
    uptime: cceHealth.data.data.uptime || "Unknown",
    version: cceHealth.data.data.version || "Unknown"
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

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">CCE Dashboard</h2>
          <p className="text-sm text-gray-400">
            Monitoring the <code className="text-primary">Context Cascade Engine</code> and variant selection
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
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
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
              <div>
                <p className="text-sm text-gray-500 mb-1">Connection</p>
                {cceHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus ? (
                  <p className="text-lg">{serviceStatus.url}</p>
                ) : (
                  <p className="text-red-500">Unreachable</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Uptime</p>
                {cceHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus?.uptime ? (
                  <p className="text-lg">{serviceStatus.uptime}</p>
                ) : (
                  <p className="text-gray-400">Unknown</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Active Variant</p>
                {recentCCEResponses.isLoading || cceHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : recentCCEResponses.isError ? (
                  <p className="text-red-500">Error</p>
                ) : (
                  <p className="text-lg font-mono text-secondary">{activeVariant}</p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Tabs for different views */}
      <Tabs defaultValue="variants" className="mb-6">
        <TabsList>
          <TabsTrigger value="variants" onClick={() => setActiveTab("variants")}>Variant Selection</TabsTrigger>
          {explainabilityEnabled && (
            <TabsTrigger value="llm" onClick={() => setActiveTab("llm")}>LLM Guidance</TabsTrigger>
          )}
          {explainabilityEnabled && (
            <TabsTrigger value="errors" onClick={() => setActiveTab("errors")}>Errors</TabsTrigger>
          )}
        </TabsList>
        
        <TabsContent value="variants" className="mt-4">
          <div className="grid grid-cols-1 gap-6">
            <CCEChart
              title="Variant Distribution (Last 12 Hours)"
              data={recentCCEResponses.data?.data?.recent_responses || []}
              isLoading={recentCCEResponses.isLoading}
              isError={recentCCEResponses.isError}
              error={recentCCEResponses.error}
            />
            
            <Card>
              <CardHeader>
                <CardTitle>Recent Variant Selections</CardTitle>
              </CardHeader>
              <CardContent>
                {recentCCEResponses.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-32 w-full" />
                  </div>
                ) : recentCCEResponses.isError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Failed to load variant selection data</AlertTitle>
                    <AlertDescription>
                      {recentCCEResponses.error?.message || "An error occurred while fetching CCE response data."}
                    </AlertDescription>
                  </Alert>
                ) : variantSelections.length > 0 ? (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-[120px]">Timestamp</TableHead>
                          <TableHead>Selected Variant</TableHead>
                          <TableHead>Reason</TableHead>
                          <TableHead className="text-center">Perf. Used</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {variantSelections.map((response: CCEResponse, index: number) => (
                          <TableRow key={index}>
                            <TableCell className="font-mono text-xs">
                              {formatTime(response.timestamp)}
                            </TableCell>
                            <TableCell>
                              <Badge className="bg-muted text-secondary">
                                {response.variant_selection?.selected_variant}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-sm">
                              {response.variant_selection?.reason || "N/A"}
                            </TableCell>
                            <TableCell className="text-center">
                              {response.variant_selection?.performance_used ? (
                                <i className="fas fa-check text-green-400"></i>
                              ) : (
                                <i className="fas fa-times text-gray-500"></i>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <p className="text-gray-400 text-center py-4">No variant selection data available</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {explainabilityEnabled && (
          <TabsContent value="llm" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>LLM Guidance Usage</CardTitle>
              </CardHeader>
              <CardContent>
                {recentCCEResponses.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-64 w-full" />
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
                    {llmGuidanceResponses.map((response: CCEResponse, index: number) => (
                      <div key={index} className="border border-border rounded-lg p-4">
                        <div className="flex justify-between mb-3">
                          <span className="text-xs text-gray-400">
                            {new Date(response.timestamp).toLocaleString()}
                          </span>
                          <Badge variant="outline" className="text-primary border-primary">
                            Confidence: {response.llm_advice_used?.confidence_level.toFixed(2)}
                          </Badge>
                        </div>
                        
                        <h4 className="text-sm font-medium mb-2">Adjusted Advice</h4>
                        <div className="bg-muted p-3 rounded text-sm mb-4 font-mono">
                          {response.llm_advice_used?.adjusted_advice || "N/A"}
                        </div>
                        
                        {response.llm_advice_used?.raw_advice && (
                          <>
                            <h4 className="text-sm font-medium mb-2">Raw LLM Advice</h4>
                            <div className="bg-muted p-3 rounded text-sm mb-4 font-mono text-xs overflow-auto max-h-32">
                              {response.llm_advice_used.raw_advice}
                            </div>
                          </>
                        )}
                        
                        {response.llm_advice_used?.adjustment_reason && (
                          <div className="text-xs text-gray-400">
                            <span className="text-secondary">Adjustment Reason:</span> {response.llm_advice_used.adjustment_reason}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-400 text-center py-4">No LLM guidance data available</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        )}
        
        {explainabilityEnabled && (
          <TabsContent value="errors" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle>Recent Errors</CardTitle>
              </CardHeader>
              <CardContent>
                {recentCCEResponses.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-32 w-full" />
                  </div>
                ) : recentCCEResponses.isError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Failed to load error data</AlertTitle>
                    <AlertDescription>
                      {recentCCEResponses.error?.message || "An error occurred while fetching error data."}
                    </AlertDescription>
                  </Alert>
                ) : errorResponses.length > 0 ? (
                  <div className="space-y-4">
                    {errorResponses.map((response: CCEResponse, index: number) => (
                      <div key={index} className="border border-border rounded-lg p-4 bg-red-900/10">
                        <div className="flex items-start">
                          <i className="fas fa-exclamation-circle text-destructive mr-3 mt-1"></i>
                          <div>
                            <div className="flex items-center mb-2">
                              <h4 className="text-sm font-medium mr-2">Error at {formatTime(response.timestamp)}</h4>
                              <Badge variant="destructive">Error</Badge>
                            </div>
                            <p className="text-sm text-gray-300 mb-2">{response.error_details}</p>
                            
                            {response.variant_selection && (
                              <div className="text-xs text-gray-400">
                                <span>Attempted variant: </span>
                                <span className="text-primary">{response.variant_selection.selected_variant}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <i className="fas fa-check-circle text-green-400 text-2xl mb-2"></i>
                    <p className="text-gray-400">No errors detected in recent responses</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </>
  );
}
