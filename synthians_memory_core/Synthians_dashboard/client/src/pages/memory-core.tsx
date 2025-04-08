import React, { useState } from "react";
import { useMemoryCoreHealth, useMemoryCoreStats, useAssemblies } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { usePollingStore } from "@/lib/store";
import { ServiceStatus as ServiceStatusType } from "@shared/schema";
import { formatDuration, formatTimeAgo } from "@/lib/utils";

export default function MemoryCore() {
  const { refreshAllData } = usePollingStore();
  const [activeTab, setActiveTab] = useState("overview");
  
  // Fetch Memory Core data
  const memoryCoreHealth = useMemoryCoreHealth();
  const memoryCoreStats = useMemoryCoreStats();
  const assemblies = useAssemblies();
  
  // Prepare service status object
  const serviceStatus = memoryCoreHealth.data?.success && memoryCoreHealth.data.data ? {
    name: "Memory Core",
    // Access status from nested data property and use robust check
    status: ["ok", "healthy"].includes((memoryCoreHealth.data.data.status || "").toLowerCase()) ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    // Handle both uptime formats (string or number)
    uptime: memoryCoreHealth.data.data.uptime || 
           (memoryCoreHealth.data.data.uptime_seconds ? formatDuration(memoryCoreHealth.data.data.uptime_seconds) : "Unknown"),
    version: memoryCoreHealth.data.data.version || "Unknown"
  } as ServiceStatusType : null;
  
  // Calculate warning thresholds for vector index drift
  const isDriftAboveWarning = (memoryCoreStats.data?.data?.vector_index_stats?.drift_count ?? 0) > 50;
  const isDriftAboveCritical = (memoryCoreStats.data?.data?.vector_index_stats?.drift_count ?? 0) > 100;
  
  // Check for loading and error states
  const isLoading = memoryCoreHealth.isLoading || memoryCoreStats.isLoading || assemblies.isLoading;
  const hasError = memoryCoreHealth.isError || memoryCoreStats.isError || assemblies.isError;
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Memory Core Dashboard</h2>
          <p className="text-sm text-gray-400">
            Detailed monitoring of the <code className="text-primary">SynthiansMemoryCore</code>
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} isLoading={isLoading} />
      </div>
      
      {/* Status Card */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <div className="flex justify-between">
            <CardTitle>Service Status</CardTitle>
            {memoryCoreHealth.isLoading ? (
              <Skeleton className="h-5 w-20" />
            ) : memoryCoreHealth.isError ? (
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
          {memoryCoreHealth.isError ? (
            <Alert variant="destructive" className="mb-4">
              <AlertTitle>Failed to connect to Memory Core</AlertTitle>
              <AlertDescription>
                {memoryCoreHealth.error?.message || "Unable to fetch service health information. Please verify the Memory Core service is running."}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
              <div>
                <p className="text-sm text-gray-500 mb-1">Connection</p>
                {memoryCoreHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus ? (
                  <p className="text-lg">{serviceStatus.url}</p>
                ) : (
                  <p className="text-red-500">Unreachable</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Uptime</p>
                {memoryCoreHealth.isLoading ? (
                  <Skeleton className="h-5 w-32" />
                ) : serviceStatus?.uptime ? (
                  <p className="text-lg">{serviceStatus.uptime}</p>
                ) : (
                  <p className="text-gray-400">Unknown</p>
                )}
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-1">Version</p>
                {memoryCoreHealth.isLoading ? (
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
      
      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-6">
        <TabsList className="mb-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="vector-index">Vector Index</TabsTrigger>
          <TabsTrigger value="assemblies">Assemblies</TabsTrigger>
          <TabsTrigger value="persistence">Persistence</TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview">
          {memoryCoreStats.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load Memory Core statistics</AlertTitle>
              <AlertDescription>
                {memoryCoreStats.error?.message || "There was an error fetching Memory Core stats. Please try again."}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Memory Stats</CardTitle>
                </CardHeader>
                <CardContent>
                  {memoryCoreStats.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : (
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="font-medium">Total Memories</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.core_stats?.total_memories?.toLocaleString() ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Total Assemblies</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.core_stats?.total_assemblies?.toLocaleString() ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Dirty Memories</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.core_stats?.dirty_memories?.toLocaleString() ?? '0'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Pending Vector Updates</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.core_stats?.pending_vector_updates?.toLocaleString() ?? '0'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Core Initialized</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.core_stats?.initialized ? (
                              <Badge variant="secondary">Yes</Badge>
                            ) : (
                              <Badge variant="destructive">No</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Assembly Stats</CardTitle>
                </CardHeader>
                <CardContent>
                  {memoryCoreStats.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : (
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="font-medium">Total Count</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.assemblies?.total_count?.toLocaleString() ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Indexed Count</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.assemblies?.indexed_count?.toLocaleString() ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Vector Indexed Count</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.assemblies?.vector_indexed_count?.toLocaleString() ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Average Size</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.assemblies?.average_size?.toFixed(2) ?? 'N/A'} memories
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Pruning Enabled</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.assemblies?.pruning_enabled ? (
                              <Badge variant="secondary">Yes</Badge>
                            ) : (
                              <Badge variant="outline">No</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Merging Enabled</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.assemblies?.merging_enabled ? (
                              <Badge variant="secondary">Yes</Badge>
                            ) : (
                              <Badge variant="outline">No</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="vector-index" className="mt-4">
          {memoryCoreStats.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load Vector Index statistics</AlertTitle>
              <AlertDescription>
                {memoryCoreStats.error?.message || "There was an error fetching Vector Index data. Please try again."}
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle>Vector Index Status</CardTitle>
                </CardHeader>
                <CardContent>
                  {memoryCoreStats.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : (
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="font-medium">Total Vectors</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.vector_index_stats?.total_vectors?.toLocaleString() ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Index Size</TableCell>
                          <TableCell className="text-right">
                            {typeof memoryCoreStats.data?.data?.vector_index_stats?.index_size_mb === 'number' ? 
                              `${memoryCoreStats.data.data.vector_index_stats.index_size_mb.toFixed(2)} MB` : 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Vector Dimensions</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.vector_index_stats?.vector_dimensions ?? 'N/A'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Drift Count</TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-2">
                              {memoryCoreStats.data?.data?.vector_index_stats?.drift_count?.toLocaleString() ?? '0'}
                              {isDriftAboveWarning && !isDriftAboveCritical && (
                                <Badge variant="secondary">WARNING</Badge>
                              )}
                              {isDriftAboveCritical && (
                                <Badge variant="destructive">CRITICAL</Badge>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Index Health</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.vector_index_stats?.healthy ? (
                              <Badge variant="secondary">Healthy</Badge>
                            ) : (
                              <Badge variant="destructive">Unhealthy</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Pending Updates</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.vector_index_stats?.pending_updates?.toLocaleString() ?? '0'}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Last Update</TableCell>
                          <TableCell className="text-right">
                            {memoryCoreStats.data?.data?.vector_index_stats?.last_update_at ? 
                              formatTimeAgo(memoryCoreStats.data.data.vector_index_stats.last_update_at) : 'Never'}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle>Performance</CardTitle>
                </CardHeader>
                <CardContent>
                  {memoryCoreStats.isLoading ? (
                    <div className="space-y-4">
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                      <Skeleton className="h-12 w-full" />
                    </div>
                  ) : (
                    <div className="space-y-6">
                      {memoryCoreStats.data?.data?.quick_recall_stats && (
                        <div>
                          <div className="flex justify-between mb-1">
                            <p className="text-sm text-gray-500">Quick Recall Rate</p>
                            <p className="text-sm font-mono">
                              {((memoryCoreStats.data.data.quick_recall_stats.recall_rate ?? 0) * 100).toFixed(2)}%
                            </p>
                          </div>
                          <Progress 
                            value={(memoryCoreStats.data.data.quick_recall_stats.recall_rate ?? 0) * 100} 
                            className="h-2" 
                          />
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Avg. Latency: {memoryCoreStats.data.data.quick_recall_stats?.avg_latency_ms?.toFixed(2) ?? 'N/A'} ms</span>
                            <span>Count: {memoryCoreStats.data.data.quick_recall_stats?.count ?? 0}</span>
                          </div>
                        </div>
                      )}
                      
                      {memoryCoreStats.data?.data?.threshold_stats && (
                        <div>
                          <div className="flex justify-between mb-1">
                            <p className="text-sm text-gray-500">Threshold Recall Rate</p>
                            <p className="text-sm font-mono">
                              {((memoryCoreStats.data.data.threshold_stats.recall_rate ?? 0) * 100).toFixed(2)}%
                            </p>
                          </div>
                          <Progress 
                            value={(memoryCoreStats.data.data.threshold_stats.recall_rate ?? 0) * 100} 
                            className="h-2" 
                          />
                          <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Avg. Latency: {memoryCoreStats.data.data.threshold_stats?.avg_latency_ms?.toFixed(2) ?? 'N/A'} ms</span>
                            <span>Count: {memoryCoreStats.data.data.threshold_stats?.count ?? 0}</span>
                          </div>
                        </div>
                      )}
                      
                      {!memoryCoreStats.data?.data?.quick_recall_stats && !memoryCoreStats.data?.data?.threshold_stats && (
                        <p className="text-gray-400 text-center py-4">No performance data available</p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="assemblies" className="mt-4">
          {assemblies.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load assemblies</AlertTitle>
              <AlertDescription>
                {assemblies.error?.message || "There was an error fetching assembly data. Please try again."}
              </AlertDescription>
            </Alert>
          ) : (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Memory Assemblies</CardTitle>
              </CardHeader>
              <CardContent>
                <AssemblyTable
                  assemblies={assemblies.data?.data || []}
                  isLoading={assemblies.isLoading}
                  isError={assemblies.isError}
                  error={assemblies.error}
                  showFilters={true}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="persistence" className="mt-4">
          {memoryCoreStats.isError ? (
            <Alert variant="destructive">
              <AlertTitle>Failed to load persistence statistics</AlertTitle>
              <AlertDescription>
                {memoryCoreStats.error?.message || "There was an error fetching persistence data. Please try again."}
              </AlertDescription>
            </Alert>
          ) : (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle>Persistence Status</CardTitle>
              </CardHeader>
              <CardContent>
                {memoryCoreStats.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                ) : (
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell className="font-medium">Last Update</TableCell>
                        <TableCell className="text-right">
                          {memoryCoreStats.data?.data?.persistence_stats?.last_update ? 
                            formatTimeAgo(memoryCoreStats.data.data.persistence_stats.last_update) : 
                            'Never'}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Last Backup</TableCell>
                        <TableCell className="text-right">
                          {memoryCoreStats.data?.data?.persistence_stats?.last_backup ? 
                            formatTimeAgo(memoryCoreStats.data.data.persistence_stats.last_backup) : 
                            'Never'}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell className="font-medium">Database Status</TableCell>
                        <TableCell className="text-right">
                          {memoryCoreHealth.data?.data?.status?.toLowerCase() === 'healthy' ? (
                            <Badge variant="secondary">Healthy</Badge>
                          ) : (
                            <Badge variant="destructive">Problem Detected</Badge>
                          )}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </>
  );
}
