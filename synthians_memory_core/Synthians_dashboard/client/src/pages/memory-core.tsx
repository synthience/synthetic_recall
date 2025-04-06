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

export default function MemoryCore() {
  const { refreshAllData } = usePollingStore();
  const [activeTab, setActiveTab] = useState("overview");
  
  // Fetch Memory Core data
  const memoryCoreHealth = useMemoryCoreHealth();
  const memoryCoreStats = useMemoryCoreStats();
  const assemblies = useAssemblies();
  
  // Prepare service status object
  const serviceStatus = memoryCoreHealth.data?.data ? {
    name: "Memory Core",
    status: memoryCoreHealth.data.data.status === "ok" ? "Healthy" : "Unhealthy",
    url: "/api/memory-core/health",
    uptime: memoryCoreHealth.data.data.uptime || "Unknown",
    version: memoryCoreHealth.data.data.version || "Unknown"
  } as ServiceStatusType : null;
  
  // Calculate warning thresholds for vector index drift
  const isDriftAboveWarning = (memoryCoreStats.data?.data?.vector_index_stats?.drift_count ?? 0) > 50;
  const isDriftAboveCritical = (memoryCoreStats.data?.data?.vector_index_stats?.drift_count ?? 0) > 100;
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Memory Core Dashboard</h2>
          <p className="text-sm text-gray-400">
            Detailed monitoring of the <code className="text-primary">SynthiansMemoryCore</code>
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
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
      
      {/* Tabs for different views */}
      <Tabs defaultValue="overview" className="mb-6">
        <TabsList>
          <TabsTrigger value="overview" onClick={() => setActiveTab("overview")}>Overview</TabsTrigger>
          <TabsTrigger value="vector-index" onClick={() => setActiveTab("vector-index")}>Vector Index</TabsTrigger>
          <TabsTrigger value="assemblies" onClick={() => setActiveTab("assemblies")}>Assemblies</TabsTrigger>
          <TabsTrigger value="persistence" onClick={() => setActiveTab("persistence")}>Persistence</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="col-span-2">
              <CardHeader>
                <CardTitle>Core Stats</CardTitle>
              </CardHeader>
              <CardContent>
                {memoryCoreStats.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-16 w-full" />
                    <Skeleton className="h-16 w-full" />
                  </div>
                ) : memoryCoreStats.isError ? (
                  <Alert variant="destructive">
                    <AlertTitle>Failed to load statistics</AlertTitle>
                    <AlertDescription>
                      {memoryCoreStats.error?.message || "An error occurred while fetching Memory Core statistics."}
                    </AlertDescription>
                  </Alert>
                ) : memoryCoreStats.data?.data?.core_stats ? (
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="mb-4">
                        <p className="text-sm text-gray-500">Total Memories</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.core_stats.total_memories.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Dirty Memories</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.core_stats.dirty_memories.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div>
                      <div className="mb-4">
                        <p className="text-sm text-gray-500">Total Assemblies</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.core_stats.total_assemblies.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Pending Vector Updates</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.core_stats.pending_vector_updates.toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-4 text-gray-400">
                    <p>No statistics available</p>
                  </div>
                )}
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Performance</CardTitle>
              </CardHeader>
              <CardContent>
                {memoryCoreStats.isLoading ? (
                  <div className="space-y-4">
                    <Skeleton className="h-8 w-full" />
                    <Skeleton className="h-8 w-full" />
                  </div>
                ) : memoryCoreStats.isError ? (
                  <Alert variant="destructive">
                    <AlertDescription>
                      Failed to load performance data
                    </AlertDescription>
                  </Alert>
                ) : memoryCoreStats.data?.data?.quick_recal_stats || memoryCoreStats.data?.data?.threshold_stats ? (
                  <div className="space-y-4">
                    {memoryCoreStats.data?.data?.quick_recal_stats && (
                      <div>
                        <div className="flex justify-between mb-1">
                          <p className="text-sm text-gray-500">Quick Recall Rate</p>
                          <p className="text-sm font-mono">
                            {((memoryCoreStats.data.data.quick_recal_stats.recall_rate ?? 0) * 100).toFixed(2)}%
                          </p>
                        </div>
                        <Progress value={(memoryCoreStats.data.data.quick_recal_stats.recall_rate ?? 0) * 100} className="h-2" />
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
                        <Progress value={(memoryCoreStats.data.data.threshold_stats.recall_rate ?? 0) * 100} className="h-2" />
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-400">Performance data unavailable</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="vector-index" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Vector Index Stats</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.isError ? (
                <Alert variant="destructive">
                  <AlertTitle>Failed to load vector index data</AlertTitle>
                  <AlertDescription>
                    {memoryCoreStats.error?.message || "An error occurred while fetching vector index information."}
                  </AlertDescription>
                </Alert>
              ) : memoryCoreStats.data?.data?.vector_index_stats ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Index Size</p>
                      <p className="text-lg font-mono">
                        {memoryCoreStats.data.data.vector_index_stats.count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Mapping Count</p>
                      <p className="text-lg font-mono">
                        {memoryCoreStats.data.data.vector_index_stats.mapping_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Indexed Vectors</p>
                      <p className="text-lg font-mono">
                        {memoryCoreStats.data.data.vector_index_stats.count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Drift Count</p>
                      <p className="text-lg font-mono">
                        {(memoryCoreStats.data.data.vector_index_stats.drift_count ?? 0).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  
                  <div>
                    <div className="mb-2">
                      <h3 className="font-medium">Health & Consistency</h3>
                      <p className="text-sm text-gray-500">Statistics about consistency between memory store and vector index</p>
                    </div>
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-muted">
                          <TableHead className="w-1/2">Metric</TableHead>
                          <TableHead>Value</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        <TableRow>
                          <TableCell className="font-medium">Index Score</TableCell>
                          <TableCell>
                            {(memoryCoreStats.data.data.vector_index_stats.count / 
                              (memoryCoreStats.data.data.core_stats.total_memories || 1)).toFixed(2)}
                          </TableCell>
                          <TableCell>
                            {(memoryCoreStats.data.data.vector_index_stats.count / 
                              (memoryCoreStats.data.data.core_stats.total_memories || 1)) > 0.95 ? (
                              <Badge className="bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400">
                                <i className="fas fa-check mr-1"></i>
                                Good
                              </Badge>
                            ) : (memoryCoreStats.data.data.vector_index_stats.count / 
                              (memoryCoreStats.data.data.core_stats.total_memories || 1)) > 0.8 ? (
                              <Badge className="bg-yellow-100 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400">
                                <i className="fas fa-exclamation-triangle mr-1"></i>
                                Warning
                              </Badge>
                            ) : (
                              <Badge className="bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400">
                                <i className="fas fa-times mr-1"></i>
                                Critical
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Drift Count</TableCell>
                          <TableCell>
                            {(memoryCoreStats.data.data.vector_index_stats.drift_count ?? 0).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            {(memoryCoreStats.data.data.vector_index_stats.drift_count ?? 0) < 10 ? (
                              <Badge className="bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400">
                                <i className="fas fa-check mr-1"></i>
                                Good
                              </Badge>
                            ) : (memoryCoreStats.data.data.vector_index_stats.drift_count ?? 0) < 50 ? (
                              <Badge className="bg-yellow-100 dark:bg-yellow-900/20 text-yellow-600 dark:text-yellow-400">
                                <i className="fas fa-exclamation-triangle mr-1"></i>
                                Warning
                              </Badge>
                            ) : (
                              <Badge className="bg-red-100 dark:bg-red-900/20 text-red-600 dark:text-red-400">
                                <i className="fas fa-times mr-1"></i>
                                Critical
                              </Badge>
                            )}
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell className="font-medium">Index Type</TableCell>
                          <TableCell colSpan={2}>
                            {memoryCoreStats.data.data.vector_index_stats.index_type || 'Unknown'}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-gray-400">
                  <p>Vector index data unavailable</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="assemblies" className="mt-4">
          <AssemblyTable 
            assemblies={assemblies.data?.data || []} 
            isLoading={assemblies.isLoading}
            isError={assemblies.isError}
            error={assemblies.error}
          />
        </TabsContent>
        
        <TabsContent value="persistence" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Persistence Status</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.isError ? (
                <Alert variant="destructive">
                  <AlertTitle>Failed to load persistence data</AlertTitle>
                  <AlertDescription>
                    {memoryCoreStats.error?.message || "An error occurred while fetching persistence information."}
                  </AlertDescription>
                </Alert>
              ) : memoryCoreStats.data?.data?.persistence_stats ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Last Update</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.persistence_stats.last_update ? 
                          new Date(memoryCoreStats.data.data.persistence_stats.last_update).toLocaleString() : 
                          'Never'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 mb-1">Last Backup</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.persistence_stats.last_backup ? 
                          new Date(memoryCoreStats.data.data.persistence_stats.last_backup).toLocaleString() : 
                          'Never'}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-gray-400">
                  <p>Persistence data unavailable</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}
