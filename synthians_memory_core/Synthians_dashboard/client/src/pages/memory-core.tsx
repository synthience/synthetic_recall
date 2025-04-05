import React, { useState } from "react";
import { useMemoryCoreHealth, useMemoryCoreStats, useAssemblies } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { ServiceStatus } from "@/components/layout/ServiceStatus";
import { usePollingStore } from "@/lib/store";

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
  } : null;
  
  // Calculate warning thresholds
  const isDriftAboveWarning = memoryCoreStats.data?.data?.vector_index?.drift_count > 50;
  const isDriftAboveCritical = memoryCoreStats.data?.data?.vector_index?.drift_count > 100;
  
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
            {serviceStatus ? (
              <ServiceStatus service={serviceStatus} />
            ) : (
              <Skeleton className="h-5 w-20" />
            )}
          </div>
        </CardHeader>
        <CardContent>
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
                ) : memoryCoreStats.data?.data ? (
                  <div className="grid grid-cols-2 gap-6">
                    <div>
                      <div className="mb-4">
                        <p className="text-sm text-gray-500">Total Memories</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.total_memories.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Dirty Items</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.dirty_items.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div>
                      <div className="mb-4">
                        <p className="text-sm text-gray-500">Total Assemblies</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.total_assemblies.toLocaleString()}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Pending Vector Updates</p>
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.pending_vector_updates.toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-400">Failed to load Memory Core stats</p>
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
                ) : memoryCoreStats.data?.data?.performance ? (
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <p className="text-sm text-gray-500">Quick Recall Rate</p>
                        <p className="text-sm font-mono">
                          {memoryCoreStats.data.data.performance.quick_recall_rate.toFixed(2)}%
                        </p>
                      </div>
                      <Progress value={memoryCoreStats.data.data.performance.quick_recall_rate} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <p className="text-sm text-gray-500">Threshold Recall Rate</p>
                        <p className="text-sm font-mono">
                          {memoryCoreStats.data.data.performance.threshold_recall_rate.toFixed(2)}%
                        </p>
                      </div>
                      <Progress value={memoryCoreStats.data.data.performance.threshold_recall_rate} className="h-2" />
                    </div>
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
              ) : memoryCoreStats.data?.data?.vector_index ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Vector Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.vector_index.count.toLocaleString()}
                      </p>
                    </div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Mapping Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.vector_index.mapping_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Drift Count</p>
                      <div className="flex items-center">
                        <p className="text-2xl font-mono">
                          {memoryCoreStats.data.data.vector_index.drift_count.toLocaleString()}
                        </p>
                        {isDriftAboveCritical && (
                          <Badge variant="destructive" className="ml-2">Critical</Badge>
                        )}
                        {isDriftAboveWarning && !isDriftAboveCritical && (
                          <Badge variant="outline" className="ml-2 text-yellow-400 border-yellow-400">Warning</Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Index Type</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.vector_index.index_type}
                      </p>
                    </div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">GPU Status</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.vector_index.gpu_enabled ? (
                          <span className="text-green-400">Enabled</span>
                        ) : (
                          <span className="text-gray-400">Disabled</span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Vector index data unavailable</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="assemblies" className="mt-4">
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Assembly Stats</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.data?.data?.assembly_stats ? (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Total Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.total_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Average Size</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.average_size.toFixed(1)}
                      </p>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Indexed Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.indexed_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Pruning Status</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.assembly_stats.pruning_enabled ? (
                          <span className="text-green-400">Enabled</span>
                        ) : (
                          <span className="text-gray-400">Disabled</span>
                        )}
                      </p>
                    </div>
                  </div>
                  <div>
                    <div className="mb-4">
                      <p className="text-sm text-gray-500">Vector Indexed Count</p>
                      <p className="text-2xl font-mono">
                        {memoryCoreStats.data.data.assembly_stats.vector_indexed_count.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Merging Status</p>
                      <p className="text-lg">
                        {memoryCoreStats.data.data.assembly_stats.merging_enabled ? (
                          <span className="text-green-400">Enabled</span>
                        ) : (
                          <span className="text-gray-400">Disabled</span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Assembly stats unavailable</p>
              )}
            </CardContent>
          </Card>
          
          <AssemblyTable
            assemblies={assemblies.data?.data || null}
            isLoading={assemblies.isLoading}
            title="All Assemblies"
          />
        </TabsContent>
        
        <TabsContent value="persistence" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Persistence Stats</CardTitle>
            </CardHeader>
            <CardContent>
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : memoryCoreStats.data?.data?.persistence ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-gray-500 mb-2">Last Update</p>
                    <p className="text-lg">
                      {new Date(memoryCoreStats.data.data.persistence.last_update).toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 mb-2">Last Backup</p>
                    <p className="text-lg">
                      {new Date(memoryCoreStats.data.data.persistence.last_backup).toLocaleString()}
                    </p>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400">Persistence data unavailable</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}
