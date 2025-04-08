import React, { useState } from "react";
// Import hooks and components
import { useAssembly } from "@/lib/api";
import { useAssemblyExplainabilityTabs } from "@/lib/api/hooks/useAssemblyExplainabilityTabs";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { useToast } from "@/hooks/use-toast";
import { Link, useParams, useLocation } from "wouter";
import { usePollingStore } from "@/lib/store";
import { LineageView } from "@/components/dashboard/LineageView";
import { MergeExplanationView } from "@/components/dashboard/MergeExplanationView";
import { ActivationExplanationView } from "@/components/dashboard/ActivationExplanationView";
import { useFeatures } from "@/contexts/FeaturesContext";
import { Input } from "@/components/ui/input";
import { Assembly } from "@shared/schema";
import { AlertCircle, ArrowLeft, Calendar, Tag } from "lucide-react";
import { formatDistanceToNow } from 'date-fns';
// Import new components
import { ErrorDisplay } from "@/components/ui/ErrorDisplay";
import { ExplainabilityDataView } from "@/components/dashboard/ExplainabilityDataView";
import { LineageResponse, ExplainMergeResponse, ExplainActivationResponse } from "@shared/schema";

// Adapter components to solve type compatibility issues
const LineageViewAdapter = ({ data }: { data: LineageResponse }) => {
  return <LineageView lineage={data.lineage || null} />;
};

const MergeExplanationViewAdapter = ({ data }: { data: ExplainMergeResponse }) => {
  return <MergeExplanationView explanation={data.explanation || null} />;
};

const ActivationExplanationViewAdapter = ({ data }: { data: ExplainActivationResponse }) => {
  // ExplainActivationResponse has explanation.memory_id, not a direct memory_id property
  return <ActivationExplanationView 
    explanation={data.explanation || null} 
    memoryId={data.explanation?.memory_id || ''} 
  />;
};

export default function AssemblyDetail() {
  const params = useParams<{ id: string }>();
  const [, setLocation] = useLocation();
  const id = params?.id;
  const { refreshAllData } = usePollingStore();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  const { explainabilityEnabled, usingFallbackConfig } = useFeatures();

  // Core assembly query (always needed)
  const assemblyQuery = useAssembly(id || null);
  const assembly = assemblyQuery.data?.data;

  // Use the consolidated hook for all explainability tabs
  const {
    lineageQuery,
    mergeQuery: mergeExplanationQuery,
    activationQuery: activationExplanationQuery,
    selectedMemoryId,
    selectMemory
  } = useAssemblyExplainabilityTabs(id, activeTab);

  // --- Helper Functions ---
  const formatDate = (timestamp: string | undefined): string => {
    return timestamp ? new Date(timestamp).toLocaleString() : "N/A";
  };

  const getSyncStatus = (assembly: Assembly | null | undefined) => {
    if (!assembly || !assembly.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-500 dark:text-yellow-400",
        bgColor: "bg-yellow-100 dark:bg-yellow-900/20",
        icon: "fas fa-clock"
      };
    }

    // Ensure updated_at exists and is valid before creating Date object
    const updateDate = assembly.updated_at ? new Date(assembly.updated_at) : new Date(0);
    const vectorDate = new Date(assembly.vector_index_updated_at);

    if (!isNaN(vectorDate.getTime()) && !isNaN(updateDate.getTime())) {
         if (vectorDate >= updateDate) {
           return {
             label: "Indexed",
             color: "text-green-600 dark:text-green-400",
             bgColor: "bg-green-100 dark:bg-green-900/20",
             icon: "fas fa-check"
           };
         } else {
            return {
              label: "Syncing",
              color: "text-blue-600 dark:text-blue-400",
              bgColor: "bg-blue-100 dark:bg-blue-900/20",
              icon: "fas fa-sync-alt"
            };
         }
    } else {
         // Handle invalid date case
          return {
            label: "Invalid Date",
            color: "text-red-500 dark:text-red-400",
            bgColor: "bg-red-100 dark:bg-red-900/20",
            icon: "fas fa-exclamation-triangle"
          };
    }
  };

  const syncStatus = getSyncStatus(assembly);

  // --- Handlers ---
  const handleRefresh = async () => {
    await assemblyQuery.refetch();

    // The tab refetching is now handled by the hook internally
    // We just need to make sure the current tab is set properly

    toast({
      title: "Refreshing Assembly Data",
      description: "Fetching the latest information for this assembly.",
    });
  };

  const handleSelectMemory = (memoryId: string) => {
    selectMemory(memoryId);
  };

  // --- Render Logic ---

  // Loading State
  if (assemblyQuery.isLoading) {
      return (
        <div className="p-6 space-y-6">
          <div className="flex items-center">
            <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <Skeleton className="h-6 w-48 ml-2" />
          </div>
           <Skeleton className="h-40 w-full" />
           <Skeleton className="h-96 w-full" />
         </div>
      )
  }

  // Error State - Use the ErrorDisplay component
  if (assemblyQuery.isError || !assemblyQuery.data?.success) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center">
           <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
             <ArrowLeft className="h-4 w-4" />
           </Button>
           <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
        </div>
        <ErrorDisplay 
          error={assemblyQuery.error}
          refetch={assemblyQuery.refetch}
          title="Error Loading Assembly"
          message={assemblyQuery.data?.error || "Failed to load assembly details"}
        />
      </div>
    );
  }

  // Not Found State - Invalid Entity Guard
  if (!assembly) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center">
           <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
             <ArrowLeft className="h-4 w-4" />
           </Button>
           <h1 className="text-2xl font-semibold ml-2">Assembly Not Found</h1>
        </div>
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-col items-center justify-center p-6">
              <AlertCircle className="h-12 w-12 text-red-500 mb-4" />
              <h2 className="text-xl font-semibold mb-2">Assembly Not Found</h2>
              <p className="text-gray-500 mb-4">The requested assembly could not be found or may have been deleted.</p>
              <Button onClick={() => setLocation('/assemblies')}>Return to Assemblies</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Render Assembly Detail when data is available
  return (
    <div className="p-6 space-y-6">
      {/* Header with back button and title */}
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-2xl font-semibold ml-2">Assembly Detail</h1>
        </div>
        <RefreshButton onClick={handleRefresh} isLoading={assemblyQuery.isFetching} />
      </div>

      {/* Assembly Card */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="text-2xl font-semibold">
                {assembly.name || "Unnamed Assembly"}
              </CardTitle>
              <CardDescription className="text-sm mt-1">
                ID: {assembly.id}
              </CardDescription>
            </div>
            <Badge 
              variant="outline"
              className={`${syncStatus.color} ${syncStatus.bgColor} px-2 py-1`}>
                {syncStatus.label}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {/* Assembly Info Rows */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <div className="flex items-center">
                <Calendar className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">Created:</span>
                <span className="text-sm ml-2">{formatDate(assembly.created_at)}</span>
              </div>
              <div className="flex items-center">
                <Calendar className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">Updated:</span>
                <span className="text-sm ml-2">{assembly.updated_at ? `${formatDate(assembly.updated_at)} (${formatDistanceToNow(new Date(assembly.updated_at), { addSuffix: true })})` : "N/A"}</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center">
                <Tag className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">Memory Count:</span>
                <span className="text-sm ml-2">{assembly.memory_ids?.length || 0} memories</span>
              </div>
              <div className="flex items-center">
                <Tag className="h-4 w-4 mr-2" />
                <span className="text-sm font-medium">Vector Index:</span>
                <span className="text-sm ml-2">{assembly.vector_index_updated_at ? formatDate(assembly.vector_index_updated_at) : "Not indexed"}</span>
              </div>
            </div>
          </div>

          {/* Memory IDs List (Collapsible) */}
          {assembly.memory_ids && assembly.memory_ids.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium mb-2">Memory IDs:</div>
              <div className="max-h-32 overflow-y-auto text-xs bg-gray-50 dark:bg-gray-900 p-2 rounded border">
                {assembly.memory_ids.map((memoryId, index) => (
                  <div 
                    key={index} 
                    className={`py-1 px-2 cursor-pointer rounded mb-1 ${selectedMemoryId === memoryId ? 'bg-blue-100 dark:bg-blue-900/30' : 'hover:bg-gray-100 dark:hover:bg-gray-800'}`}
                    onClick={() => handleSelectMemory(memoryId)}
                  >
                    {memoryId}
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Tabs for different views */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-2 md:grid-cols-4 lg:w-[600px]">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          {explainabilityEnabled && (
            <>
              <TabsTrigger value="lineage">Lineage</TabsTrigger>
              <TabsTrigger value="merge">Merge Log</TabsTrigger>
              <TabsTrigger value="activation">Activation</TabsTrigger>
            </>
          )}
        </TabsList>

        <TabsContent value="overview" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Assembly Overview</CardTitle>
              <CardDescription>Details about this assembly and its state</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Metadata Display */}
                {(assembly as any).metadata && (
                  <div>
                    <h3 className="text-lg font-medium mb-2">Metadata</h3>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded p-4 overflow-x-auto">
                      <pre className="text-xs">{JSON.stringify((assembly as any).metadata, null, 2)}</pre>
                    </div>
                  </div>
                )}

                {/* Statistics */}
                <div>
                  <h3 className="text-lg font-medium mb-2">Statistics</h3>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Memories</div>
                        <div className="text-2xl font-bold">{assembly.memory_ids?.length || 0}</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Age</div>
                        <div className="text-2xl font-bold">
                          {assembly.created_at ? formatDistanceToNow(new Date(assembly.created_at)) : "N/A"}
                        </div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4">
                        <div className="text-sm font-medium text-gray-500 dark:text-gray-400">Status</div>
                        <div className="text-2xl font-bold">{syncStatus.label}</div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {explainabilityEnabled && (
          <>
            {/* Lineage Tab - Using new ExplainabilityDataView component */}
            <TabsContent value="lineage" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Assembly Lineage</CardTitle>
                  <CardDescription>View the hierarchical lineage of this assembly</CardDescription>
                </CardHeader>
                <CardContent>
                  <ExplainabilityDataView
                    hookResult={lineageQuery}
                    RenderComponent={LineageViewAdapter}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            {/* Merge Tab - Using new ExplainabilityDataView component */}
            <TabsContent value="merge" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Merge Explanation</CardTitle>
                  <CardDescription>Understand the merge process and decision making</CardDescription>
                </CardHeader>
                <CardContent>
                  <ExplainabilityDataView
                    hookResult={mergeExplanationQuery}
                    RenderComponent={MergeExplanationViewAdapter}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            {/* Activation Tab - Using new ExplainabilityDataView component */}
            <TabsContent value="activation" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Activation Explanation</CardTitle>
                  <CardDescription>Explore the activation process for a selected memory in this assembly</CardDescription>
                </CardHeader>
                <CardContent>
                  {!selectedMemoryId ? (
                    <div className="text-center py-8">
                      <p className="text-gray-500 mb-4">Please select a memory ID from the list above to view its activation details</p>
                    </div>
                  ) : (
                    <ExplainabilityDataView
                      hookResult={activationExplanationQuery}
                      RenderComponent={ActivationExplanationViewAdapter}
                    />
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </>
        )}
      </Tabs>

      {/* Display a small indicator if using fallback configuration */}
      {usingFallbackConfig && process.env.NODE_ENV === 'development' && (
        <div className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
          <AlertCircle className="h-3 w-3 inline mr-1" /> 
          Using fallback configuration - some features may be limited
        </div>
      )}
    </div>
  );
}