import React, { useEffect, useState } from "react";
import { useAssembly, useAssemblyLineage, useExplainMerge, useExplainActivation } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { useToast } from "@/hooks/use-toast";
import { Link, useParams } from "wouter";
import { usePollingStore } from "@/lib/store";
import { LineageView } from "@/components/dashboard/LineageView";
import { MergeExplanationView } from "@/components/dashboard/MergeExplanationView";
import { ActivationExplanationView } from "@/components/dashboard/ActivationExplanationView";
import { useFeatures } from "@/contexts/FeaturesContext";
import { Input } from "@/components/ui/input";

export default function AssemblyDetail() {
  const { id } = useParams();
  const { refreshAllData } = usePollingStore();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  const { explainabilityEnabled } = useFeatures();
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const [maxDepth, setMaxDepth] = useState<number>(5);
  
  // Fetch assembly data
  const { data, isLoading, isError, error, refetch } = useAssembly(id || null);
  
  // Fetch explainability data when needed
  const lineageQuery = useAssemblyLineage(id || null);
  const mergeExplanationQuery = useExplainMerge(id || null);
  const activationExplanationQuery = useExplainActivation(id || null, selectedMemoryId || undefined);
  
  useEffect(() => {
    if (isError) {
      toast({
        title: "Error loading assembly",
        description: (error as Error)?.message || "Could not load assembly details",
        variant: "destructive"
      });
    }
  }, [isError, error, toast]);
  
  // Helper function to get sync status
  const getSyncStatus = (assembly: any) => {
    if (!assembly?.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-500 dark:text-yellow-400",
        bgColor: "bg-yellow-100 dark:bg-yellow-900/20",
        icon: "fas fa-clock"
      };
    }
    
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    
    if (vectorDate >= updateDate) {
      return {
        label: "Indexed",
        color: "text-green-600 dark:text-green-400",
        bgColor: "bg-green-100 dark:bg-green-900/20",
        icon: "fas fa-check"
      };
    }
    
    return {
      label: "Syncing",
      color: "text-blue-600 dark:text-blue-400",
      bgColor: "bg-blue-100 dark:bg-blue-900/20",
      icon: "fas fa-sync-alt"
    };
  };
  
  // Format date for display
  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };
  
  const assembly = data;
  const syncStatus = assembly ? getSyncStatus(assembly) : null;
  
  const handleRefresh = () => {
    refetch();
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <div className="flex items-center mb-1">
            <Link href="/assemblies">
              <Button variant="ghost" size="sm" className="mr-2 -ml-2">
                <i className="fas fa-arrow-left mr-1"></i> Back
              </Button>
            </Link>
            <h2 className="text-xl font-semibold text-white">Assembly Inspector</h2>
          </div>
          <p className="text-sm text-gray-400">
            {isLoading ? (
              <Skeleton className="h-4 w-64 inline-block" />
            ) : assembly ? (
              <>Viewing details for assembly <code className="text-primary">{assembly.id}</code></>
            ) : (
              <>Assembly not found</>
            )}
          </p>
        </div>
        <RefreshButton onClick={handleRefresh} />
      </div>
      
      {isLoading ? (
        <div className="space-y-6">
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      ) : !assembly ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <i className="fas fa-folder-open text-4xl text-muted-foreground mb-4"></i>
            <h3 className="text-xl font-medium mb-2">Assembly Not Found</h3>
            <p className="text-muted-foreground mb-6">The assembly with ID "{id}" could not be found.</p>
            <Link href="/assemblies">
              <Button>
                <i className="fas fa-arrow-left mr-2"></i> Back to Assemblies
              </Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Assembly Header */}
          <Card className="mb-6">
            <CardHeader>
              <div className="flex justify-between">
                <div>
                  <CardTitle className="text-xl">{assembly.name}</CardTitle>
                  <CardDescription className="mt-2">{assembly.description || "No description provided"}</CardDescription>
                </div>
                <Badge 
                  variant="outline" 
                  className={`${syncStatus?.bgColor} ${syncStatus?.color} self-start`}
                >
                  <i className={`${syncStatus?.icon} mr-1 text-xs`}></i>
                  {syncStatus?.label}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-500 mb-1">Assembly ID</p>
                  <p className="font-mono text-secondary">{assembly.id}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Memory Count</p>
                  <p className="font-mono">{assembly.member_count}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Created</p>
                  <p className="text-sm">{formatDate(assembly.created_at)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 mb-1">Last Updated</p>
                  <p className="text-sm">{formatDate(assembly.updated_at)}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Tabs for Content */}
          <Tabs defaultValue="overview" className="mb-6">
            <TabsList>
              <TabsTrigger value="overview" onClick={() => setActiveTab("overview")}>Overview</TabsTrigger>
              <TabsTrigger value="members" onClick={() => setActiveTab("members")}>Memory Members</TabsTrigger>
              <TabsTrigger value="metadata" onClick={() => setActiveTab("metadata")}>Metadata</TabsTrigger>
              {assembly.vector_index_updated_at && (
                <TabsTrigger value="embedding" onClick={() => setActiveTab("embedding")}>Embedding</TabsTrigger>
              )}
              {explainabilityEnabled && (
                <>
                  <TabsTrigger value="lineage" onClick={() => setActiveTab("lineage")}>Lineage</TabsTrigger>
                  <TabsTrigger value="merge" onClick={() => {
                    setActiveTab("merge");
                    mergeExplanationQuery.refetch();
                  }}>Merge Explanation</TabsTrigger>
                  <TabsTrigger value="activation" onClick={() => setActiveTab("activation")}>Activation</TabsTrigger>
                </>
              )}
            </TabsList>
            
            <TabsContent value="overview" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Assembly Overview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Key Information</h3>
                      
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Name</p>
                          <p>{assembly.name}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Description</p>
                          <p>{assembly.description || "No description available"}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Member Count</p>
                          <p>{assembly.member_count} memories</p>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-4">Sync Status</h3>
                      
                      <div className="space-y-3">
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Last Updated</p>
                          <p>{formatDate(assembly.updated_at)}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Vector Index Updated</p>
                          <p>{assembly.vector_index_updated_at ? formatDate(assembly.vector_index_updated_at) : "Not indexed yet"}</p>
                        </div>
                        
                        <div>
                          <p className="text-sm text-gray-500 mb-1">Status</p>
                          <Badge 
                            variant="outline" 
                            className={`${syncStatus?.bgColor} ${syncStatus?.color}`}
                          >
                            <i className={`${syncStatus?.icon} mr-1 text-xs`}></i>
                            {syncStatus?.label}
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="members" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Memory Members</CardTitle>
                  <CardDescription>List of memory IDs that are part of this assembly</CardDescription>
                </CardHeader>
                <CardContent>
                  {assembly.memory_ids && assembly.memory_ids.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-96 overflow-y-auto p-1">
                      {assembly.memory_ids.map((memoryId: string) => (
                        <div key={memoryId} className="bg-muted p-2 rounded-md font-mono text-xs flex items-center">
                          <i className="fas fa-memory text-secondary mr-2"></i>
                          {memoryId}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      <i className="fas fa-info-circle mr-2"></i>
                      No memory members found
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="metadata" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle>Metadata</CardTitle>
                  <CardDescription>Keywords, tags, and topics associated with this assembly</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <h3 className="text-sm font-medium mb-3">Keywords</h3>
                      {assembly.keywords && assembly.keywords.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {assembly.keywords.map((keyword: string, idx: number) => (
                            <Badge key={idx} variant="secondary">
                              {keyword}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-sm">No keywords available</p>
                      )}
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-3">Tags</h3>
                      {assembly.tags && assembly.tags.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {assembly.tags.map((tag: string, idx: number) => (
                            <Badge key={idx} variant="outline" className="border-primary text-primary">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-sm">No tags available</p>
                      )}
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-3">Topics</h3>
                      {assembly.topics && assembly.topics.length > 0 ? (
                        <div className="flex flex-wrap gap-2">
                          {assembly.topics.map((topic: string, idx: number) => (
                            <Badge key={idx} variant="outline" className="border-secondary text-secondary">
                              {topic}
                            </Badge>
                          ))}
                        </div>
                      ) : (
                        <p className="text-gray-400 text-sm">No topics available</p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            {assembly.vector_index_updated_at && (
              <TabsContent value="embedding" className="mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Composite Embedding</CardTitle>
                    <CardDescription>Vector representation visualization (placeholder)</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-muted rounded-lg p-6 flex flex-col items-center justify-center min-h-[240px]">
                      <div className="mb-4 text-center">
                        <i className="fas fa-project-diagram text-4xl text-primary mb-4"></i>
                        <p className="text-muted-foreground">
                          Embedding visualization is a future enhancement.
                        </p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-md">
                        <div className="bg-card p-3 rounded">
                          <p className="text-xs text-gray-500 mb-1">Embedding Norm</p>
                          <p className="font-mono">0.9873</p>
                        </div>
                        <div className="bg-card p-3 rounded">
                          <p className="text-xs text-gray-500 mb-1">Sparsity</p>
                          <p className="font-mono">0.0418</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            )}
            
            {explainabilityEnabled && (
              <>
                <TabsContent value="lineage" className="mt-4">
                  <LineageView 
                    lineage={lineageQuery.data?.lineage} 
                    isLoading={lineageQuery.isLoading} 
                    isError={lineageQuery.isError} 
                    error={lineageQuery.error as Error | null}
                  />
                  <Card className="mt-4">
                    <CardHeader>
                      <CardTitle>Lineage Options</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center gap-4">
                        <div className="flex-1">
                          <p className="text-sm text-muted-foreground mb-2">Maximum Depth</p>
                          <Input 
                            type="number" 
                            min={1} 
                            max={10} 
                            value={maxDepth} 
                            onChange={(e) => setMaxDepth(parseInt(e.target.value) || 5)} 
                          />
                        </div>
                        <Button onClick={() => lineageQuery.refetch()} className="self-end">
                          Refresh Lineage
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="merge" className="mt-4">
                  <MergeExplanationView 
                    mergeData={mergeExplanationQuery.data?.explanation} 
                    isLoading={mergeExplanationQuery.isLoading} 
                    isError={mergeExplanationQuery.isError} 
                    error={mergeExplanationQuery.error as Error | null}
                  />
                  <Card className="mt-4">
                    <CardHeader>
                      <CardTitle>Merge Explanation Actions</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Button onClick={() => mergeExplanationQuery.refetch()}>
                        Refresh Merge Data
                      </Button>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="activation" className="mt-4">
                  <Card className="mb-4">
                    <CardHeader>
                      <CardTitle>Select Memory to Explain Activation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 max-h-48 overflow-y-auto p-1">
                        {assembly?.memory_ids?.map((memId: string) => (
                          <Button 
                            key={memId} 
                            variant={selectedMemoryId === memId ? "default" : "outline"}
                            className={`justify-start font-mono text-xs ${selectedMemoryId === memId ? "bg-primary" : ""}`}
                            onClick={() => {
                              setSelectedMemoryId(memId);
                              activationExplanationQuery.refetch();
                            }}
                          >
                            <i className="fas fa-memory mr-2"></i>
                            {memId}
                          </Button>
                        )) || <p className="text-muted-foreground">No memories in this assembly</p>}
                      </div>
                    </CardContent>
                  </Card>
                  
                  {selectedMemoryId ? (
                    <ActivationExplanationView 
                      activationData={activationExplanationQuery.data?.explanation} 
                      memoryId={selectedMemoryId}
                      isLoading={activationExplanationQuery.isLoading} 
                      isError={activationExplanationQuery.isError} 
                      error={activationExplanationQuery.error as Error | null}
                    />
                  ) : (
                    <Card>
                      <CardContent className="text-center py-8">
                        <p className="text-muted-foreground">Select a memory to view activation details</p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>
              </>
            )}
          </Tabs>
        </>
      )}
    </>
  );
}
