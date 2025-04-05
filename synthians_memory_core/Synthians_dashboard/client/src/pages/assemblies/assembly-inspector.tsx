import React, { useState } from 'react';
import { useLocation, useParams } from 'wouter';
import { useAssemblyDetails } from '@/lib/api/hooks/useAssemblyDetails';
import { useAssemblyLineage, useExplainMerge, useExplainActivation } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineageView } from '@/components/dashboard/LineageView';
import { MergeExplanationView } from '@/components/dashboard/MergeExplanationView';
import { ActivationExplanationView } from '@/components/dashboard/ActivationExplanationView';
import { ArrowLeft, Calendar, Tag } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useFeatures } from '@/contexts/FeaturesContext';
import { ExplainMergeData, ExplainActivationData } from '@shared/schema';

export default function AssemblyInspector() {
  const [, setLocation] = useLocation();
  const params = useParams<{ id: string }>();
  const assemblyId = params?.id;
  const { explainabilityEnabled } = useFeatures();
  
  // Selected memory for activation explanation
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  
  // Fetch assembly details
  const { data: assembly, isLoading, isError, error } = useAssemblyDetails(assemblyId);
  
  // Fetch lineage data
  const lineageQuery = useAssemblyLineage(assemblyId);
  
  // Prepare merge explanation query (triggered manually)
  const mergeExplanationQuery = useExplainMerge(assemblyId);
  
  // Prepare activation explanation query (triggered manually)
  const activationExplanationQuery = useExplainActivation(assemblyId, selectedMemoryId);
  
  // Handle memory selection for activation explanation
  const handleMemorySelect = (memoryId: string) => {
    setSelectedMemoryId(memoryId);
    activationExplanationQuery.refetch();
  };
  
  // Handle loading merge explanation
  const handleExplainMerge = () => {
    mergeExplanationQuery.refetch();
  };
  
  if (isLoading) {
    return (
      <div className="container py-6">
        <div className="space-y-6">
          <div className="flex items-center">
            <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
          </div>
          <Card>
            <CardHeader>
              <CardTitle><Skeleton className="h-6 w-48" /></CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-2/3" />
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }
  
  if (isError || !assembly) {
    return (
      <div className="container py-6">
        <div className="space-y-6">
          <div className="flex items-center">
            <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
          </div>
          <Card>
            <CardContent className="p-6">
              <div className="text-center py-8">
                <h2 className="text-xl font-medium text-red-500 mb-2">Failed to load assembly details</h2>
                <p className="text-muted-foreground">
                  {error?.message || 'Could not retrieve assembly information. Please try again.'}
                </p>
                <Button className="mt-4" onClick={() => setLocation('/assemblies')}>
                  Return to Assemblies
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container py-6">
      <div className="space-y-6">
        {/* Header with back button */}
        <div className="flex items-center">
          <Button variant="ghost" size="icon" onClick={() => setLocation('/assemblies')}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-2xl font-semibold ml-2">Assembly Inspector</h1>
        </div>
        
        {/* Assembly information card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{assembly.name || 'Unnamed Assembly'}</span>
              <Badge variant="outline">{assembly.id}</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">{assembly.description || 'No description available'}</p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <h3 className="text-sm font-medium flex items-center gap-1 text-muted-foreground">
                  <Calendar className="h-4 w-4" /> Created
                </h3>
                <p>
                  {new Date(assembly.created_at).toLocaleString()} 
                  <span className="text-muted-foreground ml-2 text-sm">
                    ({formatDistanceToNow(new Date(assembly.created_at), { addSuffix: true })})
                  </span>
                </p>
              </div>
              
              <div>
                <h3 className="text-sm font-medium flex items-center gap-1 text-muted-foreground">
                  <Tag className="h-4 w-4" /> Tags
                </h3>
                <div className="flex flex-wrap gap-1 mt-1">
                  {assembly.tags && assembly.tags.length > 0 ? (
                    assembly.tags.map((tag: string) => (
                      <Badge key={tag} variant="secondary">{tag}</Badge>
                    ))
                  ) : (
                    <span className="text-muted-foreground text-sm">No tags</span>
                  )}
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-muted-foreground">Memories</h3>
                <p>{assembly.memory_ids?.length || 0} memories in this assembly</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {!explainabilityEnabled && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 my-2">
            <p className="font-medium">Explainability features are disabled</p>
            <p className="text-sm">Some features like merge explanations and activation details are not available. Enable them by setting <code>ENABLE_EXPLAINABILITY=true</code> in the Memory Core configuration.</p>
          </div>
        )}
        
        {/* Explainability tabs */}
        <Tabs defaultValue="lineage" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="lineage">Lineage</TabsTrigger>
            <TabsTrigger value="merge" disabled={!explainabilityEnabled}>Merge Explanation</TabsTrigger>
            <TabsTrigger value="memories" disabled={!explainabilityEnabled}>Memories & Activation</TabsTrigger>
          </TabsList>
          
          <TabsContent value="lineage" className="pt-4">
            <LineageView 
              lineage={lineageQuery.data?.lineage} 
              isLoading={lineageQuery.isLoading} 
              isError={lineageQuery.isError} 
              error={lineageQuery.error as Error}
            />
          </TabsContent>
          
          <TabsContent value="merge" className="pt-4">
            <div className="space-y-4">
              {!mergeExplanationQuery.data && !mergeExplanationQuery.isLoading && (
                <div className="text-center p-6 bg-muted rounded-md">
                  <p className="mb-4">Merge explanation data hasn't been loaded yet.</p>
                  <Button onClick={handleExplainMerge} disabled={!explainabilityEnabled}>
                    Explain How This Assembly Was Formed
                  </Button>
                </div>
              )}
              
              {(mergeExplanationQuery.data || mergeExplanationQuery.isLoading) && (
                <MergeExplanationView 
                  mergeData={mergeExplanationQuery.data?.explanation as ExplainMergeData | undefined} 
                  isLoading={mergeExplanationQuery.isLoading} 
                  isError={mergeExplanationQuery.isError} 
                  error={mergeExplanationQuery.error as Error}
                />
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="memories" className="pt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Memory selection */}
              <Card>
                <CardHeader>
                  <CardTitle>Memories in Assembly</CardTitle>
                </CardHeader>
                <CardContent>
                  {assembly.memory_ids?.length === 0 ? (
                    <div className="text-center p-4 text-muted-foreground">
                      <p>No memories in this assembly.</p>
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                      {assembly.memory_ids?.map((memoryId: string) => (
                        <div 
                          key={memoryId}
                          className={`p-3 border rounded-md cursor-pointer hover:bg-muted transition-colors ${selectedMemoryId === memoryId ? 'border-primary bg-primary/5' : ''}`}
                          onClick={() => handleMemorySelect(memoryId)}
                        >
                          <p className="font-medium truncate">{memoryId}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
              
              {/* Activation explanation */}
              <div>
                {!selectedMemoryId ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>Memory Activation Details</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center p-4 text-muted-foreground">
                        <p>Select a memory to see activation details.</p>
                      </div>
                    </CardContent>
                  </Card>
                ) : (
                  <ActivationExplanationView 
                    activationData={activationExplanationQuery.data?.explanation as ExplainActivationData | undefined} 
                    memoryId={selectedMemoryId}
                    isLoading={activationExplanationQuery.isLoading} 
                    isError={activationExplanationQuery.isError} 
                    error={activationExplanationQuery.error as Error}
                  />
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
