import React, { useState } from "react";
import { useRuntimeConfig } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { usePollingStore } from "@/lib/store";
import { useFeatures } from "@/contexts/FeaturesContext";

// Component for displaying config key-value pairs
function ConfigItem({ label, value }: { label: string, value: any }) {
  const renderValue = () => {
    if (value === null || value === undefined) return "null";
    
    if (typeof value === "object") {
      return JSON.stringify(value, null, 2);
    }
    
    return value.toString();
  };

  return (
    <div className="py-2 border-b border-border last:border-0">
      <div className="flex justify-between items-start">
        <span className="text-sm font-medium">{label}</span>
        <span className="font-mono text-sm bg-muted px-2 py-1 rounded max-w-[50%] break-all">
          {renderValue()}
        </span>
      </div>
    </div>
  );
}

export default function Config() {
  const { refreshAllData } = usePollingStore();
  const { explainabilityEnabled, isLoading: featuresLoading } = useFeatures();
  const [selectedService, setSelectedService] = useState<string>("memory-core");
  
  // Fetch runtime configuration data for the selected service
  const memoryConfig = useRuntimeConfig("memory-core");
  const neuralConfig = useRuntimeConfig("neural-memory");
  const cceConfig = useRuntimeConfig("cce");

  const handleTabChange = (value: string) => {
    setSelectedService(value);
  };

  const handleRefresh = () => {
    memoryConfig.refetch();
    neuralConfig.refetch();
    cceConfig.refetch();
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Configuration Viewer</h2>
          <p className="text-sm text-gray-400">
            Display current runtime configurations of all services
          </p>
        </div>
        <RefreshButton onClick={handleRefresh} />
      </div>
      
      <Tabs defaultValue="memory-core" className="mb-6" onValueChange={handleTabChange}>
        <TabsList className="mb-4">
          <TabsTrigger value="memory-core">Memory Core</TabsTrigger>
          <TabsTrigger value="neural-memory">Neural Memory</TabsTrigger>
          <TabsTrigger value="cce">Context Cascade Engine</TabsTrigger>
        </TabsList>
        
        <TabsContent value="memory-core">
          <Card>
            <CardHeader>
              <CardTitle>Memory Core Configuration</CardTitle>
              <CardDescription>Runtime configuration settings for the Memory Core service</CardDescription>
            </CardHeader>
            <CardContent>
              {memoryConfig.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : memoryConfig.error ? (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load Memory Core configuration
                </div>
              ) : memoryConfig.data?.config ? (
                <div>
                  {explainabilityEnabled && (
                    <div className="bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 p-3 rounded-md mb-4">
                      <h3 className="font-semibold mb-1">Explainability Features: Enabled</h3>
                      <p className="text-sm">Diagnostic and explainability features are currently active.</p>
                    </div>
                  )}

                  {!explainabilityEnabled && (
                    <div className="bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200 p-3 rounded-md mb-4">
                      <h3 className="font-semibold mb-1">Explainability Features: Disabled</h3>
                      <p className="text-sm">Set <code className="bg-muted p-1 rounded">ENABLE_EXPLAINABILITY=true</code> to activate diagnostic features.</p>
                    </div>
                  )}

                  <div className="space-y-0">
                    {Object.entries(memoryConfig.data.config).map(([key, value]) => (
                      <ConfigItem key={key} label={key} value={value} />
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  No configuration data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="neural-memory">
          <Card>
            <CardHeader>
              <CardTitle>Neural Memory Configuration</CardTitle>
              <CardDescription>Runtime configuration settings for the Neural Memory service</CardDescription>
            </CardHeader>
            <CardContent>
              {neuralConfig.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : neuralConfig.error ? (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load Neural Memory configuration
                </div>
              ) : neuralConfig.data?.config ? (
                <div className="space-y-0">
                  {Object.entries(neuralConfig.data.config).map(([key, value]) => (
                    <ConfigItem key={key} label={key} value={value} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  No configuration data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="cce">
          <Card>
            <CardHeader>
              <CardTitle>Context Cascade Engine Configuration</CardTitle>
              <CardDescription>Runtime configuration settings for the CCE service</CardDescription>
            </CardHeader>
            <CardContent>
              {cceConfig.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : cceConfig.error ? (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load CCE configuration
                </div>
              ) : cceConfig.data?.config ? (
                <div className="space-y-0">
                  {Object.entries(cceConfig.data.config).map(([key, value]) => (
                    <ConfigItem key={key} label={key} value={value} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  No configuration data available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}
