import React from "react";
import { useMemoryCoreStats, useNeuralMemoryConfig, useCCEConfig } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { usePollingStore } from "@/lib/store";

// Component for displaying config key-value pairs
function ConfigItem({ label, value }: { label: string, value: string | number | boolean }) {
  return (
    <div className="py-2 border-b border-border last:border-0">
      <div className="flex justify-between items-start">
        <span className="text-sm font-medium">{label}</span>
        <span className="font-mono text-sm bg-muted px-2 py-1 rounded max-w-[50%] break-all">
          {typeof value === "boolean" 
            ? value ? "true" : "false"
            : value.toString()}
        </span>
      </div>
    </div>
  );
}

export default function Config() {
  const { refreshAllData } = usePollingStore();
  
  // Fetch configuration data
  const memoryCoreStats = useMemoryCoreStats();
  const neuralMemoryConfig = useNeuralMemoryConfig();
  const cceConfig = useCCEConfig();
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Configuration Viewer</h2>
          <p className="text-sm text-gray-400">
            Display current runtime configurations of all services
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      <Tabs defaultValue="memory-core" className="mb-6">
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
              {memoryCoreStats.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : memoryCoreStats.data?.data ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Vector Index Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Index Type" value={memoryCoreStats.data.data.vector_index.index_type} />
                      <ConfigItem label="GPU Enabled" value={memoryCoreStats.data.data.vector_index.gpu_enabled} />
                      <ConfigItem label="Drift Threshold" value={100} />
                      <ConfigItem label="Dimension" value={1536} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Process Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Assembly Pruning" value={memoryCoreStats.data.data.assembly_stats.pruning_enabled} />
                      <ConfigItem label="Assembly Merging" value={memoryCoreStats.data.data.assembly_stats.merging_enabled} />
                      <ConfigItem label="Quick Recall Threshold" value={0.95} />
                      <ConfigItem label="Persistence Enabled" value={true} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load Memory Core configuration
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
              {neuralMemoryConfig.isLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : neuralMemoryConfig.data?.data ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Model Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Dimensions" value={neuralMemoryConfig.data.data.dimensions} />
                      <ConfigItem label="Hidden Size" value={neuralMemoryConfig.data.data.hidden_size} />
                      <ConfigItem label="Layers" value={neuralMemoryConfig.data.data.layers} />
                      <ConfigItem label="Attention Heads" value={neuralMemoryConfig.data.data.attention_heads || 12} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Training Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Learning Rate" value={neuralMemoryConfig.data.data.learning_rate || 0.0001} />
                      <ConfigItem label="Batch Size" value={neuralMemoryConfig.data.data.batch_size || 32} />
                      <ConfigItem label="Gradient Clip" value={neuralMemoryConfig.data.data.gradient_clip || 1.0} />
                      <ConfigItem label="Emotional Boost" value={neuralMemoryConfig.data.data.emotional_boost || true} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load Neural Memory configuration
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
              ) : cceConfig.data?.data ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">Variant Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="Active Variant" value={cceConfig.data.data.active_variant} />
                      <ConfigItem label="Confidence Threshold" value={cceConfig.data.data.variant_confidence_threshold} />
                      <ConfigItem label="Auto Selection" value={true} />
                      <ConfigItem label="Performance Based" value={true} />
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-sm font-medium text-primary mb-2 uppercase">LLM Configuration</h3>
                    <div className="space-y-0">
                      <ConfigItem label="LLM Guidance Enabled" value={cceConfig.data.data.llm_guidance_enabled} />
                      <ConfigItem label="Retry Attempts" value={cceConfig.data.data.retry_attempts} />
                      <ConfigItem label="Timeout (ms)" value={3000} />
                      <ConfigItem label="Cache Results" value={true} />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                  Failed to load CCE configuration
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      <Card>
        <CardHeader>
          <CardTitle>Environment Variables</CardTitle>
          <CardDescription>System environment variables affecting service behavior</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-primary mb-2 uppercase">Service URLs</h3>
              <div className="space-y-0">
                <ConfigItem label="MEMORY_CORE_URL" value={process.env.MEMORY_CORE_URL || "http://memory-core:8080"} />
                <ConfigItem label="NEURAL_MEMORY_URL" value={process.env.NEURAL_MEMORY_URL || "http://neural-memory:8080"} />
                <ConfigItem label="CCE_URL" value={process.env.CCE_URL || "http://cce:8080"} />
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-primary mb-2 uppercase">Dashboard Configuration</h3>
              <div className="space-y-0">
                <ConfigItem label="NODE_ENV" value={process.env.NODE_ENV || "production"} />
                <ConfigItem label="Default Poll Rate (ms)" value={5000} />
                <ConfigItem label="Max Visible Alerts" value={10} />
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
