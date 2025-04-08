import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { useRuntimeConfig } from "@/lib/api";
import { useFeatures } from "@/contexts/FeaturesContext";
import { AlertCircle, RefreshCw } from "lucide-react";

// Helper component to display a single config item
interface ConfigItemProps {
  label: string;
  value: any;
}

function ConfigItem({ label, value }: ConfigItemProps) {
  let valueDisplay: string;
  if (value === null) {
    valueDisplay = "null";
  } else if (typeof value === 'object') {
    try {
      // Use compact spacing for objects/arrays
      valueDisplay = JSON.stringify(value, null, 1);
    } catch {
      valueDisplay = "[Unserializable Object]";
    }
  } else {
    valueDisplay = String(value);
  }

  return (
    <div className="py-3 border-b last:border-b-0 flex justify-between items-start gap-4">
      <div className="font-medium text-sm shrink-0 break-words">{label}</div>
      {/* Use pre-wrap for objects/arrays, truncate for others */}
      <pre className={`text-sm text-right text-muted-foreground ${typeof value === 'object' ? 'whitespace-pre-wrap bg-muted/50 p-1 rounded text-xs' : 'truncate'}`}>
        {valueDisplay}
      </pre>
    </div>
  );
}

// Main Page Component
export default function ConfigPage() {
  const [activeService, setActiveService] = useState<string>("memory-core");
  const { explainabilityEnabled, debugMode } = useFeatures(); // Get feature flag and debug mode

  // Fetch config based on active tab
  const { data, isLoading, isError, error, refetch } = useRuntimeConfig(activeService);

  // Determine if the response looks like health data instead of config
  const isHealthDataResponse = data?.success && data.data && !('config' in data.data) && ('status' in data.data);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="space-y-2 mt-4">
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-5/6" />
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-3/4" />
          <Skeleton className="h-6 w-full" />
        </div>
      );
    }

    if (isError || !data?.success) {
      const errorMessage = error?.message || data?.error || "Failed to load configuration data.";
      return (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error Loading Configuration</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      );
    }

    // Handle case where API returned health data instead of config
    // This indicates a backend/proxy issue, but we show something informative
    if (isHealthDataResponse) {
      return (
        <Alert className="mt-4">
          <AlertCircle className="h-4 w-4 text-blue-500" />
          <AlertTitle>Unexpected Data Format</AlertTitle>
          <AlertDescription>
            Received health status instead of configuration. The backend endpoint might not be serving the correct data. Displaying available health data:
            {debugMode && (
              <div className="mt-2 text-xs bg-muted p-2 rounded overflow-auto max-h-40">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(data.data, null, 2)}
                </pre>
              </div>
            )}
          </AlertDescription>
        </Alert>
      );
    }

    // Successfully loaded config data
    const configObject = data.data?.config;
    if (!configObject || typeof configObject !== 'object' || Object.keys(configObject).length === 0) {
      return (
        <Alert className="mt-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>No Configuration Data</AlertTitle>
          <AlertDescription>
            No runtime configuration parameters were returned for {activeService}.
          </AlertDescription>
        </Alert>
      );
    }

    return (
      <div className="space-y-0 mt-4">
        {Object.entries(configObject).map(([key, value]) => (
          <ConfigItem key={key} label={key} value={value} />
        ))}
      </div>
    );
  };

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Configuration</h1>
          <p className="text-sm text-muted-foreground mt-1">
            View sanitized runtime configuration for Synthians services.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isLoading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {!explainabilityEnabled && (
         <Alert variant="default" className="mb-4 border-blue-600 bg-blue-950/30 text-blue-200">
          <AlertCircle className="h-4 w-4 text-blue-400" />
          <AlertTitle>Explainability Features: Disabled</AlertTitle>
          <AlertDescription>
            Viewing configuration requires the 'ENABLE_EXPLAINABILITY' flag to be true in the Memory Core service. Some configurations might be hidden.
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={activeService} onValueChange={setActiveService} className="space-y-4">
        <TabsList>
          <TabsTrigger value="memory-core">Memory Core</TabsTrigger>
          <TabsTrigger value="neural-memory">Neural Memory</TabsTrigger>
          <TabsTrigger value="cce">Context Cascade</TabsTrigger>
        </TabsList>

        {/* Render content within a Card for each tab */}
        <Card>
          <CardHeader>
            <CardTitle className="capitalize">{activeService.replace('-', ' ')} Configuration</CardTitle>
            <CardDescription>
              Current runtime parameters for the {activeService} service.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {renderContent()}
          </CardContent>
        </Card>
      </Tabs>
    </>
  );
}
