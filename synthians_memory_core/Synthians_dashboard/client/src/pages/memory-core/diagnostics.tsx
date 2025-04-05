import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { useMergeLog, useRuntimeConfig } from '@/lib/api';
import { MergeLogView } from '@/components/dashboard/MergeLogView';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { RefreshCw } from 'lucide-react';
import { MergeLogEntry } from '@shared/schema';

export default function MemoryCoreDiagnostics() {
  const [selectedService, setSelectedService] = useState<string>('memory-core');
  const [logLimit, setLogLimit] = useState<number>(50);
  
  // Fetch merge log data
  const mergeLogQuery = useMergeLog(logLimit);
  
  // Fetch runtime configuration
  const configQuery = useRuntimeConfig(selectedService);
  
  // Handle refresh for both queries
  const handleRefresh = () => {
    mergeLogQuery.refetch();
    configQuery.refetch();
  };
  
  return (
    <div className="container py-6">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-semibold">Memory Core Diagnostics</h1>
          <Button variant="outline" size="sm" onClick={handleRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
        
        {/* Merge Log */}
        <MergeLogView
          entries={mergeLogQuery.data as MergeLogEntry[] | undefined}
          isLoading={mergeLogQuery.isLoading}
          isError={mergeLogQuery.isError}
          error={mergeLogQuery.error as Error}
        />
        
        {/* Runtime Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Runtime Configuration</CardTitle>
            <CardDescription>
              View current runtime configuration values for the selected service.
            </CardDescription>
            <div className="flex items-center gap-2 mt-2">
              <Select value={selectedService} onValueChange={setSelectedService}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="Select service" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="memory-core">Memory Core</SelectItem>
                  <SelectItem value="neural-memory">Neural Memory</SelectItem>
                  <SelectItem value="cce">Controlled Context Exchange</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardHeader>
          <CardContent>
            {configQuery.isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-5/6" />
              </div>
            ) : configQuery.isError ? (
              <div className="p-4 text-center">
                <p className="text-red-500">
                  {configQuery.error?.message || 'Failed to load configuration data'}
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b text-xs text-muted-foreground">
                      <th className="text-left py-2 font-medium">Parameter</th>
                      <th className="text-left py-2 font-medium">Value</th>
                      <th className="text-left py-2 font-medium">Type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {Object.entries(configQuery.data || {}).map(([key, value]) => (
                      <tr key={key} className="hover:bg-muted/50">
                        <td className="py-2 font-mono text-sm">{key}</td>
                        <td className="py-2 font-mono text-sm">
                          {typeof value === 'object' 
                            ? JSON.stringify(value)
                            : String(value)}
                        </td>
                        <td className="py-2 text-sm text-muted-foreground">
                          {Array.isArray(value) 
                            ? 'array' 
                            : typeof value === 'object' && value !== null 
                              ? 'object' 
                              : typeof value}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
