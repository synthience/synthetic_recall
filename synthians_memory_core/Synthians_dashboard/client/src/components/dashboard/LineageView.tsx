import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { LineageEntry } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface LineageViewProps {
  lineage: LineageEntry[] | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function LineageView({ lineage, isLoading, isError, error }: LineageViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
          <CardDescription>Showing the ancestry and merge history</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError || !lineage) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
          <CardDescription>Showing the ancestry and merge history</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error?.message || 'Failed to load lineage data'}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (lineage.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
          <CardDescription>Showing the ancestry and merge history</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center italic text-muted-foreground">
            <p>This assembly has no ancestry. It wasn't formed by a merge operation.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Assembly Lineage</span>
          <Badge variant="outline">{lineage.length} generation{lineage.length !== 1 ? 's' : ''}</Badge>
        </CardTitle>
        <CardDescription>Showing the ancestry and merge history of this assembly</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="relative space-y-4 max-h-80 overflow-y-auto pr-2 pt-2 pb-2">
          {/* Vertical line connecting all nodes */}
          <div className="absolute top-0 bottom-0 left-5 w-0.5 bg-border z-0"></div>
          
          {lineage.map((entry, index) => (
            <div key={entry.assembly_id} className="border rounded-md p-3 relative z-10 bg-background">
              {/* Circle connector for the vertical line */}
              <div className="absolute w-3 h-3 rounded-full bg-primary border-2 border-background left-3.5 top-1/2 transform -translate-x-1/2 -translate-y-1/2 -ml-px"></div>
              
              <div className="flex items-start justify-between pl-6">
                <div>
                  <h4 className="font-semibold flex items-center">
                    {entry.depth !== undefined && (
                      <Badge variant="outline" className="mr-2">Level {entry.depth}</Badge>
                    )}
                    <span className="font-mono text-secondary">{entry.assembly_id.substring(0, 8)}...</span>
                    {entry.name && <span className="ml-2">{entry.name}</span>}
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Created: {entry.created_at ? formatTimeAgo(entry.created_at) : 'Unknown'}
                  </p>
                  {entry.status && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Status: 
                      <Badge variant="outline" className={
                        entry.status === 'active' ? 'bg-green-100 text-green-600 dark:bg-green-900/20 dark:text-green-400' :
                        'bg-blue-100 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400'
                      }>
                        {entry.status}
                      </Badge>
                    </p>
                  )}
                </div>
                {entry.memory_count !== undefined && entry.memory_count !== null && (
                  <Badge variant="secondary">{entry.memory_count.toLocaleString()} memories</Badge>
                )}
              </div>
              
              {entry.parent_ids && entry.parent_ids.length > 0 && (
                <div className="mt-3 pl-6">
                  <p className="text-xs text-muted-foreground mb-1">Merged from:</p>
                  <div className="flex flex-wrap gap-1">
                    {entry.parent_ids.map((sourceId: string) => (
                      <Badge key={sourceId} variant="secondary" className="text-xs font-mono">
                        {sourceId.substring(0, 8)}...
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
