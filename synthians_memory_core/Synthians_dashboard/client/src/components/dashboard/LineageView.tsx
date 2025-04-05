import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
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
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-5/6" />
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
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load lineage data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (lineage.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Lineage</CardTitle>
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
          <Badge variant="outline">{lineage.length} generations</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4 max-h-80 overflow-y-auto pr-2">
          {lineage.map((entry, index) => (
            <div key={entry.assembly_id} className="border rounded-md p-3 relative">
              {/* Connector lines */}
              {index < lineage.length - 1 && (
                <div className="absolute h-10 w-0.5 bg-border left-5 top-full"></div>
              )}
              
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold">
                    {entry.depth !== undefined && (
                      <Badge variant="outline" className="mr-2">Level {entry.depth}</Badge>
                    )}
                    {entry.name ? `${entry.name} (${entry.assembly_id})` : entry.assembly_id}
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Created: {entry.created_at ? formatTimeAgo(entry.created_at) : 'Unknown'}
                  </p>
                  {entry.status && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Status: <Badge variant="outline">{entry.status}</Badge>
                    </p>
                  )}
                </div>
                {entry.memory_count !== undefined && entry.memory_count !== null && (
                  <Badge variant="secondary">{entry.memory_count} memories</Badge>
                )}
              </div>
              
              {entry.parent_ids && entry.parent_ids.length > 0 && (
                <div className="mt-2">
                  <p className="text-xs text-muted-foreground mb-1">Merged from:</p>
                  <div className="flex flex-wrap gap-1">
                    {entry.parent_ids.map((sourceId: string) => (
                      <Badge key={sourceId} variant="secondary" className="text-xs">
                        {sourceId}
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
