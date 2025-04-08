import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ReconciledMergeLogEntry } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';
import { AlertCircle, Info } from 'lucide-react';

interface MergeLogViewProps {
  entries: ReconciledMergeLogEntry[] | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function MergeLogView({ entries, isLoading, isError, error }: MergeLogViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
          <CardDescription>Historical record of assembly merge operations</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
          <CardDescription>Historical record of assembly merge operations</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Failed to load merge log data</AlertTitle>
            <AlertDescription>
              {error?.message || 'An error occurred while loading the merge log data'}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!entries || entries.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
          <CardDescription>Historical record of assembly merge operations</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>No Merge Events</AlertTitle>
            <AlertDescription>
              No assembly merge events have been recorded yet. Merge events will appear here when assemblies are combined.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  // Helper function to get status badge variant
  const getStatusBadgeVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'default';
      case 'pending':
        return 'secondary';
      case 'failed':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  // Helper function for similarity display
  const formatSimilarity = (value: number | null | undefined) => {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>Merge Activity Log</CardTitle>
            <CardDescription>Historical record of assembly merge operations</CardDescription>
          </div>
          <Badge variant="outline">{entries.length} events</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b text-xs text-muted-foreground">
                <th className="text-left py-2 font-medium">Event ID</th>
                <th className="text-left py-2 font-medium">Timestamp</th>
                <th className="text-left py-2 font-medium">Source Assemblies</th>
                <th className="text-left py-2 font-medium">Target Assembly</th>
                <th className="text-left py-2 font-medium">Similarity</th>
                <th className="text-left py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {entries.map(entry => {
                return (
                  <tr key={entry.merge_event_id} className="hover:bg-muted/50 text-sm">
                    <td className="py-3 font-mono text-xs">
                      {entry.merge_event_id.substring(0, 8)}...
                    </td>
                    <td className="py-3">
                      <span title={new Date(entry.creation_timestamp).toLocaleString()}>
                        {formatTimeAgo(entry.creation_timestamp)}
                      </span>
                    </td>
                    <td className="py-3">
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {entry.source_assembly_ids?.map(id => (
                          <Badge key={id} variant="secondary" className="text-xs font-mono">
                            {id.substring(0, 6)}...
                          </Badge>
                        )) || 'N/A'}
                      </div>
                    </td>
                    <td className="py-3 font-mono text-xs">
                      {entry.target_assembly_id ? `${entry.target_assembly_id.substring(0, 8)}...` : 'N/A'}
                    </td>
                    <td className="py-3">
                      {formatSimilarity(entry.similarity_at_merge)}
                      {entry.merge_threshold && (
                        <span className="text-xs text-muted-foreground ml-1">
                          (threshold: {formatSimilarity(entry.merge_threshold)})
                        </span>
                      )}
                    </td>
                    <td className="py-3">
                      <Badge 
                        variant={getStatusBadgeVariant(entry.final_cleanup_status)}
                        className="text-xs capitalize"
                      >
                        {entry.final_cleanup_status}
                      </Badge>
                      {entry.cleanup_error && (
                        <span className="block text-xs text-destructive mt-1" title={entry.cleanup_error}>
                          Error: {entry.cleanup_error.substring(0, 20)}...
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
