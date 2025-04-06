import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ReconciledMergeLogEntry } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface MergeLogViewProps {
  entries: ReconciledMergeLogEntry[] | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function MergeLogView({ entries, isLoading, isError, error }: MergeLogViewProps) {
  if (isLoading) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
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

  if (isError || !entries) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load merge log data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (entries.length === 0) {
    return (
      <Card className="col-span-full">
        <CardHeader>
          <CardTitle>Merge Activity Log</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center italic text-muted-foreground">
            <p>No merge events have been recorded yet.</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="col-span-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Merge Activity Log</span>
          <Badge variant="outline">{entries.length} events</Badge>
        </CardTitle>
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
                <th className="text-left py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {entries.map(entry => {
                return (
                  <tr key={entry.merge_event_id} className="hover:bg-muted/50 text-sm">
                    <td className="py-3 font-mono">
                      {entry.merge_event_id.substring(0, 8)}...
                    </td>
                    <td className="py-3">
                      {formatTimeAgo(entry.creation_timestamp)}
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
                    <td className="py-3 font-mono">
                      {entry.target_assembly_id.substring(0, 8)}...
                    </td>
                    <td className="py-3">
                      {entry.final_cleanup_status === "pending" && (
                        <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100">
                          Pending
                        </Badge>
                      )}
                      {entry.final_cleanup_status === "completed" && (
                        <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                          Completed
                        </Badge>
                      )}
                      {entry.final_cleanup_status === "failed" && (
                        <Badge variant="destructive">
                          Failed
                          {entry.cleanup_error && (
                            <span className="ml-1 cursor-help" title={entry.cleanup_error}>ⓘ</span>
                          )}
                        </Badge>
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
