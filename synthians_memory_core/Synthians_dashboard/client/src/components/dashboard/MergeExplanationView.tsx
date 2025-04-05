import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { ExplainMergeData } from '@shared/schema';
import { formatDistanceToNow } from 'date-fns';

interface MergeExplanationViewProps {
  mergeData: ExplainMergeData | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function MergeExplanationView({ mergeData, isLoading, isError, error }: MergeExplanationViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Merge Explanation</CardTitle>
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

  if (isError || !mergeData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Merge Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load merge explanation data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Format the timestamp
  const formattedTime = mergeData.merge_timestamp ? 
    `${new Date(mergeData.merge_timestamp).toLocaleString()} (${formatDistanceToNow(new Date(mergeData.merge_timestamp), { addSuffix: true })})` : 
    'Unknown';

  return (
    <Card>
      <CardHeader>
        <CardTitle>Assembly Merge Explanation</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Assembly ID</h3>
            <p className="font-mono">{mergeData.assembly_id}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Merge Timestamp</h3>
            <p>{formattedTime}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Similarity Threshold</h3>
            <p>{mergeData.similarity_threshold.toFixed(4)}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Source Assemblies</h3>
            <div className="flex flex-wrap gap-2 mt-1">
              {mergeData.source_assembly_ids.map(id => (
                <Badge key={id} variant="outline" className="font-mono">{id}</Badge>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Cleanup Status</h3>
            <div className="mt-1">
              {mergeData.cleanup_status === 'completed' && (
                <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                  Completed
                </Badge>
              )}
              {mergeData.cleanup_status === 'pending' && (
                <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100">
                  Pending
                </Badge>
              )}
              {mergeData.cleanup_status === 'failed' && (
                <Badge variant="destructive">
                  Failed
                </Badge>
              )}
            </div>
          </div>

          {mergeData.cleanup_status === 'failed' && mergeData.error && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Error Details</h3>
              <p className="text-red-500 text-sm mt-1 font-mono bg-red-50 dark:bg-red-900/20 p-2 rounded">
                {mergeData.error}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
