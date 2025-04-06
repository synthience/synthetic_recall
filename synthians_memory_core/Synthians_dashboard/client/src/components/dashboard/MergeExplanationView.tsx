import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ExplainMergeData, ExplainMergeEmpty } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface MergeExplanationViewProps {
  mergeData: ExplainMergeData | ExplainMergeEmpty | undefined;
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
          <CardDescription>Details about how this assembly was formed</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Skeleton className="h-6 w-3/4" />
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-12 w-full" />
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
          <CardDescription>Details about how this assembly was formed</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error?.message || 'Failed to load merge explanation data'}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  // Check if this is an empty explanation (not a merged assembly)
  if ('notes' in mergeData && !('source_assembly_ids' in mergeData)) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Assembly Merge Explanation</CardTitle>
          <CardDescription>Details about how this assembly was formed</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTitle>Not a Merged Assembly</AlertTitle>
            <AlertDescription>
              {mergeData.notes || "This assembly was created directly, not through a merge operation."}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }
  
  // At this point TypeScript knows mergeData has the ExplainMergeData shape
  const mergeDataDetailed = mergeData as ExplainMergeData;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Assembly Merge Explanation</span>
          <Badge variant="outline" className="font-mono">
            Event Details
          </Badge>
        </CardTitle>
        <CardDescription>Details about how this assembly was formed through merging</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-muted/40 p-3 rounded-md">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Merge Timestamp</h3>
              <p>{mergeDataDetailed.merge_timestamp ? 
                 formatTimeAgo(mergeDataDetailed.merge_timestamp) : 
                 'Unknown'}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Similarity Threshold</h3>
              <p className="font-mono">{mergeDataDetailed.merge_threshold?.toFixed(4) || 'Not available'}</p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Source Assemblies</h3>
            <div className="bg-muted/30 p-3 rounded-md">
              <div className="flex flex-wrap gap-2">
                {mergeDataDetailed.source_assembly_ids.map(id => (
                  <Badge key={id} variant="secondary" className="font-mono">{id.substring(0, 8)}...</Badge>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {mergeDataDetailed.source_assembly_ids.length} source {mergeDataDetailed.source_assembly_ids.length === 1 ? 'assembly' : 'assemblies'} merged
              </p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Process Status</h3>
            <div className="bg-muted/30 p-3 rounded-md">
              <div className="flex items-center">
                <span className="mr-2">Cleanup:</span>
                {mergeDataDetailed.cleanup_status === 'completed' && (
                  <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                    <i className="fas fa-check mr-1"></i> Completed
                  </Badge>
                )}
                {mergeDataDetailed.cleanup_status === 'pending' && (
                  <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100">
                    <i className="fas fa-clock mr-1"></i> Pending
                  </Badge>
                )}
                {mergeDataDetailed.cleanup_status === 'failed' && (
                  <Badge variant="destructive">
                    <i className="fas fa-exclamation-triangle mr-1"></i> Failed
                  </Badge>
                )}
              </div>

              {mergeDataDetailed.cleanup_status === 'failed' && mergeDataDetailed.cleanup_error && (
                <div className="mt-3">
                  <Alert variant="destructive">
                    <AlertTitle>Error Details</AlertTitle>
                    <AlertDescription className="font-mono text-sm break-all">
                      {mergeDataDetailed.cleanup_error}
                    </AlertDescription>
                  </Alert>
                </div>
              )}
            </div>
          </div>
          
          {mergeDataDetailed.notes && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Notes</h3>
              <p className="text-sm italic bg-muted/30 p-3 rounded-md">{mergeDataDetailed.notes}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
