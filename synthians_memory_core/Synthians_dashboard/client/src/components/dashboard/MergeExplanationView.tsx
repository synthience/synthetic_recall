import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ExplainMergeData, ExplainMergeEmpty } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface MergeExplanationViewProps {
  explanation: ExplainMergeData | ExplainMergeEmpty | null;
}

export function MergeExplanationView({ explanation }: MergeExplanationViewProps) {
  if (!explanation) {
    return (
      <div className="text-center py-8 text-gray-400">
        <i className="fas fa-info-circle mr-2"></i>
        No merge explanation data available
      </div>
    );
  }

  // Check if this is an empty explanation (not a merged assembly)
  if ('notes' in explanation && !('source_assembly_ids' in explanation)) {
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
              {explanation.notes || "This assembly was created directly, not through a merge operation."}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }
  
  // At this point TypeScript knows explanation has the ExplainMergeData shape
  const explanationDetailed = explanation as ExplainMergeData;

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
              <p>{explanationDetailed.merge_timestamp ? 
                 formatTimeAgo(explanationDetailed.merge_timestamp) : 
                 'Unknown'}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Similarity Threshold</h3>
              <p className="font-mono">{explanationDetailed.merge_threshold?.toFixed(4) || 'Not available'}</p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Source Assemblies</h3>
            <div className="bg-muted/30 p-3 rounded-md">
              <div className="flex flex-wrap gap-2">
                {explanationDetailed.source_assembly_ids.map(id => (
                  <Badge key={id} variant="secondary" className="font-mono">{id.substring(0, 8)}...</Badge>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {explanationDetailed.source_assembly_ids.length} source {explanationDetailed.source_assembly_ids.length === 1 ? 'assembly' : 'assemblies'} merged
              </p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Process Status</h3>
            <div className="bg-muted/30 p-3 rounded-md">
              <div className="flex items-center">
                <span className="mr-2">Cleanup:</span>
                {explanationDetailed.cleanup_status === 'completed' && (
                  <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
                    <i className="fas fa-check mr-1"></i> Completed
                  </Badge>
                )}
                {explanationDetailed.cleanup_status === 'pending' && (
                  <Badge variant="outline" className="bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100">
                    <i className="fas fa-clock mr-1"></i> Pending
                  </Badge>
                )}
                {explanationDetailed.cleanup_status === 'failed' && (
                  <Badge variant="destructive">
                    <i className="fas fa-exclamation-triangle mr-1"></i> Failed
                  </Badge>
                )}
              </div>

              {explanationDetailed.cleanup_status === 'failed' && explanationDetailed.cleanup_error && (
                <div className="mt-3">
                  <Alert variant="destructive">
                    <AlertTitle>Error Details</AlertTitle>
                    <AlertDescription className="font-mono text-sm break-all">
                      {explanationDetailed.cleanup_error}
                    </AlertDescription>
                  </Alert>
                </div>
              )}
            </div>
          </div>
          
          {explanationDetailed.notes && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Notes</h3>
              <p className="text-sm italic bg-muted/30 p-3 rounded-md">{explanationDetailed.notes}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
