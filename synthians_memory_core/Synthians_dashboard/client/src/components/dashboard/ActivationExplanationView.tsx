import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ExplainActivationData, ExplainActivationEmpty } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface ActivationExplanationViewProps {
  activationData: ExplainActivationData | ExplainActivationEmpty | undefined;
  memoryId: string;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function ActivationExplanationView({ activationData, memoryId, isLoading, isError, error }: ActivationExplanationViewProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Memory Activation Explanation</CardTitle>
          <CardDescription>Details about how this memory was activated</CardDescription>
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

  if (isError || !activationData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Memory Activation Explanation</CardTitle>
          <CardDescription>Details about how this memory was activated</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>
              {error?.message || 'Failed to load activation explanation data'}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  // Check if this is an empty explanation (no activation record available)
  if ('notes' in activationData && !('check_timestamp' in activationData)) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Memory Activation Explanation</CardTitle>
          <CardDescription>Details about how this memory was activated</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTitle>No Activation Record</AlertTitle>
            <AlertDescription>
              {activationData.notes || "No activation record found for this memory in this assembly."}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }
  
  // At this point TypeScript knows activationData has the ExplainActivationData shape
  const activationDataDetailed = activationData as ExplainActivationData;

  // Calculate how close the similarity is to the threshold as a percentage
  const similarityPercentage = activationDataDetailed.calculated_similarity != null && 
                             activationDataDetailed.activation_threshold != null ? 
                             Math.min(
                               100,
                               Math.max(0, (activationDataDetailed.calculated_similarity / activationDataDetailed.activation_threshold) * 100)
                             ) : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Memory Activation Explanation</span>
          {activationDataDetailed.passed_threshold ? (
            <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
              <i className="fas fa-check-circle mr-1"></i> Activated
            </Badge>
          ) : (
            <Badge variant="secondary">
              <i className="fas fa-times-circle mr-1"></i> Not Activated
            </Badge>
          )}
        </CardTitle>
        <CardDescription>Analysis of memory activation during retrieval</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-muted/40 p-3 rounded-md">
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Memory ID</h3>
              <p className="font-mono text-xs break-all">{memoryId}</p>
            </div>

            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Check Time</h3>
              <p>{formatTimeAgo(activationDataDetailed.check_timestamp)}</p>
            </div>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Similarity Analysis</h3>
            <div className="bg-muted/30 p-3 rounded-md">
              <div className="flex justify-between mb-1">
                <span>Score vs Threshold</span>
                <span className="font-mono">
                  {activationDataDetailed.calculated_similarity != null ? activationDataDetailed.calculated_similarity.toFixed(4) : 'N/A'} / 
                  {activationDataDetailed.activation_threshold != null ? activationDataDetailed.activation_threshold.toFixed(4) : 'N/A'}
                </span>
              </div>
              <div className="relative pt-1">
                <Progress 
                  value={similarityPercentage} 
                  className="h-2 bg-muted"
                />
                <div className="absolute top-0 left-0 right-0 flex justify-between">
                  <span className="text-xs text-muted-foreground">0</span>
                  <span className="text-xs text-muted-foreground">
                    Threshold: {activationDataDetailed.activation_threshold?.toFixed(2)}
                  </span>
                  <span className="text-xs text-muted-foreground">1.0</span>
                </div>
              </div>
              
              <div className="mt-3 bg-muted/50 p-2 rounded-md">
                <p className="text-sm">
                  {activationDataDetailed.calculated_similarity != null && activationDataDetailed.activation_threshold != null && (
                    activationDataDetailed.passed_threshold
                      ? <span className="text-green-600 dark:text-green-400">
                          <i className="fas fa-arrow-up mr-1"></i>
                          Exceeded threshold by {((activationDataDetailed.calculated_similarity / activationDataDetailed.activation_threshold - 1) * 100).toFixed(1)}%
                        </span>
                      : <span className="text-amber-600 dark:text-amber-400">
                          <i className="fas fa-arrow-down mr-1"></i>
                          {(100 - similarityPercentage).toFixed(1)}% below activation threshold
                        </span>
                  )}
                </p>
              </div>
            </div>
          </div>

          {activationDataDetailed.trigger_context && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Trigger Context</h3>
              <div className="bg-muted/30 p-3 rounded-md">
                <p className="text-sm whitespace-pre-wrap">
                  {activationDataDetailed.trigger_context}
                </p>
              </div>
            </div>
          )}

          {activationDataDetailed.notes && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground mb-2">Notes</h3>
              <p className="text-sm italic bg-muted/30 p-3 rounded-md">{activationDataDetailed.notes}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
