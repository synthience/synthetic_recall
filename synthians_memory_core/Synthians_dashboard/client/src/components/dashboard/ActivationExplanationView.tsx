import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { ExplainActivationData } from '@shared/schema';
import { formatTimeAgo } from '@/lib/utils';

interface ActivationExplanationViewProps {
  activationData: ExplainActivationData | undefined;
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

  if (isError || !activationData) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Memory Activation Explanation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 text-center">
            <p className="text-red-500">
              {error?.message || 'Failed to load activation explanation data'}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate how close the similarity is to the threshold as a percentage
  const similarityPercentage = activationData.calculated_similarity != null && 
                               activationData.activation_threshold != null ? 
                               Math.min(
                                 100,
                                 Math.max(0, (activationData.calculated_similarity / activationData.activation_threshold) * 100)
                               ) : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Memory Activation Explanation</span>
          {activationData.passed_threshold ? (
            <Badge className="bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100">
              Activated
            </Badge>
          ) : (
            <Badge variant="secondary">
              Not Activated
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Memory ID</h3>
            <p className="font-mono">{memoryId}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Assembly ID</h3>
            <p className="font-mono">{activationData.assembly_id}</p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Check Time</h3>
            <p>{new Date(activationData.check_timestamp).toLocaleString()} ({formatTimeAgo(activationData.check_timestamp)})</p>
          </div>

          {activationData.trigger_context && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Trigger Context</h3>
              <p className="text-sm mt-1 bg-muted p-2 rounded whitespace-pre-wrap">
                {activationData.trigger_context}
              </p>
            </div>
          )}

          <div>
            <div className="flex justify-between mb-1">
              <h3 className="text-sm font-medium text-muted-foreground">Similarity Score</h3>
              <span className="text-sm">
                {activationData.calculated_similarity != null ? activationData.calculated_similarity.toFixed(4) : 'N/A'} / 
                {activationData.activation_threshold != null ? activationData.activation_threshold.toFixed(4) : 'N/A'}
              </span>
            </div>
            <Progress 
              value={similarityPercentage} 
              className="h-2 bg-muted"
            />
            <div 
              className={`h-1 mt-1 rounded-full ${activationData.passed_threshold ? 'bg-green-500' : 'bg-amber-500'}`} 
              style={{ width: `${similarityPercentage}%` }}
            ></div>
            <p className="text-xs mt-1 text-muted-foreground">
              {activationData.calculated_similarity != null && activationData.activation_threshold != null && (
                activationData.passed_threshold
                  ? `Exceeded threshold by ${((activationData.calculated_similarity / activationData.activation_threshold - 1) * 100).toFixed(1)}%`
                  : `${(100 - similarityPercentage).toFixed(1)}% below activation threshold`
              )}
            </p>
          </div>

          {activationData.notes && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Notes</h3>
              <p className="text-sm italic">{activationData.notes}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
