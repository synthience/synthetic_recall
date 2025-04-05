import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { ExplainActivationData } from '@shared/schema';

interface ActivationExplanationViewProps {
  activationData: ExplainActivationData | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

export function ActivationExplanationView({ activationData, isLoading, isError, error }: ActivationExplanationViewProps) {
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
  const similarityPercentage = Math.min(
    100,
    Math.max(0, (activationData.similarity_score / activationData.activation_threshold) * 100)
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Memory Activation Explanation</span>
          {activationData.activated ? (
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
            <p className="font-mono">{activationData.memory_id}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Assembly ID</h3>
            <p className="font-mono">{activationData.assembly_id}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground">Activation Context</h3>
            <p className="text-sm mt-1 bg-muted p-2 rounded whitespace-pre-wrap">
              {activationData.activation_query_or_context}
            </p>
          </div>

          <div>
            <div className="flex justify-between mb-1">
              <h3 className="text-sm font-medium text-muted-foreground">Similarity Score</h3>
              <span className="text-sm">{activationData.similarity_score.toFixed(4)} / {activationData.activation_threshold.toFixed(4)}</span>
            </div>
            <Progress 
              value={similarityPercentage} 
              className={`h-2 ${activationData.activated ? 'bg-muted' : 'bg-muted'}`}
            />
            <div className={`h-1 mt-1 rounded-full ${activationData.activated ? 'bg-green-500' : 'bg-amber-500'}`} style={{ width: `${similarityPercentage}%` }}></div>
            <p className="text-xs mt-1 text-muted-foreground">
              {activationData.activated
                ? `Exceeded threshold by ${((activationData.similarity_score / activationData.activation_threshold - 1) * 100).toFixed(1)}%`
                : `${(100 - similarityPercentage).toFixed(1)}% below activation threshold`
              }
            </p>
          </div>

          {activationData.notes && (
            <div>
              <h3 className="text-sm font-medium text-muted-foreground">Additional Notes</h3>
              <p className="text-sm mt-1 italic">{activationData.notes}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
