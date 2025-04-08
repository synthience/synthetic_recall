import React from 'react';
import { UseQueryResult } from '@tanstack/react-query';
import { ApiResponse } from '@shared/schema';
import { Skeleton } from '@/components/ui/skeleton';
import { ErrorDisplay } from '@/components/ui/ErrorDisplay';
import { formatTimeAgo } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { RefreshCw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

interface ExplainabilityDataViewProps<TData> {
  hookResult: UseQueryResult<ApiResponse<TData>, Error>;
  RenderComponent: React.ComponentType<{ data: TData }>; // Pass the specific component to render
  skeletonHeight?: string; // e.g., 'h-64'
  title?: string; // Optional title for context
}

export function ExplainabilityDataView<TData>(
  {
    hookResult,
    RenderComponent,
    skeletonHeight = 'h-64',
    title
  }: ExplainabilityDataViewProps<TData>
) {
  const { data: apiResponse, isLoading, isFetching, isError, error, refetch, dataUpdatedAt } = hookResult;

  // Combine isLoading and isFetching for a smoother initial load experience
  const showLoading = isLoading || (isFetching && !apiResponse);

  return (
    <Card>
      {title && (
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="text-lg font-medium">{title}</CardTitle>
          {dataUpdatedAt && !showLoading && (
            <div className="text-xs text-muted-foreground flex items-center gap-1">
              <span>Updated: {formatTimeAgo(new Date(dataUpdatedAt).toISOString())}</span>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-5 w-5" 
                onClick={() => refetch()} 
                disabled={isFetching}
              >
                <RefreshCw className={`h-3 w-3 ${isFetching ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          )}
        </CardHeader>
      )}
      <CardContent className="pt-4">
        {showLoading ? (
          <Skeleton className={`${skeletonHeight} w-full`} />
        ) : isError ? (
          <ErrorDisplay error={error} refetch={refetch} />
        ) : !apiResponse?.success ? (
          <ErrorDisplay error={new Error(apiResponse?.error || 'API request failed')} refetch={refetch} />
        ) : !apiResponse.data ? (
          <div className="text-center py-8 text-muted-foreground">No data available.</div>
        ) : (
          <RenderComponent data={apiResponse.data} />
        )}
      </CardContent>
    </Card>
  );
}
