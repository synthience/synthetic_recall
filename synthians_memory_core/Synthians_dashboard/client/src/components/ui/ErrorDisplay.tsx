import React from 'react';
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertCircle, RotateCw } from 'lucide-react';

interface ErrorDisplayProps {
  error: Error | null;
  refetch?: () => void; // Optional refetch function
  title?: string;
  message?: string; // Optional override message
}

export function ErrorDisplay({
  error,
  refetch,
  title = "Error Loading Data",
  message
}: ErrorDisplayProps) {
  const errorMessage = message || error?.message || "An unknown error occurred.";

  return (
    <Alert variant="destructive">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        {errorMessage}
        {refetch && (
          <div className="mt-3">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => refetch()}
              className="flex items-center gap-1"
            >
              <RotateCw className="h-3 w-3" /> Retry
            </Button>
          </div>
        )}
      </AlertDescription>
    </Alert>
  );
}
