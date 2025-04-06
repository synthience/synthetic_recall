import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface RefreshButtonProps {
  onClick: () => void;
  className?: string;
  isLoading?: boolean;
}

export function RefreshButton({ onClick, className, isLoading = false }: RefreshButtonProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    onClick();
    
    // Reset the animation after a short delay
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1000);
  };

  // Determine if the button should be spinning/disabled
  const isSpinning = isRefreshing || isLoading;

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleRefresh}
      className={cn(className)}
      disabled={isSpinning}
    >
      <i className={cn(
        "fas fa-sync-alt",
        isSpinning && "animate-spin"
      )}></i>
    </Button>
  );
}
