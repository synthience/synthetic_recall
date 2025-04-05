import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface RefreshButtonProps {
  onClick: () => void;
  className?: string;
}

export function RefreshButton({ onClick, className }: RefreshButtonProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = () => {
    setIsRefreshing(true);
    onClick();
    
    // Reset the animation after a short delay
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1000);
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={handleRefresh}
      className={cn(className)}
      disabled={isRefreshing}
    >
      <i className={cn(
        "fas fa-sync-alt",
        isRefreshing && "animate-spin"
      )}></i>
    </Button>
  );
}
