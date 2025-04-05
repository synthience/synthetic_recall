import React from 'react';
import { cn } from '@/lib/utils';

interface StatusIndicatorProps {
  status: 'healthy' | 'warning' | 'error' | 'checking';
  pulsing?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

export function StatusIndicator({ 
  status, 
  pulsing = true, 
  size = 'sm',
  label
}: StatusIndicatorProps) {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };

  const statusClasses = {
    healthy: 'bg-secondary',
    warning: 'bg-yellow-400',
    error: 'bg-destructive',
    checking: 'bg-gray-400'
  };

  const statusTextClasses = {
    healthy: 'text-secondary',
    warning: 'text-yellow-400',
    error: 'text-destructive',
    checking: 'text-gray-400'
  };

  return (
    <div className="flex items-center">
      <div 
        className={cn(
          "rounded-full mr-1",
          sizeClasses[size],
          statusClasses[status],
          pulsing && "pulse"
        )}
      ></div>
      {label && (
        <span className={cn("text-xs", statusTextClasses[status])}>
          {label}
        </span>
      )}
    </div>
  );
}
