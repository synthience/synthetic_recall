import React from "react";
import { Link } from "wouter";
import { Alert } from "@shared/schema";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

interface DiagnosticAlertsProps {
  alerts: Alert[] | null;
  isLoading: boolean;
}

export function DiagnosticAlerts({ alerts, isLoading }: DiagnosticAlertsProps) {
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error':
        return 'fa-exclamation-circle';
      case 'warning':
        return 'fa-exclamation-triangle';
      case 'info':
      default:
        return 'fa-info-circle';
    }
  };

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'text-destructive';
      case 'warning':
        return 'text-primary';
      case 'info':
      default:
        return 'text-secondary';
    }
  };
  
  const getActionColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'text-destructive';
      case 'warning':
        return 'text-primary';
      case 'info':
      default:
        return 'text-secondary';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const date = new Date(timestamp);
    const diffMs = now.getTime() - date.getTime();
    const diffMin = Math.floor(diffMs / 60000);
    
    if (diffMin < 60) {
      return `${diffMin} minute${diffMin === 1 ? '' : 's'} ago`;
    } else if (diffMin < 1440) {
      const hours = Math.floor(diffMin / 60);
      return `${hours} hour${hours === 1 ? '' : 's'} ago`;
    } else {
      const days = Math.floor(diffMin / 1440);
      return `${days} day${days === 1 ? '' : 's'} ago`;
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Recent Diagnostic Alerts</h3>
        <Link href="/logs">
          <Button variant="outline" size="sm" className="text-xs">
            View All
          </Button>
        </Link>
      </div>
      
      <div className="space-y-3">
        {isLoading ? (
          Array(3).fill(0).map((_, index) => (
            <div key={index} className="p-3 bg-card rounded-lg border border-border">
              <div className="flex">
                <Skeleton className="h-5 w-5 rounded-full mr-3" />
                <div className="flex-1">
                  <Skeleton className="h-4 w-48 mb-2" />
                  <Skeleton className="h-3 w-60 mb-3" />
                  <div className="flex justify-between">
                    <Skeleton className="h-3 w-20" />
                    <Skeleton className="h-3 w-16" />
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : alerts && alerts.length > 0 ? (
          alerts.map((alert) => (
            <div 
              key={alert.id} 
              className="p-3 bg-card rounded-lg border border-border hover:border-primary flex items-start"
            >
              <div className={`${getAlertColor(alert.type)} mr-3 mt-0.5`}>
                <i className={`fas ${getAlertIcon(alert.type)}`}></i>
              </div>
              <div className="flex-1">
                <h4 className="text-sm font-medium mb-1">{alert.title}</h4>
                <p className="text-xs text-gray-400 mb-2">{alert.description}</p>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-500">{formatTimeAgo(alert.timestamp)}</span>
                  {alert.action && (
                    <button className={`text-xs ${getActionColor(alert.type)}`}>
                      {alert.action}
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="p-4 text-center text-sm text-gray-400">
            <i className="fas fa-info-circle mr-2"></i>
            No diagnostic alerts to display
          </div>
        )}
      </div>
    </div>
  );
}
