import React from "react";
import { ServiceStatus } from "@shared/schema";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ServiceStatus as ServiceStatusComponent } from "../layout/ServiceStatus";
import { Skeleton } from "@/components/ui/skeleton";

interface OverviewCardProps {
  title: string;
  icon: string;
  service: ServiceStatus | null;
  metrics: Record<string, string | number> | null;
  isLoading: boolean;
}

export function OverviewCard({ title, icon, service, metrics, isLoading }: OverviewCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <div className="flex items-center">
          <i className={`fas fa-${icon} text-secondary mr-2`}></i>
          <CardTitle className="font-medium text-base">{title}</CardTitle>
        </div>
        {isLoading ? (
          <Skeleton className="w-16 h-5" />
        ) : service ? (
          <ServiceStatusComponent service={service} />
        ) : (
          <div className="text-xs text-destructive">
            <i className="fas fa-exclamation-circle mr-1"></i>
            <span>Unreachable</span>
          </div>
        )}
      </CardHeader>
      <CardContent className="p-4">
        {isLoading ? (
          <div className="grid grid-cols-2 gap-4 mb-4">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
        ) : metrics ? (
          <div className="grid grid-cols-2 gap-4 mb-4">
            {Object.entries(metrics).map(([key, value], index) => (
              <div key={index} className="bg-muted p-3 rounded-md">
                <div className="text-xs text-gray-500 mb-1">{key}</div>
                <div className="text-lg font-mono">{value}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-4 text-center text-sm text-gray-400">
            <i className="fas fa-exclamation-circle mr-2"></i>
            No metrics available
          </div>
        )}
        
        {service && (
          <div className="text-xs text-gray-400 flex justify-between">
            {service.uptime && <span>Uptime: {service.uptime}</span>}
            {service.version && <span>Version: {service.version}</span>}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
