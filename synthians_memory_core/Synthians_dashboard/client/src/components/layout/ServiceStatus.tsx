import React from "react";
import { ServiceStatus as ServiceStatusType } from "@shared/schema";
import { cn } from "@/lib/utils";

interface ServiceStatusProps {
  service: ServiceStatusType;
}

export function ServiceStatus({ service }: ServiceStatusProps) {
  // Determine status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Healthy':
        return 'text-secondary';
      case 'Unhealthy':
      case 'Error':
        return 'text-destructive';
      case 'Checking...':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Healthy':
        return 'fa-check-circle';
      case 'Unhealthy':
      case 'Error':
        return 'fa-exclamation-circle';
      case 'Checking...':
        return 'fa-spinner fa-spin';
      default:
        return 'fa-question-circle';
    }
  };

  const statusColor = getStatusColor(service.status);
  const statusIcon = getStatusIcon(service.status);

  return (
    <div className="flex items-center">
      <div className={cn("w-2 h-2 rounded-full mr-1", {
        "bg-secondary pulse": service.status === 'Healthy',
        "bg-destructive pulse": service.status === 'Unhealthy' || service.status === 'Error',
        "bg-yellow-400 pulse": service.status === 'Checking...'
      })}></div>
      <span className={`text-xs ${statusColor}`}>
        <i className={`fas ${statusIcon} mr-1`}></i>
        {service.status}
      </span>
    </div>
  );
}
