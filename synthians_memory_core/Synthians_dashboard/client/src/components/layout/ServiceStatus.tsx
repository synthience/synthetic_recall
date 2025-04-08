import React from "react";
import { ServiceStatus as ServiceStatusType } from "@shared/schema";
import { cn } from "@/lib/utils";

interface ServiceStatusProps {
  service: ServiceStatusType | null; // Allow null for loading/error states
}

export function ServiceStatus({ service }: ServiceStatusProps) {
  // Handle null service object gracefully
  if (!service) {
    return (
      <div className="flex items-center">
        <div className={cn("w-2 h-2 rounded-full mr-1 bg-gray-500")}></div>
        <span className={`text-xs text-gray-500`}>
          <i className={`fas fa-question-circle mr-1`}></i>
          Unknown
        </span>
      </div>
    );
  }
  
  const status = service.status || 'Unknown'; // Default to Unknown if status field is missing
  const normalizedStatus = status.toLowerCase();

  // Determine status color - normalizing input
  const getStatusColor = (normStatus: string) => {
    if (normStatus === 'healthy' || normStatus === 'ok') { // Handle 'ok' as well
      return 'text-secondary'; // Assuming secondary is green-ish
    }
    if (normStatus === 'unhealthy' || normStatus === 'error') {
      return 'text-destructive';
    }
    if (normStatus === 'checking...') {
      return 'text-yellow-400';
    }
    return 'text-gray-400'; // Default for 'Unknown' or other statuses
  };

  // Determine status icon - normalizing input
  const getStatusIcon = (normStatus: string) => {
    if (normStatus === 'healthy' || normStatus === 'ok') {
      return 'fa-check-circle';
    }
    if (normStatus === 'unhealthy' || normStatus === 'error') {
      return 'fa-exclamation-circle';
    }
    if (normStatus === 'checking...') {
      return 'fa-spinner fa-spin';
    }
    return 'fa-question-circle'; // Default
  };
  
  // Determine background pulse color - normalizing input
  const getPulseColor = (normStatus: string) => {
    if (normStatus === 'healthy' || normStatus === 'ok') return 'bg-secondary';
    if (normStatus === 'unhealthy' || normStatus === 'error') return 'bg-destructive';
    if (normStatus === 'checking...') return 'bg-yellow-400';
    return 'bg-gray-500'; // Default
  };

  const statusColor = getStatusColor(normalizedStatus);
  const statusIcon = getStatusIcon(normalizedStatus);
  const pulseColor = getPulseColor(normalizedStatus);

  return (
    <div className="flex items-center">
      <div className={cn("w-2 h-2 rounded-full mr-1", pulseColor, "pulse")}></div>
      <span className={`text-xs ${statusColor}`}>
        <i className={`fas ${statusIcon} mr-1`}></i>
        {status} 
      </span>
    </div>
  );
}
