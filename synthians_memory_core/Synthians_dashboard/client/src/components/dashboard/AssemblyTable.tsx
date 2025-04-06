import React from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Link } from "wouter";
import { formatTimeAgo } from "@/lib/utils";

interface Assembly {
  id: string;
  name: string;
  member_count: number;
  updated_at: string;
  vector_index_updated_at?: string;
}

interface AssemblyTableProps {
  assemblies: Assembly[] | null;
  isLoading: boolean;
  isError?: boolean;
  error?: Error | null;
  title?: string;
  showFilters?: boolean;
}

export function AssemblyTable({
  assemblies,
  isLoading,
  isError = false,
  error = null,
  title = "Assemblies",
  showFilters = true
}: AssemblyTableProps) {
  // Helper function to get sync status
  const getSyncStatus = (assembly: Assembly) => {
    if (!assembly.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-500 dark:text-yellow-400",
        bgColor: "bg-yellow-100 dark:bg-yellow-900/20",
        icon: "fas fa-clock"
      };
    }
    
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    
    if (vectorDate >= updateDate) {
      return {
        label: "Indexed",
        color: "text-green-600 dark:text-green-400",
        bgColor: "bg-green-100 dark:bg-green-900/20",
        icon: "fas fa-check"
      };
    }
    
    return {
      label: "Syncing",
      color: "text-blue-600 dark:text-blue-400",
      bgColor: "bg-blue-100 dark:bg-blue-900/20",
      icon: "fas fa-sync-alt"
    };
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <div className="flex items-center">
          <CardTitle className="font-medium">{title}</CardTitle>
          {!isLoading && assemblies && (
            <Badge variant="outline" className="ml-2">
              {assemblies.length} {assemblies.length === 1 ? 'assembly' : 'assemblies'}
            </Badge>
          )}
        </div>
        
        {showFilters && (
          <div className="flex space-x-2">
            <Button variant="ghost" size="icon" className="text-xs p-1 text-gray-400 hover:text-foreground">
              <i className="fas fa-filter"></i>
            </Button>
            <Button variant="ghost" size="icon" className="text-xs p-1 text-gray-400 hover:text-foreground">
              <i className="fas fa-sync-alt"></i>
            </Button>
          </div>
        )}
      </CardHeader>
      
      <div className="overflow-x-auto">
        <Table>
          <TableHeader className="bg-muted">
            <TableRow>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Assembly ID</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Name</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Member Count</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Updated</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Sync Status</TableHead>
              <TableHead className="px-4 py-2 text-left text-xs font-medium text-gray-400 uppercase tracking-wider"></TableHead>
            </TableRow>
          </TableHeader>
          
          <TableBody className="divide-y divide-border">
            {isLoading ? (
              // Loading state
              Array(5).fill(0).map((_, index) => (
                <TableRow key={index}>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-24" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-40" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-12" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-20" /></TableCell>
                  <TableCell className="px-4 py-2"><Skeleton className="h-4 w-16" /></TableCell>
                  <TableCell className="px-4 py-2 text-right"><Skeleton className="h-4 w-10 ml-auto" /></TableCell>
                </TableRow>
              ))
            ) : isError ? (
              // Error state
              <TableRow>
                <TableCell colSpan={6} className="text-center py-4 text-destructive">
                  <div className="flex flex-col items-center">
                    <i className="fas fa-exclamation-triangle mb-2"></i>
                    <span>{error?.message || 'Failed to load assembly data'}</span>
                  </div>
                </TableCell>
              </TableRow>
            ) : assemblies && assemblies.length > 0 ? (
              // Data loaded successfully
              assemblies.map((assembly) => {
                const syncStatus = getSyncStatus(assembly);
                return (
                  <TableRow key={assembly.id} className="hover:bg-muted">
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm font-mono text-secondary">
                      {assembly.id.substring(0, 8)}...
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm font-medium">
                      {assembly.name}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm">
                      {assembly.member_count.toLocaleString()}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-xs text-muted-foreground">
                      {formatTimeAgo(assembly.updated_at)}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap">
                      <Badge variant="outline" className={`${syncStatus.bgColor} ${syncStatus.color}`}>
                        <i className={`${syncStatus.icon} mr-1 text-xs`}></i>
                        {syncStatus.label}
                      </Badge>
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-right text-sm font-medium">
                      <Link href={`/assemblies/${assembly.id}`}>
                        <Button variant="ghost" size="sm" className="text-primary hover:text-accent text-xs">
                          View <i className="fas fa-chevron-right ml-1"></i>
                        </Button>
                      </Link>
                    </TableCell>
                  </TableRow>
                );
              })
            ) : (
              // No data
              <TableRow>
                <TableCell colSpan={6} className="text-center py-4 text-muted-foreground">
                  <div className="flex flex-col items-center">
                    <i className="fas fa-info-circle mb-2"></i>
                    <span>No assemblies found</span>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
