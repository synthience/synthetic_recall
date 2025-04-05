import React from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Link } from "wouter";

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
  title?: string;
  showFilters?: boolean;
}

export function AssemblyTable({ assemblies, isLoading, title = "Assemblies", showFilters = true }: AssemblyTableProps) {
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

  const getSyncStatus = (assembly: Assembly) => {
    if (!assembly.vector_index_updated_at) {
      return {
        label: "Pending",
        color: "text-yellow-400",
        bgColor: "bg-muted/50"
      };
    }
    
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    
    if (vectorDate >= updateDate) {
      return {
        label: "Indexed",
        color: "text-secondary",
        bgColor: "bg-muted/50"
      };
    }
    
    return {
      label: "Syncing",
      color: "text-primary",
      bgColor: "bg-muted/50"
    };
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <div className="flex items-center">
          <CardTitle className="font-medium">{title}</CardTitle>
          <Badge variant="outline" className="ml-2 text-xs bg-muted/50 text-gray-300">Memory Core</Badge>
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
            ) : assemblies && assemblies.length > 0 ? (
              assemblies.map((assembly) => {
                const syncStatus = getSyncStatus(assembly);
                return (
                  <TableRow key={assembly.id} className="hover:bg-muted">
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm font-mono text-secondary">
                      {assembly.id}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm">
                      {assembly.name}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-sm">
                      {assembly.member_count}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-xs text-gray-400">
                      {formatTimeAgo(assembly.updated_at)}
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap">
                      <span className={`text-xs ${syncStatus.bgColor} ${syncStatus.color} px-2 py-0.5 rounded-full`}>
                        {syncStatus.label}
                      </span>
                    </TableCell>
                    <TableCell className="px-4 py-2 whitespace-nowrap text-right text-sm font-medium">
                      <Link href={`/assemblies/${assembly.id}`}>
                        <a className="text-primary hover:text-accent text-xs">View</a>
                      </Link>
                    </TableCell>
                  </TableRow>
                );
              })
            ) : (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-4 text-gray-400">
                  No assemblies found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
