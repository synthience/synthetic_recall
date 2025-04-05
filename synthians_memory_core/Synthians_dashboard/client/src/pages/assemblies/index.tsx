import React, { useState } from "react";
import { useAssemblies } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { Link } from "wouter";
import { usePollingStore } from "@/lib/store";

export default function AssembliesIndex() {
  const { refreshAllData } = usePollingStore();
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("updated");
  const [sortOrder, setSortOrder] = useState("desc");
  const [statusFilter, setStatusFilter] = useState("all");
  
  // Fetch assemblies data
  const { data, isLoading, isError } = useAssemblies();
  
  // Filter and sort assemblies
  const filteredAssemblies = React.useMemo(() => {
    if (!data?.data) return [];
    
    let filtered = [...data.data];
    
    // Apply search filter
    if (searchTerm) {
      const lowercaseTerm = searchTerm.toLowerCase();
      filtered = filtered.filter(assembly => 
        assembly.id.toLowerCase().includes(lowercaseTerm) ||
        assembly.name.toLowerCase().includes(lowercaseTerm)
      );
    }
    
    // Apply status filter
    if (statusFilter !== "all") {
      filtered = filtered.filter(assembly => {
        if (!assembly.vector_index_updated_at) {
          return statusFilter === "pending";
        }
        
        const vectorDate = new Date(assembly.vector_index_updated_at);
        const updateDate = new Date(assembly.updated_at);
        
        if (statusFilter === "indexed") {
          return vectorDate >= updateDate;
        } else if (statusFilter === "syncing") {
          return vectorDate < updateDate;
        }
        
        return true;
      });
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let aValue, bValue;
      
      if (sortBy === "id") {
        aValue = a.id;
        bValue = b.id;
      } else if (sortBy === "name") {
        aValue = a.name;
        bValue = b.name;
      } else if (sortBy === "members") {
        aValue = a.member_count;
        bValue = b.member_count;
      } else if (sortBy === "updated") {
        aValue = new Date(a.updated_at).getTime();
        bValue = new Date(b.updated_at).getTime();
      }
      
      if (sortOrder === "asc") {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
    
    return filtered;
  }, [data, searchTerm, sortBy, sortOrder, statusFilter]);
  
  // Helper function to get sync status
  const getSyncStatus = (assembly: any) => {
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
  
  // Format time ago
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
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Assembly Inspector</h2>
          <p className="text-sm text-gray-400">Browse and inspect memory assemblies</p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Filter Controls */}
      <Card className="mb-6">
        <CardContent className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <Input
                type="text"
                placeholder="Search by ID or name..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
              <i className="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-gray-500"></i>
            </div>
            
            <div className="flex space-x-2">
              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="id">ID</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="members">Member Count</SelectItem>
                  <SelectItem value="updated">Last Updated</SelectItem>
                </SelectContent>
              </Select>
              
              <Button
                variant="outline"
                size="icon"
                onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              >
                <i className={`fas fa-sort-${sortOrder === "asc" ? "up" : "down"}`}></i>
              </Button>
            </div>
            
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger>
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="indexed">Indexed</SelectItem>
                <SelectItem value="syncing">Syncing</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>
      
      {/* Assemblies Table */}
      <Card>
        <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <CardTitle className="font-medium">Memory Assemblies</CardTitle>
            {!isLoading && (
              <Badge variant="outline" className="ml-2">
                {filteredAssemblies.length} {filteredAssemblies.length === 1 ? 'assembly' : 'assemblies'}
              </Badge>
            )}
          </div>
        </CardHeader>
        
        <div className="overflow-x-auto">
          <Table>
            <TableHeader className="bg-muted">
              <TableRow>
                <TableHead className="w-[180px]">Assembly ID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead className="text-center">Member Count</TableHead>
                <TableHead>Last Updated</TableHead>
                <TableHead>Sync Status</TableHead>
                <TableHead></TableHead>
              </TableRow>
            </TableHeader>
            
            <TableBody>
              {isLoading ? (
                Array(5).fill(0).map((_, index) => (
                  <TableRow key={index}>
                    <TableCell><Skeleton className="h-6 w-28" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-48" /></TableCell>
                    <TableCell className="text-center"><Skeleton className="h-6 w-16 mx-auto" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-12 ml-auto" /></TableCell>
                  </TableRow>
                ))
              ) : isError ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-400">
                    <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
                    Failed to load assemblies. Please try again.
                  </TableCell>
                </TableRow>
              ) : filteredAssemblies.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-400">
                    {searchTerm ? (
                      <>
                        <i className="fas fa-search mr-2"></i>
                        No assemblies matching "{searchTerm}"
                      </>
                    ) : (
                      <>
                        <i className="fas fa-info-circle mr-2"></i>
                        No assemblies found
                      </>
                    )}
                  </TableCell>
                </TableRow>
              ) : (
                filteredAssemblies.map((assembly) => {
                  const syncStatus = getSyncStatus(assembly);
                  return (
                    <TableRow key={assembly.id} className="hover:bg-muted">
                      <TableCell className="font-mono text-secondary">{assembly.id}</TableCell>
                      <TableCell className="font-medium">{assembly.name}</TableCell>
                      <TableCell className="text-center">{assembly.member_count}</TableCell>
                      <TableCell className="text-sm text-gray-400">{formatTimeAgo(assembly.updated_at)}</TableCell>
                      <TableCell>
                        <Badge 
                          variant="outline" 
                          className={`${syncStatus.bgColor} ${syncStatus.color}`}
                        >
                          {syncStatus.label}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Link href={`/assemblies/${assembly.id}`}>
                          <Button variant="ghost" size="sm" className="text-primary hover:text-accent text-xs">
                            View <i className="fas fa-chevron-right ml-1"></i>
                          </Button>
                        </Link>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </Card>
    </>
  );
}
