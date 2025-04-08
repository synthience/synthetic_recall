import React, { useState, useMemo } from "react";
import { useAssemblies } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { AssemblyTable } from "@/components/dashboard/AssemblyTable";
import { usePollingStore } from "@/lib/store";
import { useFeatures } from "@/contexts/FeaturesContext";
import { ErrorDisplay } from "@/components/ui/ErrorDisplay";

export default function AssembliesIndex() {
  const { refreshAllData } = usePollingStore();
  const { explainabilityEnabled } = useFeatures();
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("updated_at");
  const [sortOrder, setSortOrder] = useState("desc");
  const [statusFilter, setStatusFilter] = useState("all");

  const { data: apiResponse, isLoading, isError, error, refetch } = useAssemblies();

  const getSyncStatusLabel = (assembly: any): string => {
    if (!assembly?.vector_index_updated_at) return "pending";
    const vectorDate = new Date(assembly.vector_index_updated_at);
    const updateDate = new Date(assembly.updated_at);
    if (isNaN(vectorDate.getTime()) || isNaN(updateDate.getTime())) return "error";
    return vectorDate >= updateDate ? "indexed" : "syncing";
  };

  const filteredAssemblies = useMemo(() => {
    if (!apiResponse?.success || !Array.isArray(apiResponse?.data)) {
      console.warn("[AssembliesIndex] Invalid or missing assembly data array.");
      return [];
    }

    let filtered = [...apiResponse.data];

    if (searchTerm) {
      const lowercaseTerm = searchTerm.toLowerCase();
      filtered = filtered.filter(assembly =>
        assembly?.name?.toLowerCase().includes(lowercaseTerm) ||
        assembly?.id?.toLowerCase().includes(lowercaseTerm)
      );
    }

    if (statusFilter !== "all") {
      filtered = filtered.filter(assembly => {
        if (!assembly || !assembly.updated_at) return false;
        const statusLabel = getSyncStatusLabel(assembly);
        return statusLabel === statusFilter;
      });
    }

    filtered.sort((a, b) => {
      let aValue: string | number | Date = '';
      let bValue: string | number | Date = '';

      if (sortBy === "id") { aValue = a?.id || ''; bValue = b?.id || ''; }
      else if (sortBy === "name") { aValue = a?.name || ''; bValue = b?.name || ''; }
      else if (sortBy === "members") { aValue = a?.member_count ?? 0; bValue = b?.member_count ?? 0; }
      else if (sortBy === "updated_at") {
        aValue = a?.updated_at ? new Date(a.updated_at).getTime() : 0;
        bValue = b?.updated_at ? new Date(b.updated_at).getTime() : 0;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === "asc" ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
      } else if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === "asc" ? aValue - bValue : bValue - aValue;
      }
      return 0;
    });

    return filtered;
  }, [apiResponse, searchTerm, sortBy, sortOrder, statusFilter]);

  if (isLoading) {
    return (
      <div className="p-6 space-y-4">
        <Skeleton className="h-10 w-1/3 mb-6" />
        <Skeleton className="h-16 w-full mb-6" />
        <Card><CardContent className="pt-6"><Skeleton className="h-64 w-full" /></CardContent></Card>
      </div>
    );
  }

  if (isError || !apiResponse?.success) {
    const displayError = error || new Error(apiResponse?.error || 'Failed to load assemblies. An unknown error occurred.');
    return (
      <div className="p-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xl font-semibold text-white mb-1">Assembly Inspector</h2>
            <p className="text-sm text-gray-400">Browse and inspect memory assemblies</p>
          </div>
          <RefreshButton onClick={() => refetch()} isLoading={isLoading} />
        </div>
        <ErrorDisplay
          error={displayError}
          refetch={refetch}
          title="Error Loading Assemblies"
        />
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Assembly Inspector</h2>
          <p className="text-sm text-gray-400">Browse and inspect memory assemblies</p>
        </div>
        <RefreshButton onClick={() => refetch()} isLoading={isLoading} />
      </div>

      <Card className="mb-6">
        <CardContent className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Input
              placeholder="Search by ID or name..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger><SelectValue placeholder="Sort by..." /></SelectTrigger>
              <SelectContent>
                <SelectItem value="updated_at">Last Updated</SelectItem>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="members">Members</SelectItem>
                <SelectItem value="id">ID</SelectItem>
              </SelectContent>
            </Select>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger><SelectValue placeholder="Filter status..." /></SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="indexed">Indexed</SelectItem>
                <SelectItem value="syncing">Syncing</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="error">Error</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <CardTitle className="font-medium">Memory Assemblies</CardTitle>
            <Badge variant="outline">
              {filteredAssemblies.length} {filteredAssemblies.length === 1 ? 'assembly' : 'assemblies'} found
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {filteredAssemblies.length > 0 ? (
            <AssemblyTable
              assemblies={filteredAssemblies}
              isLoading={false}
              isError={false}
            />
          ) : (
            <div className="p-6 text-center text-gray-500">
              No assemblies match the current filters.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
