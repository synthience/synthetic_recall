import React, { useState } from "react";
import { useRecentCCEResponses } from "@/lib/api";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { RefreshButton } from "@/components/ui/RefreshButton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { usePollingStore } from "@/lib/store";

export default function LLMGuidance() {
  const { refreshAllData } = usePollingStore();
  const [confidenceFilter, setConfidenceFilter] = useState("all");
  const [variantFilter, setVariantFilter] = useState("all");
  const [expandedResponse, setExpandedResponse] = useState<string | null>(null);
  
  // Fetch CCE responses that include LLM guidance
  const { data, isLoading, isError } = useRecentCCEResponses();
  
  // Filter responses that have LLM advice
  const llmResponses = React.useMemo(() => {
    if (!data?.data?.recent_responses) return [];
    
    // Only include responses with LLM advice
    let filtered = data.data.recent_responses.filter(
      response => response.llm_advice_used
    );
    
    // Apply confidence filter
    if (confidenceFilter !== "all") {
      filtered = filtered.filter(response => {
        const confidence = response.llm_advice_used?.confidence_level || 0;
        
        if (confidenceFilter === "high") {
          return confidence >= 0.8;
        } else if (confidenceFilter === "medium") {
          return confidence >= 0.5 && confidence < 0.8;
        } else if (confidenceFilter === "low") {
          return confidence < 0.5;
        }
        
        return true;
      });
    }
    
    // Apply variant filter
    if (variantFilter !== "all") {
      filtered = filtered.filter(response => {
        const variantHint = response.llm_advice_used?.adjusted_advice || "";
        return variantHint.toLowerCase().includes(variantFilter.toLowerCase());
      });
    }
    
    return filtered;
  }, [data, confidenceFilter, variantFilter]);
  
  // Calculate statistics
  const stats = React.useMemo(() => {
    if (!data?.data?.recent_responses) {
      return {
        totalRequests: 0,
        avgConfidence: 0,
        variantDistribution: {}
      };
    }
    
    const llmResponses = data.data.recent_responses.filter(
      response => response.llm_advice_used
    );
    
    // Calculate average confidence
    const totalConfidence = llmResponses.reduce((acc, response) => {
      return acc + (response.llm_advice_used?.confidence_level || 0);
    }, 0);
    
    const avgConfidence = llmResponses.length > 0 
      ? totalConfidence / llmResponses.length 
      : 0;
    
    // Calculate variant distribution
    const variantDistribution: Record<string, number> = {};
    
    llmResponses.forEach(response => {
      const advice = response.llm_advice_used?.adjusted_advice || "";
      
      if (advice.toLowerCase().includes("mac-7b")) {
        variantDistribution["MAC-7b"] = (variantDistribution["MAC-7b"] || 0) + 1;
      } else if (advice.toLowerCase().includes("mac-13b")) {
        variantDistribution["MAC-13b"] = (variantDistribution["MAC-13b"] || 0) + 1;
      } else if (advice.toLowerCase().includes("titan")) {
        variantDistribution["TITAN-7b"] = (variantDistribution["TITAN-7b"] || 0) + 1;
      }
    });
    
    return {
      totalRequests: llmResponses.length,
      avgConfidence: avgConfidence,
      variantDistribution
    };
  }, [data]);
  
  // Format timestamp
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };
  
  // Get confidence badge color
  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) {
      return <Badge className="bg-green-600">High ({confidence.toFixed(2)})</Badge>;
    } else if (confidence >= 0.5) {
      return <Badge className="bg-blue-600">Medium ({confidence.toFixed(2)})</Badge>;
    } else {
      return <Badge className="bg-orange-600">Low ({confidence.toFixed(2)})</Badge>;
    }
  };

  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">LLM Guidance Monitor</h2>
          <p className="text-sm text-gray-400">
            Monitor interactions with external LLM services for context orchestration
          </p>
        </div>
        <RefreshButton onClick={refreshAllData} />
      </div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Total LLM Requests</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-mono">{stats.totalRequests}</p>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Average Confidence</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-mono">{stats.avgConfidence.toFixed(2)}</p>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-500">Top Variant Hint</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-32" />
            ) : (
              <div>
                {Object.entries(stats.variantDistribution).length > 0 ? (
                  <p className="text-2xl font-mono">
                    {Object.entries(stats.variantDistribution)
                      .sort((a, b) => b[1] - a[1])[0]?.[0] || "None"}
                  </p>
                ) : (
                  <p className="text-gray-400">No data available</p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      
      {/* Filters */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Filter LLM Guidance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 mb-2">Confidence Level</p>
              <Select value={confidenceFilter} onValueChange={setConfidenceFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by confidence" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Confidence Levels</SelectItem>
                  <SelectItem value="high">High (≥ 0.8)</SelectItem>
                  <SelectItem value="medium">Medium (0.5 - 0.8)</SelectItem>
                  <SelectItem value="low">Low (&lt; 0.5)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <p className="text-sm text-gray-500 mb-2">Variant Hint</p>
              <Select value={variantFilter} onValueChange={setVariantFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by variant" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Variants</SelectItem>
                  <SelectItem value="mac-7b">MAC-7b</SelectItem>
                  <SelectItem value="mac-13b">MAC-13b</SelectItem>
                  <SelectItem value="titan">TITAN-7b</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* LLM Guidance Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent LLM Guidance</CardTitle>
          <CardDescription>
            {isLoading ? (
              <Skeleton className="h-4 w-48" />
            ) : (
              <>Showing {llmResponses.length} LLM guidance requests</>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-64 w-full" />
            </div>
          ) : isError ? (
            <div className="text-center py-8 text-gray-400">
              <i className="fas fa-exclamation-circle text-destructive mr-2"></i>
              Failed to load LLM guidance data
            </div>
          ) : llmResponses.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <i className="fas fa-info-circle mr-2"></i>
              No LLM guidance data available
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[180px]">Timestamp</TableHead>
                    <TableHead>Input Summary</TableHead>
                    <TableHead>Adjusted Advice</TableHead>
                    <TableHead className="w-[120px]">Confidence</TableHead>
                    <TableHead className="w-[80px]">Raw Data</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {llmResponses.map((response, index) => (
                    <TableRow key={index}>
                      <TableCell className="font-mono text-xs">
                        {formatTimestamp(response.timestamp)}
                      </TableCell>
                      <TableCell>
                        <div className="max-w-xs truncate">
                          {response.variant_selection?.reason || "N/A"}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="max-w-xs truncate">
                          {response.llm_advice_used?.adjusted_advice || "N/A"}
                        </div>
                      </TableCell>
                      <TableCell>
                        {getConfidenceBadge(response.llm_advice_used?.confidence_level || 0)}
                      </TableCell>
                      <TableCell>
                        <Collapsible>
                          <CollapsibleTrigger asChild>
                            <Button 
                              variant="ghost" 
                              size="sm" 
                              onClick={() => setExpandedResponse(
                                expandedResponse === response.timestamp ? null : response.timestamp
                              )}
                            >
                              <i className={`fas fa-chevron-${expandedResponse === response.timestamp ? 'up' : 'down'}`}></i>
                            </Button>
                          </CollapsibleTrigger>
                          <CollapsibleContent className="mt-2">
                            <div className="bg-muted p-3 rounded text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                              {response.llm_advice_used?.raw_advice || "Raw advice not available"}
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}
