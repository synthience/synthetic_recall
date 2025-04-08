import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert"; 
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Define proper types for the CCE response data
interface VariantSelection {
  selected_variant: string;
  confidence?: number;
  selection_time?: number;
}

interface CCEResponse {
  timestamp: string;
  variant_selection?: VariantSelection;
  [key: string]: any; // Allow for other fields
}

// Type for the prepared chart data
interface ChartDataPoint {
  hour: string;
  MAC: number;
  MAG: number;
  MAL: number;
}

interface CCEChartProps {
  data: CCEResponse[];
  isLoading: boolean;
  isError?: boolean;
  error?: any;
  errorMessage?: string | null;
  title: string;
}

export function CCEChart({ data, isLoading, isError = false, error, errorMessage, title }: CCEChartProps) {
  // Prepare data for the stacked bar chart
  const prepareStackedData = (rawData: CCEResponse[]): ChartDataPoint[] => {
    if (!Array.isArray(rawData) || rawData.length === 0 || isError) {
      // Empty dataset - no data to display
      return []; // Return empty array if we're in an error state
    }
    
    // Group data by hour
    const hourlyData: Record<string, { mac7b: number, mac13b: number, titan7b: number }> = {};
    
    // Create empty hourly buckets for the last 12 hours
    const now = new Date();
    for (let i = 0; i < 12; i++) {
      const hourDate = new Date(now.getTime() - (i * 60 * 60 * 1000));
      // Make sure we have a valid date
      if (!isNaN(hourDate.getTime())) {
        const hour = hourDate.getHours();
        const hourLabel = `${hour}h`;
        hourlyData[hourLabel] = { mac7b: 0, mac13b: 0, titan7b: 0 };
      }
    }
    
    // Fill in the actual data
    rawData.forEach(response => {
      if (!response?.variant_selection?.selected_variant) return;
      
      let timestamp: Date;
      try {
        timestamp = new Date(response.timestamp);
        // Check if the date is valid
        if (isNaN(timestamp.getTime())) return;
      } catch (e) {
        console.warn("Invalid timestamp in CCE response:", response.timestamp);
        return;
      }
      
      const hour = timestamp.getHours();
      const hourLabel = `${hour}h`;
      
      if (!hourlyData[hourLabel]) {
        hourlyData[hourLabel] = { mac7b: 0, mac13b: 0, titan7b: 0 };
      }
      
      const variant = (response.variant_selection.selected_variant || "").toLowerCase();
      if (variant.includes('mac-7b')) {
        hourlyData[hourLabel].mac7b += 1;
      } else if (variant.includes('mac-13b')) {
        hourlyData[hourLabel].mac13b += 1;
      } else if (variant.includes('titan')) {
        hourlyData[hourLabel].titan7b += 1;
      }
    });
    
    // Convert to array format for Recharts
    return Object.entries(hourlyData).map(([hour, counts]) => ({
      hour,
      'MAC': counts.mac7b,
      'MAG': counts.mac13b,
      'MAL': counts.titan7b
    }));
  };

  const chartData = prepareStackedData(data);
  
  // Calculate percentages for the legend
  const calculatePercentages = () => {
    if (!Array.isArray(data) || data.length === 0 || isError) {
      return { mac7b: 0, mac13b: 0, titan7b: 0 };
    }
    
    let mac7b = 0, mac13b = 0, titan7b = 0;
    let total = 0;
    
    data.forEach(response => {
      if (!response?.variant_selection?.selected_variant) return;
      
      const variant = (response.variant_selection.selected_variant || "").toLowerCase();
      if (variant.includes('mac-7b')) {
        mac7b += 1;
      } else if (variant.includes('mac-13b')) {
        mac13b += 1;
      } else if (variant.includes('titan')) {
        titan7b += 1;
      }
      total += 1;
    });
    
    return {
      mac7b: total > 0 ? Math.round((mac7b / total) * 100) : 0,
      mac13b: total > 0 ? Math.round((mac13b / total) * 100) : 0,
      titan7b: total > 0 ? Math.round((titan7b / total) * 100) : 0
    };
  };
  
  const percentages = calculatePercentages();
  
  const colors = {
    'MAC': '#3b82f6', // Blue
    'MAG': '#a855f7', // Purple
    'MAL': '#ec4899'  // Pink
  };
  
  return (
    <Card className="w-full">
      <CardHeader className="px-4 py-3 flex flex-col sm:flex-row sm:justify-between sm:items-center bg-muted border-b border-border">
        <CardTitle className="font-medium text-base">{title}</CardTitle>
      </CardHeader>
      
      <CardContent className="p-4 pt-4">
        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-[200px] w-full" />
            <div className="grid grid-cols-3 gap-4">
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
            </div>
          </div>
        ) : isError ? (
          <Alert variant="destructive" className="mb-4">
            <AlertTitle>Failed to load CCE data</AlertTitle>
            <AlertDescription>
              {errorMessage || error?.message || "There was an error fetching CCE data. Please try again later."}
            </AlertDescription>
          </Alert>
        ) : (!Array.isArray(data) || data.length === 0) ? (
          <div className="text-center py-8 text-muted-foreground">
            <p>No variant selection data available</p>
          </div>
        ) : (
          <>
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData.reverse()} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="hour" stroke="#888" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#888" tick={{ fontSize: 12 }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e1e2d', borderColor: '#333' }}
                    formatter={(value) => [(value ?? 0).toString()]}
                  />
                  <Legend />
                  <Bar dataKey="MAC" stackId="a" fill={colors['MAC']} name="MAC (7B)" />
                  <Bar dataKey="MAG" stackId="a" fill={colors['MAG']} name="MAG (13B)" />
                  <Bar dataKey="MAL" stackId="a" fill={colors['MAL']} name="MAL (Titan)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="bg-muted p-3 rounded">
                <div className="text-xs text-gray-500 mb-1">MAC: 7B</div>
                <div className="text-xl font-mono text-blue-500">{percentages?.mac7b ?? 0}%</div>
              </div>
              <div className="bg-muted p-3 rounded">
                <div className="text-xs text-gray-500 mb-1">MAG: 13B</div>
                <div className="text-xl font-mono text-purple-500">{percentages?.mac13b ?? 0}%</div>
              </div>
              <div className="bg-muted p-3 rounded">
                <div className="text-xs text-gray-500 mb-1">MAL: Titan</div>
                <div className="text-xl font-mono text-pink-500">{percentages?.titan7b ?? 0}%</div>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
