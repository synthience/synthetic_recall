import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface MetricsChartProps {
  title: string;
  data: any[];
  dataKeys: { key: string; color: string; name: string }[];
  isLoading: boolean;
  isError?: boolean;
  error?: any;
  timeRange: string;
  onTimeRangeChange: (range: string) => void;
  summary?: { label: string; value: string | number; color?: string }[];
}

export function MetricsChart({
  title,
  data,
  dataKeys,
  isLoading,
  isError = false,
  error,
  timeRange,
  onTimeRangeChange,
  summary
}: MetricsChartProps) {
  const timeRanges = ["24h", "12h", "6h", "1h"];
  
  // Empty data for initial state or errors - no random values
  const emptyData = [
    { timestamp: "2025-04-05T10:00:00Z" },
    { timestamp: "2025-04-05T11:00:00Z" },
    { timestamp: "2025-04-05T12:00:00Z" },
    { timestamp: "2025-04-05T13:00:00Z" },
    { timestamp: "2025-04-05T14:00:00Z" },
  ];
  
  // Use empty data only if we're not in error state and have no data
  const chartData = isError ? [] : (data && data.length > 0 ? data : emptyData);
  
  // Format date for display in chart
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
  };

  return (
    <Card className="w-full">
      <CardHeader className="px-4 py-3 flex flex-col sm:flex-row sm:justify-between sm:items-center bg-muted border-b border-border space-y-2 sm:space-y-0">
        <CardTitle className="font-medium text-base">{title}</CardTitle>
        <div className="flex space-x-1">
          {timeRanges.map((range) => (
            <button
              key={range}
              onClick={() => onTimeRangeChange(range)}
              className={`px-2 py-1 text-xs rounded ${timeRange === range ? 'bg-primary text-primary-foreground' : 'bg-muted-foreground/20 hover:bg-muted-foreground/30'}`}
            >
              {range}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-4">
        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-[240px] w-full" />
            {summary && (
              <div className="grid grid-cols-3 gap-4">
                {summary.map((_, i) => (
                  <Skeleton key={i} className="h-16" />
                ))}
              </div>
            )}
          </div>
        ) : isError ? (
          <Alert variant="destructive" className="mb-4">
            <AlertTitle>Failed to load chart data</AlertTitle>
            <AlertDescription>
              {error?.message || "There was an error fetching the metrics data. Please try again later."}
            </AlertDescription>
          </Alert>
        ) : (
          <>
            <div className="h-[240px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={formatDate} 
                    stroke="#888" 
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis stroke="#888" tick={{ fontSize: 12 }} />
                  <Tooltip 
                    labelFormatter={(label) => formatDate(label)}
                    contentStyle={{ backgroundColor: '#1e1e2d', borderColor: '#333' }}
                  />
                  <Legend />
                  {dataKeys.map((dataKey) => (
                    <Line
                      key={dataKey.key}
                      type="monotone"
                      dataKey={dataKey.key}
                      stroke={dataKey.color}
                      name={dataKey.name}
                      strokeWidth={2}
                      dot={{ r: 2 }}
                      activeDot={{ r: 4 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            {summary && (
              <div className="grid grid-cols-3 gap-4 mt-4">
                {summary.map((item, index) => (
                  <div key={index} className="bg-muted p-3 rounded">
                    <div className="text-xs text-gray-500 mb-1">{item.label}</div>
                    <div className={`text-xl font-mono ${item.color || 'text-white'}`}>
                      {item.value}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
