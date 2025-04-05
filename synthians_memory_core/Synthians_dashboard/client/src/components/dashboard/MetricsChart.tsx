import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

interface MetricsChartProps {
  title: string;
  data: any[];
  dataKeys: { key: string; color: string; name: string }[];
  isLoading: boolean;
  timeRange: string;
  onTimeRangeChange: (range: string) => void;
  summary?: { label: string; value: string | number; color?: string }[];
}

export function MetricsChart({
  title,
  data,
  dataKeys,
  isLoading,
  timeRange,
  onTimeRangeChange,
  summary
}: MetricsChartProps) {
  const timeRanges = ["24h", "12h", "6h", "1h"];
  
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <CardTitle className="font-medium text-base">{title}</CardTitle>
        
        <div className="flex space-x-2">
          {timeRanges.map((range) => (
            <button
              key={range}
              className={`text-xs px-2 py-1 rounded ${
                timeRange === range
                  ? "bg-muted text-secondary"
                  : "bg-muted/50 text-gray-300 hover:bg-muted"
              }`}
              onClick={() => onTimeRangeChange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent className="p-4">
        {isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : (
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={{
                  top: 10,
                  right: 10,
                  left: 10,
                  bottom: 10,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#666" 
                  fontSize={10}
                  tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                />
                <YAxis stroke="#666" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#1E1E1E", border: "1px solid #333" }}
                  labelStyle={{ color: "#ddd" }}
                />
                
                {dataKeys.map((dataKey) => (
                  <Line
                    key={dataKey.key}
                    type="monotone"
                    dataKey={dataKey.key}
                    name={dataKey.name}
                    stroke={dataKey.color}
                    activeDot={{ r: 4 }}
                    strokeWidth={2}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {summary && !isLoading && (
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            {summary.map((item, index) => (
              <div key={index} className="bg-muted p-2 rounded">
                <div className="text-gray-500">{item.label}</div>
                <div className={item.color || "text-primary"}>{item.value}</div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
