import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Cell
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface CCEChartProps {
  data: any[];
  isLoading: boolean;
  title: string;
}

export function CCEChart({ data, isLoading, title }: CCEChartProps) {
  // Prepare data for the stacked bar chart
  const prepareStackedData = (rawData: any[]) => {
    // Group data by hour
    const hourlyData: Record<string, { mac7b: number, mac13b: number, titan7b: number }> = {};
    
    if (!rawData || rawData.length === 0) {
      return [];
    }
    
    // Create empty hourly buckets for the last 12 hours
    const now = new Date();
    for (let i = 0; i < 12; i++) {
      const hour = new Date(now.getTime() - (i * 60 * 60 * 1000)).getHours();
      const hourLabel = `${hour}h`;
      hourlyData[hourLabel] = { mac7b: 0, mac13b: 0, titan7b: 0 };
    }
    
    // Fill in the actual data
    rawData.forEach(response => {
      if (!response.variant_selection) return;
      
      const timestamp = new Date(response.timestamp);
      const hour = timestamp.getHours();
      const hourLabel = `${hour}h`;
      
      if (!hourlyData[hourLabel]) {
        hourlyData[hourLabel] = { mac7b: 0, mac13b: 0, titan7b: 0 };
      }
      
      const variant = response.variant_selection.selected_variant.toLowerCase();
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
      'MAC-7b': counts.mac7b,
      'MAC-13b': counts.mac13b,
      'TITAN-7b': counts.titan7b
    }));
  };

  const chartData = prepareStackedData(data);
  
  // Calculate percentages for the legend
  const calculatePercentages = () => {
    if (!data || data.length === 0) return { mac7b: 0, mac13b: 0, titan7b: 0 };
    
    let mac7b = 0, mac13b = 0, titan7b = 0;
    let total = 0;
    
    data.forEach(response => {
      if (!response.variant_selection) return;
      
      const variant = response.variant_selection.selected_variant.toLowerCase();
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

  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border flex justify-between items-center">
        <CardTitle className="font-medium text-base">{title}</CardTitle>
        <button className="text-xs text-gray-400 hover:text-foreground">
          <i className="fas fa-expand-alt"></i>
        </button>
      </CardHeader>
      
      <CardContent className="p-4">
        {isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : (
          <div className="h-48 bg-muted rounded-md relative">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{
                  top: 20,
                  right: 10,
                  left: 10,
                  bottom: 20,
                }}
              >
                <XAxis 
                  dataKey="hour" 
                  tick={{ fontSize: 10, fill: '#666' }}
                  tickLine={{ stroke: '#333' }}
                />
                <YAxis 
                  tick={{ fontSize: 10, fill: '#666' }}
                  tickLine={{ stroke: '#333' }}
                  axisLine={{ stroke: '#333' }}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1E1E1E', border: '1px solid #333' }}
                  labelStyle={{ color: '#ddd' }}
                />
                <Bar 
                  dataKey="MAC-7b" 
                  stackId="a" 
                  fill="#1EE4FF" 
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  dataKey="MAC-13b" 
                  stackId="a" 
                  fill="#FF008C" 
                />
                <Bar 
                  dataKey="TITAN-7b" 
                  stackId="a" 
                  fill="#FF3EE8" 
                  radius={[0, 0, 4, 4]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        <div className="mt-4 flex items-center space-x-6 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-secondary mr-1"></div>
            <span className="text-gray-400">MAC-7b ({percentages.mac7b}%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-primary mr-1"></div>
            <span className="text-gray-400">MAC-13b ({percentages.mac13b}%)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-accent mr-1"></div>
            <span className="text-gray-400">TITAN-7b ({percentages.titan7b}%)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
