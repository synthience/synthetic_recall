import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export function SystemArchitecture() {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="px-4 py-3 bg-muted border-b border-border">
        <CardTitle className="font-medium text-base">System Architecture</CardTitle>
      </CardHeader>
      
      <CardContent className="p-6">
        <div className="relative h-64">
          {/* Memory Core Box */}
          <div className="absolute top-6 left-4 md:left-20 w-48 h-20 border border-secondary rounded-md bg-card flex flex-col items-center justify-center">
            <div className="text-sm font-medium text-secondary">Memory Core</div>
            <div className="text-xs text-gray-400 mt-1">Vector Database</div>
          </div>
          
          {/* Neural Memory Box */}
          <div className="absolute top-6 right-4 md:right-20 w-48 h-20 border border-primary rounded-md bg-card flex flex-col items-center justify-center">
            <div className="text-sm font-medium text-primary">Neural Memory</div>
            <div className="text-xs text-gray-400 mt-1">Emotional Loop</div>
          </div>
          
          {/* CCE Box */}
          <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 w-48 h-20 border border-accent rounded-md bg-card flex flex-col items-center justify-center">
            <div className="text-sm font-medium text-accent">CCE</div>
            <div className="text-xs text-gray-400 mt-1">Context Orchestration</div>
          </div>
          
          {/* Connection lines using SVG */}
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 240" preserveAspectRatio="none">
            {/* Left to Bottom */}
            <path 
              d="M120,70 L120,140 L300,140" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#1EE4FF" 
              fill="none" 
            />
            
            {/* Right to Bottom */}
            <path 
              d="M480,70 L480,140 L300,140" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#FF008C" 
              fill="none" 
            />
            
            {/* Bottom to top */}
            <path 
              d="M300,180 L300,140" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#FF3EE8" 
              fill="none" 
            />
            
            {/* Left to Right */}
            <path 
              d="M168,40 C240,40 360,40 432,40" 
              strokeDasharray="5,5" 
              strokeWidth="2" 
              stroke="#FFFFFF" 
              fill="none" 
              opacity="0.3" 
            />
          </svg>
          
          {/* Connection labels */}
          <div className="absolute text-[10px] text-gray-500" style={{ left: '35%', top: '15%' }}>Memory Exchange</div>
          <div className="absolute text-[10px] text-gray-500" style={{ left: '20%', top: '40%' }}>Vector Queries</div>
          <div className="absolute text-[10px] text-gray-500" style={{ right: '20%', top: '40%' }}>Emotional Processing</div>
          <div className="absolute text-[10px] text-gray-500" style={{ left: '46%', top: '65%' }}>Context Flow</div>
        </div>
        
        <div className="flex justify-center mt-4 space-x-6 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 border border-secondary rounded-sm mr-1"></div>
            <span className="text-gray-400">Storage</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 border border-primary rounded-sm mr-1"></div>
            <span className="text-gray-400">Processing</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 border border-accent rounded-sm mr-1"></div>
            <span className="text-gray-400">Orchestration</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
