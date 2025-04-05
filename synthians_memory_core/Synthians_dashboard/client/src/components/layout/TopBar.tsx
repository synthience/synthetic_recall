import React from "react";
import { RefreshButton } from "../ui/RefreshButton";
import { usePollingStore } from "@/lib/store";
import { Link } from "wouter";

interface TopBarProps {
  toggleSidebar: () => void;
}

export function TopBar({ toggleSidebar }: TopBarProps) {
  const { pollingRate, setPollingRate, refreshAllData } = usePollingStore();

  const handleRefresh = () => {
    refreshAllData();
  };

  const handlePollingRateChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setPollingRate(parseInt(e.target.value));
  };

  return (
    <header className="bg-card border-b border-border px-4 py-3 flex justify-between items-center">
      <div className="flex items-center md:hidden">
        <button 
          onClick={toggleSidebar} 
          className="p-2 rounded-md bg-muted text-primary"
        >
          <i className="fas fa-bars"></i>
        </button>
        <div className="ml-3">
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-primary to-accent flex items-center justify-center mr-2">
            <span className="text-white font-bold text-xs">S</span>
          </div>
          <h1 className="text-sm font-bold text-primary">Synthians</h1>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="relative max-w-xs w-64 hidden md:block">
          <span className="absolute inset-y-0 left-0 pl-3 flex items-center">
            <i className="fas fa-search text-gray-500"></i>
          </span>
          <input 
            type="text" 
            placeholder="Search..." 
            className="bg-muted text-sm rounded-md pl-10 pr-4 py-1.5 w-full focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>
        
        <RefreshButton onClick={handleRefresh} />
        
        <div className="w-px h-6 bg-muted mx-2"></div>
        
        <div className="text-xs text-gray-400 flex items-center">
          Poll rate: 
          <select 
            value={pollingRate} 
            onChange={handlePollingRateChange} 
            className="ml-2 bg-muted text-secondary p-1 rounded border border-border"
          >
            <option value={5000}>5s</option>
            <option value={10000}>10s</option>
            <option value={30000}>30s</option>
            <option value={60000}>60s</option>
          </select>
        </div>
      </div>
      
      <div className="flex items-center">
        <span className="mr-2 text-xs text-gray-400 hidden md:inline-block">
          Memory Core: <span className="text-secondary">Healthy</span>
        </span>
        <Link href="/admin">
          <div className="text-xs px-2 py-1 rounded bg-muted border border-border text-gray-300 hover:bg-muted/90 cursor-pointer">
            <i className="fas fa-exclamation-triangle text-yellow-400 mr-1"></i>
            <span>Diagnostics</span>
          </div>
        </Link>
      </div>
    </header>
  );
}
