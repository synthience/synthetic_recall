import React, { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { MergeLogView } from "@/components/dashboard/MergeLogView";
import { LogLevelFilter } from "@/components/dashboard/LogLevelFilter";
import { ServiceFilter } from "@/components/dashboard/ServiceFilter";
import { useFeatures } from "@/contexts/FeaturesContext";
import { useMergeLog } from "@/lib/api";
import { useWebSocketLogs } from "@/lib/hooks/useWebSocketLogs";
import { LogMessage, LogLevel, ConnectionStatus } from "@/lib/websocket";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { format } from "date-fns";
import { toast } from "@/hooks/use-toast";
import { VariableSizeList as List } from 'react-window';
import { AlertCircle, RotateCw } from "lucide-react";

// Define ServiceType here
type ServiceType = 'memory-core' | 'neural-memory' | 'cce' | 'all';

// Component for individual log entry
function LogEntryRow({ log }: { log: LogMessage }) {
  // Map of log levels to text colors
  const levelColors = {
    debug: "text-gray-400",
    info: "text-blue-400",
    warning: "text-yellow-400",
    error: "text-red-400"
  };
  
  // Map of services to border colors
  const serviceColors = {
    "memory-core": "border-secondary",
    "neural-memory": "border-primary",
    "cce": "border-accent"
  };
  
  // Format timestamp
  const formattedTime = format(new Date(log.timestamp), "HH:mm:ss.SSS");
  
  return (
    <div 
      className={`px-3 py-2 border-l-2 ${serviceColors[log.service]} text-xs font-mono mb-1 hover:bg-muted`}
      data-testid={`log-entry-${log.id}`}
      ref={node => { 
        // Example: If using react-measure or similar
      }}
    >
      <span className="text-gray-500 mr-2">{formattedTime}</span>
      <Badge variant="outline" className={`${levelColors[log.level]} mr-2 capitalize`}>
        {log.level}
      </Badge>
      <Badge variant="outline" className="mr-2">
        {log.service}
      </Badge>
      <span>{log.message}</span>
      {log.context && (
        <div className="mt-1 pl-4 text-gray-500 text-xs">
          Context: {JSON.stringify(log.context)}
        </div>
      )}
    </div>
  );
}

// Estimate row height based on content - refine this based on actual rendering
const estimateLogHeight = (log: LogMessage): number => {
  const baseHeight = 35; // Base height for a single line log
  const messageLines = Math.ceil(log.message.length / 100); // Estimate lines based on char count
  const lineHeight = 16; // Approximate line height in pixels
  let estimatedHeight = baseHeight + (messageLines - 1) * lineHeight;
  
  if (log.context) {
    try {
      const contextString = JSON.stringify(log.context);
      const contextLines = Math.ceil(contextString.length / 80); // Estimate context lines
      estimatedHeight += (contextLines * lineHeight) + 5; // Add space for context title
    } catch (e) {
      estimatedHeight += 20; // Fallback if context stringify fails
    }
  }
  
  return Math.max(baseHeight, estimatedHeight); // Ensure minimum height
};

// Virtual List Row Renderer for optimized log display
const VirtualizedLogRow = ({ index, style, data }: { index: number; style: React.CSSProperties; data: LogMessage[] }) => {
  const log = data[index]; // Access log directly from the data array
  return (
    <div style={style}>
      <LogEntryRow log={log} />
    </div>
  );
};

// Component for connection status indicator
function ConnectionStatusIndicator({ status }: { status: ConnectionStatus }) {
  const statusConfig = {
    connected: { color: "bg-green-600", text: "Connected" },
    connecting: { color: "bg-yellow-600", text: "Connecting" },
    disconnected: { color: "bg-gray-600", text: "Disconnected" },
    error: { color: "bg-red-600", text: "Connection Error" }
  };
  
  const config = statusConfig[status];
  
  return (
    <Badge 
      variant={status === "connected" ? "default" : "outline"} 
      className={status === "connected" ? `${config.color} text-white` : "text-gray-400"}
    >
      <div className={`w-2 h-2 rounded-full mr-1 ${status === "connected" ? "bg-background" : config.color}`}></div>
      {config.text}
    </Badge>
  );
}

export default function Logs() {
  // Get feature flags
  const { explainabilityEnabled, isLoading: featuresAreLoading } = useFeatures();
  
  // State for auto-scrolling
  const [autoScroll, setAutoScroll] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const listRef = useRef<List>(null);
  const sizeMap = useRef<{ [key: number]: number }>({});
  
  // Fetch merge logs if explainability is enabled
  const { data: mergeLogData, isLoading: mergeLogLoading, isError: mergeLogError, error: mergeLogErrorData, refetch: refetchMergeLogs } = useMergeLog(50, { 
    enabled: !featuresAreLoading && explainabilityEnabled 
  });
  
  // Extract merge log entries for better readability
  const mergeLogEntries = mergeLogData?.data?.reconciled_log_entries || [];
  
  // Use our WebSocket logs hook
  const {
    logs,
    connectionStatus,
    connect,
    disconnect,
    clearLogs,
    filter,
    setFilter,
    isConnecting,
    error
  } = useWebSocketLogs({
    // Wait for features context to settle before initializing WebSocket
    autoConnect: !featuresAreLoading,
    maxLogs: 1000,
    initialFilter: { service: 'all', level: 'all', search: '' }
  });
  
  // Set search filter when search term changes
  useEffect(() => {
    setFilter({ search: searchTerm });
  }, [searchTerm, setFilter]);
  
  // Function to get item size for VariableSizeList
  const getItemSize = useCallback((index: number): number => {
    if (!sizeMap.current[index]) {
      sizeMap.current[index] = estimateLogHeight(logs[index]);
    }
    return sizeMap.current[index];
  }, [logs]);
  
  // Reset size map when logs change significantly (e.g., clear)
  useEffect(() => {
    sizeMap.current = {};
    listRef.current?.resetAfterIndex(0);
  }, [logs.length === 0]); // Trigger only when cleared or initially loaded
  
  // Handle auto-scrolling with virtualized list
  useEffect(() => {
    if (autoScroll && listRef.current && logs.length > 0 && connectionStatus === 'connected') {
      // Ensure scrollToItem exists before calling
      if (typeof listRef.current.scrollToItem === 'function') {
        listRef.current.scrollToItem(0, 'start'); // Scroll to the newest log (index 0)
      }
    }
  }, [logs, autoScroll, connectionStatus]);
  
  // Count logs by service and level for badge counts
  const logCounts = React.useMemo(() => {
    const serviceCounts = { 'all': logs.length, 'memory-core': 0, 'neural-memory': 0, 'cce': 0 };
    const levelCounts = { 'all': logs.length, 'debug': 0, 'info': 0, 'warning': 0, 'error': 0 };
    
    logs.forEach(log => {
      serviceCounts[log.service]++;
      levelCounts[log.level]++;
    });
    
    return { serviceCounts, levelCounts };
  }, [logs]);
  
  // Handle connection changes - WebSockets will be implemented in the next phase (5.9.3)
  const handleConnectionToggle = () => {
    if (connectionStatus === 'connected' || connectionStatus === 'connecting') {
      disconnect();
      toast({
        title: "Disconnected from log stream",
        description: "Log stream connection closed.",
        variant: "default"
      });
    } else {
      toast({
        title: "Connecting to log stream...",
        description: `Attempting connection to ${import.meta.env.VITE_WS_URL || 'ws://localhost:5000/logs'}`,
        variant: "default"
      });
      connect();
    }
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Logs</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Stream logs from Synthians services for real-time debugging
          </p>
        </div>
        
        <div className="flex items-center">
          <ConnectionStatusIndicator status={connectionStatus} />
        </div>
      </div>
      
      <Tabs defaultValue="realtime" className="space-y-4">
        <TabsList>
          <TabsTrigger value="realtime">Real-time Logs</TabsTrigger>
          {explainabilityEnabled && (
            <TabsTrigger value="merges">Merge Log</TabsTrigger>
          )}
        </TabsList>
        
        <TabsContent value="realtime">
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle>Log Filters</CardTitle>
              <CardDescription>
                Filter logs by service, level, or search for specific content
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col gap-4 md:flex-row">
                <div className="flex-1">
                  <h4 className="text-sm font-medium mb-2">Service Filter</h4>
                  <ServiceFilter 
                    selectedService={filter.service as ServiceType} 
                    onChange={(service: ServiceType) => setFilter({ service })}
                    counts={logCounts.serviceCounts}
                  />
                </div>
                
                <div className="flex-1">
                  <h4 className="text-sm font-medium mb-2">Log Level Filter</h4>
                  <LogLevelFilter 
                    selectedLevel={filter.level as LogLevel | 'all'} 
                    onChange={(level: LogLevel | 'all') => setFilter({ level })}
                    counts={logCounts.levelCounts}
                  />
                </div>
                
                <div className="flex-1">
                  <h4 className="text-sm font-medium mb-2">Search Logs</h4>
                  <Input 
                    placeholder="Search log content..." 
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full"
                  />
                </div>
              </div>
              
              <div className="flex justify-between items-center mt-4">
                <div className="flex items-center space-x-2">
                  <Switch 
                    id="auto-scroll" 
                    checked={autoScroll}
                    onCheckedChange={setAutoScroll}
                  />
                  <label htmlFor="auto-scroll" className="text-sm font-medium cursor-pointer">
                    Auto-scroll to latest logs
                  </label>
                </div>
                
                <div className="flex space-x-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    disabled={isConnecting}
                    onClick={handleConnectionToggle}
                  >
                    {connectionStatus === 'connected' ? 'Disconnect' : 'Connect'}
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={clearLogs}
                  >
                    Clear Logs
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="pb-2">
              <div className="flex justify-between items-center">
                <CardTitle>Log Stream</CardTitle>
                <Badge variant="outline">{logs.length} events</Badge>
              </div>
            </CardHeader>
            <CardContent>
              {connectionStatus === 'error' && (
                <Alert variant="destructive" className="mb-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Connection Error</AlertTitle>
                  <AlertDescription>
                    {typeof error === 'string' ? error : error?.message || "Failed to connect to log stream. Please try reconnecting."}
                  </AlertDescription>
                  {connectionStatus === 'error' && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="mt-2"
                      onClick={connect} 
                      disabled={isConnecting}
                    >
                      <RotateCw className={`mr-2 h-4 w-4 ${isConnecting ? 'animate-spin' : ''}`} />
                      Retry Connection
                    </Button>
                  )}
                </Alert>
              )}
              
              <div className="h-[500px] border border-border rounded-md overflow-hidden bg-card">
                {logs.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-center p-4 text-muted-foreground">
                    <div className="flex flex-col items-center">
                      <p className="mb-2 font-medium">
                        {connectionStatus === 'connecting' && 'Attempting connection...'}
                        {connectionStatus === 'disconnected' && 'Disconnected from log stream.'}
                        {connectionStatus === 'connected' && 'Waiting for new log events...'}
                        {connectionStatus === 'error' && 'Connection failed.'}
                      </p>
                      <p className="text-sm">
                        {connectionStatus === 'connected' 
                          ? 'No logs match the current filters.'
                          : connectionStatus !== 'connecting' ? 'Press Connect to start receiving logs.' : ''}
                      </p>
                      {(connectionStatus === 'disconnected' || connectionStatus === 'error') && !isConnecting && (
                        <Button size="sm" onClick={connect} className="mt-4">Connect</Button>
                      )}
                    </div>
                  </div>
                ) : (
                  <List
                    ref={listRef}
                    height={500}
                    width="100%"
                    itemData={logs} // Pass only the logs array
                    itemCount={logs.length}
                    itemSize={getItemSize} // Use the function for dynamic height
                    estimatedItemSize={40} // Provide an estimated size
                  >
                    {VirtualizedLogRow}
                  </List>
                )}
              </div>
              
              <p className="text-xs text-muted-foreground mt-2">
                {connectionStatus === 'connected' 
                  ? `Connected to log stream with ${filter.service !== 'all' ? filter.service + ' ' : ''}${filter.level !== 'all' ? filter.level + ' level ' : ''}filter${searchTerm ? ` and search term "${searchTerm}"` : ''}`
                  : 'Connect to start receiving log events'}
              </p>
            </CardContent>
          </Card>
        </TabsContent>
        
        {explainabilityEnabled && (
          <TabsContent value="merges">
            <div className="flex justify-between items-center mb-4">
              <Alert className="mb-0 flex-1">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Phase 5.9.2 REST Implementation</AlertTitle>
                <AlertDescription>
                  This view shows merged assembly events via REST API. WebSocket streaming will be implemented in Phase 5.9.3.
                </AlertDescription>
              </Alert>
              <Button 
                variant="outline" 
                size="sm" 
                className="ml-4"
                onClick={() => refetchMergeLogs()}
              >
                Refresh
              </Button>
            </div>
            
            <MergeLogView
              entries={mergeLogEntries}
              isLoading={mergeLogLoading}
              isError={mergeLogError}
              error={mergeLogErrorData}
            />
          </TabsContent>
        )}
      </Tabs>
    </>
  );
}
