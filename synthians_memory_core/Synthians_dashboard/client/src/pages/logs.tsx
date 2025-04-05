import React, { useState, useEffect, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";

// Mock log data structure (will be replaced with WebSocket data in the future)
interface LogEntry {
  timestamp: string;
  level: "DEBUG" | "INFO" | "WARN" | "ERROR";
  service: "MemoryCore" | "NeuralMemory" | "CCE";
  message: string;
}

// Component for individual log entry
function LogEntryRow({ entry }: { entry: LogEntry }) {
  const levelColors = {
    DEBUG: "text-gray-400",
    INFO: "text-blue-400",
    WARN: "text-yellow-400",
    ERROR: "text-red-400"
  };
  
  const serviceColors = {
    MemoryCore: "border-secondary",
    NeuralMemory: "border-primary",
    CCE: "border-accent"
  };
  
  return (
    <div className={`px-3 py-2 border-l-2 ${serviceColors[entry.service]} text-xs font-mono mb-1 hover:bg-muted`}>
      <span className="text-gray-500 mr-2">{entry.timestamp}</span>
      <Badge variant="outline" className={`${levelColors[entry.level]} mr-2`}>
        {entry.level}
      </Badge>
      <Badge variant="outline" className="mr-2">
        {entry.service}
      </Badge>
      <span>{entry.message}</span>
    </div>
  );
}

export default function Logs() {
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [serviceFilter, setServiceFilter] = useState<string>("all");
  const [levelFilter, setLevelFilter] = useState<string>("all");
  const [searchTerm, setSearchTerm] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Create placeholder text explaining this is a future feature
  const placeholderInfo = (
    <div className="text-center py-12">
      <i className="fas fa-stream text-4xl text-muted-foreground mb-4"></i>
      <h3 className="text-lg font-medium mb-2">Real-time Log Viewer</h3>
      <p className="text-muted-foreground mb-6 max-w-md mx-auto">
        This feature will connect to a WebSocket endpoint on each service to stream logs in real-time.
        It is currently a placeholder for a future implementation.
      </p>
      <div className="flex justify-center">
        <Button disabled className="mr-2">
          Connect to Log Stream
        </Button>
      </div>
    </div>
  );
  
  // Filter logs based on selected filters
  const filteredLogs = React.useMemo(() => {
    return logEntries.filter(entry => {
      // Apply service filter
      if (serviceFilter !== "all" && entry.service !== serviceFilter) {
        return false;
      }
      
      // Apply level filter
      if (levelFilter !== "all" && entry.level !== levelFilter) {
        return false;
      }
      
      // Apply search filter
      if (searchTerm && !entry.message.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      return true;
    });
  }, [logEntries, serviceFilter, levelFilter, searchTerm]);
  
  // Handle auto-scrolling
  useEffect(() => {
    if (autoScroll && scrollAreaRef.current) {
      const scrollArea = scrollAreaRef.current;
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }
  }, [filteredLogs, autoScroll]);
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Real-time Log Viewer</h2>
          <p className="text-sm text-gray-400">
            Stream logs from Synthians Cognitive Architecture services for real-time debugging
          </p>
        </div>
        
        <div className="flex items-center">
          <Badge 
            variant={isConnected ? "default" : "outline"} 
            className={isConnected ? "bg-green-600" : "text-gray-400"}
          >
            <div className={`w-2 h-2 rounded-full mr-1 ${isConnected ? "bg-background" : "bg-gray-400"}`}></div>
            {isConnected ? "Connected" : "Disconnected"}
          </Badge>
        </div>
      </div>
      
      {/* Log Controls */}
      <Card className="mb-6">
        <CardHeader className="pb-2">
          <CardTitle>Log Stream Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-500 mb-2">Service Filter</p>
              <Select value={serviceFilter} onValueChange={setServiceFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by service" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Services</SelectItem>
                  <SelectItem value="MemoryCore">Memory Core</SelectItem>
                  <SelectItem value="NeuralMemory">Neural Memory</SelectItem>
                  <SelectItem value="CCE">CCE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <p className="text-sm text-gray-500 mb-2">Log Level</p>
              <Select value={levelFilter} onValueChange={setLevelFilter}>
                <SelectTrigger>
                  <SelectValue placeholder="Filter by level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="DEBUG">DEBUG</SelectItem>
                  <SelectItem value="INFO">INFO</SelectItem>
                  <SelectItem value="WARN">WARN</SelectItem>
                  <SelectItem value="ERROR">ERROR</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <p className="text-sm text-gray-500 mb-2">Search</p>
              <Input
                placeholder="Search logs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
          
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Switch
                id="auto-scroll"
                checked={autoScroll}
                onCheckedChange={setAutoScroll}
              />
              <label htmlFor="auto-scroll" className="text-sm">Auto-scroll</label>
            </div>
            
            <div>
              <Button disabled={isConnected} variant="outline" className="mr-2">
                Connect
              </Button>
              <Button disabled={!isConnected} variant="outline">
                Clear Logs
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
      
      {/* Log Viewer */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex justify-between items-center">
            <CardTitle>Log Stream</CardTitle>
            <Badge variant="outline">
              {filteredLogs.length} entries
            </Badge>
          </div>
          <CardDescription>
            Streaming logs will appear here once connected
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-card border border-border rounded-md h-[500px]">
            {placeholderInfo}
            
            {/* Log entries would go here in a ScrollArea once implemented */}
            {/* <ScrollArea className="h-[500px] p-2" ref={scrollAreaRef}>
              {filteredLogs.map((entry, index) => (
                <LogEntryRow key={index} entry={entry} />
              ))}
              {filteredLogs.length === 0 && (
                <div className="text-center py-8 text-gray-400">
                  <i className="fas fa-info-circle mr-2"></i>
                  No log entries match your filters
                </div>
              )}
            </ScrollArea> */}
          </div>
        </CardContent>
      </Card>
    </>
  );
}
