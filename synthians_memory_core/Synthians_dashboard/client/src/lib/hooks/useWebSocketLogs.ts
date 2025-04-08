import { useState, useEffect, useCallback, useRef } from 'react';
import { getWebSocketManager, LogMessage, ConnectionStatus, LogLevel } from '../websocket';

/**
 * Filter options for log messages
 */
export interface LogFilter {
  service?: 'memory-core' | 'neural-memory' | 'cce' | 'all';
  level?: LogLevel | 'all';
  search?: string;
}

/**
 * Result of the useWebSocketLogs hook
 */
export interface UseWebSocketLogsResult {
  logs: LogMessage[];
  connectionStatus: ConnectionStatus;
  connect: () => Promise<void>;
  disconnect: () => void;
  clearLogs: () => void;
  filter: LogFilter;
  setFilter: (filter: Partial<LogFilter>) => void;
  isConnecting: boolean;
  error: Error | null;
}

/**
 * Configuration options for the useWebSocketLogs hook
 */
export interface UseWebSocketLogsOptions {
  maxLogs?: number; // Maximum number of logs to keep in memory
  autoConnect?: boolean; // Whether to connect automatically
  initialFilter?: LogFilter; // Initial filter to apply
}

/**
 * Helper function to determine if a log should be shown based on filter
 */
function shouldShowLog(log: LogMessage, filter: LogFilter): boolean {
  // Filter by service
  if (filter.service && filter.service !== 'all' && log.service !== filter.service) {
    return false;
  }

  // Filter by log level
  if (filter.level && filter.level !== 'all' && log.level !== filter.level) {
    return false;
  }

  // Filter by search text
  if (filter.search && filter.search.trim() !== '') {
    const searchLower = filter.search.toLowerCase();
    const messageLower = log.message.toLowerCase();
    const serviceStr = log.service.toString().toLowerCase();
    const levelStr = log.level.toString().toLowerCase();
    
    // Search in message, service, level, and context (if available)
    if (
      !messageLower.includes(searchLower) &&
      !serviceStr.includes(searchLower) &&
      !levelStr.includes(searchLower) &&
      !(log.context && JSON.stringify(log.context).toLowerCase().includes(searchLower))
    ) {
      return false;
    }
  }

  return true;
}

/**
 * Hook to interact with WebSocket logs
 * Provides real-time log streaming with filtering capabilities
 */
export function useWebSocketLogs(options: UseWebSocketLogsOptions = {}): UseWebSocketLogsResult {
  const {
    maxLogs = 1000,
    autoConnect = false,
    initialFilter = { service: 'all', level: 'all', search: '' }
  } = options;

  // State for logs and connection status
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [filter, setFilterState] = useState<LogFilter>(initialFilter);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Refs to avoid dependencies in useEffect
  const filterRef = useRef<LogFilter>(initialFilter);
  const maxLogsRef = useRef<number>(maxLogs);
  // Store ALL received logs before filtering (critical for Phase 5.9.4)
  const allReceivedLogsRef = useRef<LogMessage[]>([]);

  // Update refs when state changes
  useEffect(() => {
    filterRef.current = filter;
  }, [filter]);

  useEffect(() => {
    maxLogsRef.current = maxLogs;
  }, [maxLogs]);

  // Function to set filter - enhanced for Phase 5.9.4
  const setFilter = useCallback((newFilter: Partial<LogFilter>) => {
    // Update filter reference first to ensure latest filter is used in any concurrent operations
    const updatedFilter = { ...filterRef.current, ...newFilter };
    filterRef.current = updatedFilter;

    // Re-filter ALL received logs based on the NEW filter
    const filteredLogs = allReceivedLogsRef.current
      .filter(log => shouldShowLog(log, updatedFilter));
    
    // Apply maxLogs limit AFTER filtering
    setLogs(filteredLogs.slice(0, maxLogsRef.current));
    
    // Update filter state to trigger re-render
    setFilterState(updatedFilter);
  }, []);

  // Function to add a new log message - enhanced for Phase 5.9.4
  const addLog = useCallback((log: LogMessage) => {
    // Always add to unfiltered logs collection
    allReceivedLogsRef.current = [log, ...allReceivedLogsRef.current];
    
    // Enforce internal buffer size limit (2x the display limit to allow for filtering)
    if (allReceivedLogsRef.current.length > maxLogsRef.current * 2) {
      allReceivedLogsRef.current = allReceivedLogsRef.current.slice(0, maxLogsRef.current * 2);
    }
    
    // Check if the log passes the current filter
    if (shouldShowLog(log, filterRef.current)) {
      // Add to the visible logs state
      setLogs(prevLogs => {
        const newLogs = [log, ...prevLogs];
        // Apply maxLogs limit for display
        return newLogs.slice(0, maxLogsRef.current);
      });
    }
  }, []);

  // Function to clear logs - enhanced for Phase 5.9.4
  const clearLogs = useCallback(() => {
    // Clear both the filtered and unfiltered logs
    allReceivedLogsRef.current = [];
    setLogs([]);
  }, []);

  // Handle WebSocket connection
  const connect = useCallback(async () => {
    try {
      setIsConnecting(true);
      setError(null);
      const wsManager = getWebSocketManager();
      await wsManager.connect();
      // Request history on successful connection
      if (wsManager.getStatus() === 'connected') {
        try {
          wsManager.send({ type: 'getLogs' });
        } catch (historyErr) {
          console.warn('Failed to request log history:', historyErr);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to connect to log stream'));
    } finally {
      setIsConnecting(false);
    }
  }, []);

  // Handle WebSocket disconnection
  const disconnect = useCallback(() => {
    const wsManager = getWebSocketManager();
    wsManager.disconnect();
  }, []);

  // Initialize WebSocket listeners
  useEffect(() => {
    const wsManager = getWebSocketManager();

    // Set up connection status listener
    const unsubscribeStatus = wsManager.onConnectionStatus(status => {
      setConnectionStatus(status);
      
      // Auto-request log history when connected
      if (status === 'connected') {
        try {
          wsManager.send({ type: 'getLogs' });
        } catch (err) {
          console.warn('Failed to request log history:', err);
        }
      }
    });

    // Set up message listener with improved handling for Phase 5.9.4
    const handleMessage = (data: any) => {
      try {
        // Handle different message types
        if (data && typeof data === 'object') {
          if (data.type === 'history' && Array.isArray(data.logs)) {
            // Process batch of logs (history)
            // Add newest logs first (reverse order) to ensure correct sorting
            const historyLogs = [...data.logs].reverse();
            
            // Add to unfiltered collection
            allReceivedLogsRef.current = [...historyLogs, ...allReceivedLogsRef.current];
            
            // Apply internal buffer size limit
            if (allReceivedLogsRef.current.length > maxLogsRef.current * 2) {
              allReceivedLogsRef.current = allReceivedLogsRef.current.slice(0, maxLogsRef.current * 2);
            }
            
            // Filter and update visible logs
            const filteredLogs = historyLogs.filter(log => shouldShowLog(log, filterRef.current));
            
            setLogs(prevLogs => {
              const combinedLogs = [...filteredLogs, ...prevLogs];
              return combinedLogs.slice(0, maxLogsRef.current);
            });
          } 
          else if (data.type === 'logsCleared') {
            // Handle logs cleared message
            clearLogs();
          }
          else if (!data.type) {
            // Regular single log message
            addLog(data as LogMessage);
          }
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    };

    const unsubscribeMessage = wsManager.addMessageHandler(handleMessage);

    // Auto-connect if specified
    if (autoConnect) {
      connect();
    }

    // Clean up listeners on unmount
    return () => {
      unsubscribeStatus();
      unsubscribeMessage();
    };
  }, [autoConnect, connect, addLog, clearLogs]);

  return {
    logs,
    connectionStatus,
    connect,
    disconnect,
    clearLogs,
    filter,
    setFilter,
    isConnecting,
    error
  };
}
