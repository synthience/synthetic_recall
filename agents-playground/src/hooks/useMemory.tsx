// src/hooks/useMemory.tsx
import { useState, useEffect, useCallback, useRef } from 'react';
import { getMemoryService } from '@/lib/memoryService';

// Define hook props with default URLs
interface UseMemoryProps {
  defaultTensorUrl?: string;
  defaultHpcUrl?: string;
  enabled?: boolean;
}

type MemoryMetric = {
  id: string;
  text: string;
  similarity: number;
  significance: number;
  surprise: number;
  timestamp: number;
};

type MemoryStats = {
  memory_count: number;
  gpu_memory: number;
  active_connections: number;
};

// Define memory hook
export const useMemory = ({ defaultTensorUrl, defaultHpcUrl, enabled }: UseMemoryProps = {}) => {
  console.log("useMemory hook initialized with URLs:", { defaultTensorUrl, defaultHpcUrl });
  
  // State for URLs
  const tensorUrlRef = useRef<string>(defaultTensorUrl || 'ws://localhost:5001');
  const hpcUrlRef = useRef<string>(defaultHpcUrl || 'ws://localhost:5005');

  // Memory enabled state
  const [memoryEnabled, setMemoryEnabled] = useState<boolean>(enabled !== undefined ? enabled : false);
  
  // Connection states
  const [connectionStatus, setConnectionStatus] = useState<string>('Disconnected');
  const [hpcStatus, setHpcStatus] = useState<string>('Disconnected');
  
  // Results and selection states
  const [memoryResults, setMemoryResults] = useState<MemoryMetric[]>([]);
  const [selectedMemories, setSelectedMemories] = useState<Set<string>>(new Set());
  
  // Stats and metrics
  const [stats, setStats] = useState<MemoryStats>({
    memory_count: 0,
    gpu_memory: 0,
    active_connections: 0
  });
  
  const [processingMetrics, setProcessingMetrics] = useState<{ 
    significance: number[], 
    surprise: number[] 
  }>({
    significance: Array(5).fill(0.5),
    surprise: Array(5).fill(0.25)
  });

  // Initialize memory service
  const initialize = useCallback((tensorUrl?: string, hpcUrl?: string) => {
    if (tensorUrl) tensorUrlRef.current = tensorUrl;
    if (hpcUrl) hpcUrlRef.current = hpcUrl;

    console.log(`Initializing memory service with enabled state: ${memoryEnabled}`);
    const memoryService = getMemoryService(tensorUrlRef.current, hpcUrlRef.current);
    
    // Make sure our local state is synced with the memory service
    setConnectionStatus('Connecting');
    memoryService.initialize()
      .then((success: boolean) => {
        if (success) {
          console.log('Memory system initialized successfully');
          
          // Set memory service enabled state based on our enabled state
          memoryService.setEnabled(memoryEnabled);
        } else {
          console.error('Failed to initialize memory system');
        }
      })
      .catch((error: Error) => {
        console.error('Error initializing memory system:', error);
        setConnectionStatus('Error');
      });
  }, [memoryEnabled]);

  // Disconnect memory service
  const disconnect = useCallback(() => {
    const memoryService = getMemoryService();
    memoryService.disconnect();
    setConnectionStatus('Disconnected');
    setHpcStatus('Disconnected');
  }, []);

  // Toggle connection
  const toggleConnection = useCallback(() => {
    if (connectionStatus === 'Connected') {
      disconnect();
    } else {
      initialize();
    }
  }, [connectionStatus, disconnect, initialize]);

  // Toggle memory system enabled/disabled
  const toggleMemorySystem = useCallback(() => {
    const memoryService = getMemoryService();
    
    const newEnabledState = !memoryEnabled;
    console.log(`Toggling memory system to: ${newEnabledState}`);
    
    memoryService.setEnabled(newEnabledState);
    // State will be updated through the event handler
  }, [memoryEnabled]);

  // Set search text (for external components)
  const setSearchText = useCallback((text: string) => {
    // Currently just a passthrough but could add preprocessing
    return text;
  }, []);

  // Search for memories
  const search = useCallback((query: string) => {
    if (!query.trim()) return;
    
    const memoryService = getMemoryService();
    memoryService.search(query);
  }, []);

  // Clear search results
  const clearSearch = useCallback(() => {
    setMemoryResults([]);
  }, []);

  // Toggle selection of a memory
  const toggleSelection = useCallback((id: string) => {
    setSelectedMemories(prev => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      
      // Update memory service with selection
      const memoryService = getMemoryService();
      memoryService.updateSelection(Array.from(newSet));
      
      return newSet;
    });
  }, []);

  // Set up event handlers
  useEffect(() => {
    const memoryService = getMemoryService(tensorUrlRef.current, hpcUrlRef.current);
    
    // Register event handlers
    const statusHandler = (data: any) => {
      setConnectionStatus(data.status);
    };
    
    const hpcStatusHandler = (data: any) => {
      setHpcStatus(data.status);
    };
    
    const searchResultsHandler = (data: any) => {
      if (data.results) {
        // Ensure each result has surprise metric (fallback to calculated value if not provided)
        const enhancedResults = data.results.map((result: any, index: number) => {
          // If surprise is not provided, generate one based on significance
          const surprise = result.surprise !== undefined ? 
            result.surprise : 
            Math.min(1.0, Math.max(0.1, result.significance * (1 + Math.random() * 0.5)));
            
          return {
            ...result,
            surprise
          };
        });
        
        setMemoryResults(enhancedResults);
      }
    };
    
    const statsHandler = (data: any) => {
      setStats(prev => ({
        ...prev,
        memory_count: data.memory_count || prev.memory_count,
        gpu_memory: data.gpu_memory || prev.gpu_memory,
        active_connections: data.active_connections || prev.active_connections
      }));
    };
    
    const selectionChangedHandler = (data: any) => {
      setSelectedMemories(new Set(data.selectedMemories));
    };
    
    const enabledChangedHandler = (data: any) => {
      setMemoryEnabled(data.enabled);
    };
    
    const memoryProcessedHandler = (data: any) => {
      if (data.significance !== undefined || data.surprise !== undefined) {
        setProcessingMetrics(prev => {
          // Add new metrics to the beginning of the arrays and keep only last 5
          const newSignificance = [...prev.significance];
          const newSurprise = [...prev.surprise];
          
          if (data.significance !== undefined) {
            newSignificance.unshift(data.significance);
            newSignificance.length = Math.min(newSignificance.length, 5);
          }
          
          if (data.surprise !== undefined) {
            newSurprise.unshift(data.surprise);
            newSurprise.length = Math.min(newSurprise.length, 5);
          }
          
          return {
            significance: newSignificance,
            surprise: newSurprise
          };
        });
      }
    };
    
    // Register callbacks
    memoryService.on('status', statusHandler);
    memoryService.on('hpc_status', hpcStatusHandler);
    memoryService.on('search_results', searchResultsHandler);
    memoryService.on('stats', statsHandler);
    memoryService.on('selection_changed', selectionChangedHandler);
    memoryService.on('enabled_changed', enabledChangedHandler);
    memoryService.on('memory_processed', memoryProcessedHandler);
    
    // Initialize if default URLs are provided
    if (defaultTensorUrl && defaultHpcUrl) {
      initialize(defaultTensorUrl, defaultHpcUrl);
    }
    
    // Cleanup on unmount
    return () => {
      memoryService.off('status', statusHandler);
      memoryService.off('hpc_status', hpcStatusHandler);
      memoryService.off('search_results', searchResultsHandler);
      memoryService.off('stats', statsHandler);
      memoryService.off('selection_changed', selectionChangedHandler);
      memoryService.off('enabled_changed', enabledChangedHandler);
      memoryService.off('memory_processed', memoryProcessedHandler);
    };
  }, [defaultTensorUrl, defaultHpcUrl, initialize]);

  return {
    connectionStatus,
    hpcStatus,
    memoryEnabled,
    setSearchText,
    search,
    clearSearch,
    selectedMemories,
    toggleSelection,
    stats,
    results: memoryResults,
    processingMetrics,
    toggleMemorySystem,
    
    // Return URLs
    memoryWsUrl: tensorUrlRef.current,
    memoryHpcUrl: hpcUrlRef.current,
    
    // For backward compatibility
    memoryResults,
    searchMemory: search,
    toggleMemorySelection: toggleSelection,
    clearSelectedMemories: clearSearch,
    processInput: (text: string) => console.log('Legacy processInput called', text),
    toggleConnection: () => console.log('Legacy toggleConnection called')
  };
}