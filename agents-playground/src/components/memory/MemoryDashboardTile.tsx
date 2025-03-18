// src/components/memory/MemoryDashboardTile.tsx
import React, { useEffect, useRef, useState } from 'react';
import { useMemory } from '@/hooks/useMemory';
import { ConfigurationPanelItem } from '@/components/config/ConfigurationPanelItem';
import { NameValueRow } from '@/components/config/NameValueRow';
import { Button } from '@/components/button/Button';
import { createTextFlicker, createNeuralParticles, createGlitchEffect } from '@/lib/animations';
import MemoryMetrics from './MemoryMetrics';
import { useConfig } from '@/hooks/useConfig';

type MemoryDashboardTileProps = {
  accentColor: string;
};

export const MemoryDashboardTile: React.FC<MemoryDashboardTileProps> = ({
  accentColor
}) => {
  console.log("MemoryDashboardTile rendering with accentColor:", accentColor);
  
  // Get configuration settings
  const { config } = useConfig();
  console.log("MemoryDashboardTile loaded config settings:", config.settings);
  
  // Ensure memory settings exist with defaults
  const memory_enabled = config.settings.memory_enabled !== undefined ? 
    config.settings.memory_enabled : false;
  const memory_ws_url = config.settings.memory_ws_url || "ws://localhost:5001";
  const memory_hpc_url = config.settings.memory_hpc_url || "ws://localhost:5005";

  // Use memory hook with config URLs
  const { 
    connectionStatus,
    hpcStatus,
    memoryEnabled,
    setSearchText,
    search,
    clearSearch,
    selectedMemories,
    toggleSelection,
    stats,
    results: searchResults,
    processingMetrics,
    toggleMemorySystem,
    memoryWsUrl,
    memoryHpcUrl
  } = useMemory({
    defaultTensorUrl: memory_ws_url,
    defaultHpcUrl: memory_hpc_url,
    enabled: memory_enabled
  });
  
  console.log("Memory state in MemoryDashboardTile:", { 
    memory_enabled,
    memoryEnabled,
    connectionStatus,
    hpcStatus,
    memoryWsUrl,
    memoryHpcUrl  
  });
  
  const [searchQuery, setSearchQuery] = useState<string>('');
  const containerRef = useRef<HTMLDivElement>(null);
  const titleRefs = useRef<(HTMLElement | null)[]>([]);

  // Apply cyberpunk effects to the component
  useEffect(() => {
    if (containerRef.current) {
      // Add neural particles for visual effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 10);
      
      // Add occasional glitch effect for cyberpunk feel
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.3);
      
      // Add text flicker effects to titles
      titleRefs.current.forEach(el => {
        if (el) createTextFlicker(el);
      });
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  // Sync memory system enabled state with config
  useEffect(() => {
    // If memory state doesn't match config, update it
    console.log(`Memory sync effect: config.enabled=${memory_enabled}, hook.enabled=${memoryEnabled}`);
    
    if (memory_enabled !== undefined && memory_enabled !== memoryEnabled) {
      console.log(`Syncing memory system state from config: ${memory_enabled}`);
      toggleMemorySystem();
    }
  }, [memory_enabled, memoryEnabled, toggleMemorySystem]);

  // Handle search submission
  const handleSearch = () => {
    if (!searchQuery.trim()) return;
    search(searchQuery);
  };

  // Handle input key press (Enter)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div 
      ref={containerRef}
      className="w-full h-full flex flex-col gap-3 overflow-y-auto scrollbar-thin scrollbar-track-gray-900/20 scrollbar-thumb-cyan-500/20"
    >
      {/* Neural Memory System Status */}
      <ConfigurationPanelItem title="Neural Memory System">
        <div className="flex flex-col gap-2">
          <NameValueRow
            name="Memory Link"
            value={
              <span className={`status-active ${connectionStatus === 'Connected' ? '' : 'text-red-400'}`}>
                {connectionStatus}
              </span>
            }
            valueColor={connectionStatus === 'Connected' ? `${accentColor}-400` : 'red-400'}
          />
          <NameValueRow
            name="HPC Engine"
            value={
              <span className={`status-active ${hpcStatus === 'Connected' ? '' : 'text-red-400'}`}>
                {hpcStatus}
              </span>
            }
            valueColor={hpcStatus === 'Connected' ? `${accentColor}-400` : 'red-400'}
          />
          <NameValueRow
            name="Memory Count"
            value={stats.memory_count}
            valueColor={`${accentColor}-400`}
          />
          <NameValueRow
            name="Neural Processing"
            value={`${stats.gpu_memory.toFixed(2)} GB`}
            valueColor={`${accentColor}-400`}
          />
          <NameValueRow
            name="Memory Status"
            value={memoryEnabled ? "Enabled" : "Disabled"}
            valueColor={memoryEnabled ? 'green-400' : 'red-400'}
          />
          <div className="flex justify-between mt-2">
            <Button
              accentColor={memoryEnabled ? 'red' : 'green'}
              onClick={toggleMemorySystem}
              className="text-xs py-1 px-3"
            >
              {memoryEnabled ? 'DISABLE MEMORY' : 'ENABLE MEMORY'}
            </Button>
            <Button
              accentColor={connectionStatus === 'Connected' ? 'red' : accentColor}
              onClick={() => {}}
              className="text-xs py-1 px-3"
            >
              {connectionStatus === 'Connected' ? 'DISCONNECT' : 'CONNECT'}
            </Button>
          </div>
        </div>
      </ConfigurationPanelItem>

      {/* Memory Search Interface */}
      <ConfigurationPanelItem title="Memory Search">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={handleKeyPress}
              className={`
                w-full text-sm bg-transparent 
                text-${accentColor}-100 p-2 rounded-sm 
                focus:outline-none focus:ring-1 focus:ring-${accentColor}-700
                transition-all duration-300 backdrop-blur-sm
                bg-gray-900/20 border border-${accentColor}-800/30
              `}
              placeholder="Search neural memories..."
            />
            <Button
              accentColor={accentColor}
              onClick={handleSearch}
              className="text-xs py-1 px-3"
            >
              SEARCH
            </Button>
          </div>
        </div>
      </ConfigurationPanelItem>

      {/* Memory Results */}
      {searchResults.length > 0 && (
        <ConfigurationPanelItem title="Memory Results">
          <div className="space-y-3 max-h-[300px] overflow-y-auto pr-1">
            {searchResults.map((memory) => (
              <div
                key={memory.id}
                className={`
                  relative p-3 border rounded-sm transition-all duration-300
                  ${selectedMemories.has(memory.id) 
                    ? `bg-${accentColor}-900/20 border-${accentColor}-500/50` 
                    : 'bg-gray-900/30 border-gray-800/30'
                  }
                  cursor-pointer hover:bg-${accentColor}-900/10
                `}
                onClick={() => toggleSelection(memory.id)}
              >
                {/* Left accent bar */}
                <div 
                  className={`absolute left-0 top-0 bottom-0 w-1 ${
                    selectedMemories.has(memory.id) ? `bg-${accentColor}-500` : `bg-${accentColor}-900/30`
                  }`}
                ></div>
                
                {/* Memory metrics */}
                <div className="flex flex-wrap gap-2 mb-2">
                  <div className={`text-${accentColor}-400 text-xs font-mono bg-${accentColor}-900/20 px-2 py-1 rounded-sm`}>
                    Match: {(memory.similarity * 100).toFixed(1)}%
                  </div>
                  <div className="text-amber-400 text-xs font-mono bg-amber-900/20 px-2 py-1 rounded-sm">
                    Quickrecal Score: {memory.quickrecal_score.toFixed(2)}
                  </div>
                  <div className="text-violet-400 text-xs font-mono bg-violet-900/20 px-2 py-1 rounded-sm">
                    Surprise: {memory.surprise.toFixed(2)}
                  </div>
                </div>
                
                {/* Memory text */}
                <div className="text-gray-300 text-sm mt-2 border-l-2 border-gray-700/50 pl-2">
                  {memory.text}
                </div>
                
                {/* Holographic overlay */}
                <div 
                  className="absolute inset-0 pointer-events-none opacity-10"
                  style={{
                    background: `linear-gradient(90deg, 
                      transparent 0%, 
                      rgba(0, 255, 255, 0.05) 25%, 
                      transparent 50%
                    )`
                  }}
                ></div>
              </div>
            ))}
          </div>
          
          {/* Actions */}
          {selectedMemories.size > 0 && (
            <div className="flex justify-between items-center mt-3 pt-2 border-t border-gray-800/30">
              <div className={`text-xs text-${accentColor}-400 font-mono`}>
                {selectedMemories.size} memories selected
              </div>
              <Button
                accentColor={accentColor}
                onClick={() => {}}
                className="text-xs py-1 px-3"
              >
                CLEAR
              </Button>
            </div>
          )}
        </ConfigurationPanelItem>
      )}

      {/* Metric Visualizations */}
      <ConfigurationPanelItem title="Neural Metrics">
        <div className="flex flex-col gap-4">
          {/* Significance & Surprise Visualization */}
          {searchResults.length > 0 && (
            <div className="w-full h-[140px] border border-gray-800/50 bg-gray-900/30 rounded-sm p-3 relative">
              <div className={`text-xs text-${accentColor}-400 mb-2 font-mono`}>Memory Significance & Surprise</div>
              
              <div className="flex h-[80px] w-full items-end justify-around relative">
                {/* Dynamic grid lines */}
                <div className="absolute inset-0 grid grid-cols-10 grid-rows-4 border-t border-l border-gray-800/30">
                  {[...Array(10)].map((_, i) => (
                    <div key={`gridcol-${i}`} className="border-r border-gray-800/20 h-full"></div>
                  ))}
                  {[...Array(4)].map((_, i) => (
                    <div key={`gridrow-${i}`} className="border-b border-gray-800/20 w-full"></div>
                  ))}
                </div>
                
                {/* Bars */}
                {searchResults.slice(0, 5).map((memory, index) => (
                  <div key={`metric-${memory.id}`} className="flex gap-1 h-full items-end z-10">
                    {/* Quickrecal Score bar */}
                    <div 
                      className="w-4 bg-gradient-to-t from-amber-500 to-amber-300 relative group"
                      style={{ height: `${memory.quickrecal_score * 100}%` }}
                    >
                      <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 hidden group-hover:block">
                        <div className="text-xs text-amber-400 whitespace-nowrap bg-gray-900/80 px-1 py-0.5 rounded">
                          {memory.quickrecal_score.toFixed(2)}
                        </div>
                      </div>
                    </div>
                    
                    {/* Surprise bar */}
                    <div 
                      className="w-4 bg-gradient-to-t from-violet-500 to-violet-300 relative group"
                      style={{ height: `${memory.surprise * 100}%` }}
                    >
                      <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 hidden group-hover:block">
                        <div className="text-xs text-violet-400 whitespace-nowrap bg-gray-900/80 px-1 py-0.5 rounded">
                          {memory.surprise.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Legend */}
              <div className="flex justify-center gap-4 mt-2">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-amber-500"></div>
                  <span className="text-xs text-gray-400">Quickrecal Score</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-violet-500"></div>
                  <span className="text-xs text-gray-400">Surprise</span>
                </div>
              </div>
              
              {/* Scan line effect */}
              <div className="absolute inset-0 scan-line pointer-events-none"></div>
            </div>
          )}
        </div>
      </ConfigurationPanelItem>

      {/* Memory System Status */}
      <div className={`flex flex-col gap-2 p-4 rounded-sm border border-${accentColor}-500/20 bg-black/30`}>
        <div className={`text-${accentColor}-400 text-sm font-mono`}>
          Memory System Status
        </div>
        <div className={`text-${accentColor}-500/70 text-xs font-mono`}>
          {connectionStatus === 'Connected' ? (
            <span className="text-green-400">● Online</span>
          ) : connectionStatus === 'Connecting' ? (
            <span className="text-yellow-400">● Connecting...</span>
          ) : (
            <span className="text-red-400">● Offline</span>
          )}
          {memoryEnabled && connectionStatus === 'Connected' ? (
            <span className="ml-2 text-green-400">| Memory Enabled</span>
          ) : connectionStatus === 'Connected' ? (
            <span className="ml-2 text-yellow-400">| Memory Disabled</span>
          ) : null}
        </div>
      </div>

      {/* Neural Activity Visualization */}
      <MemoryMetrics 
        accentColor={accentColor} 
        quickrecal_score={processingMetrics?.quickrecal_score}
        surprise={processingMetrics?.surprise}
      />

      {/* Connection Details */}
      <div className={`flex flex-col gap-2 p-4 rounded-sm border border-${accentColor}-500/20 bg-black/30`}>
        <div className={`text-${accentColor}-400 text-sm font-mono`}>
          Connection Details
        </div>
        <div className={`text-${accentColor}-500/70 text-xs font-mono flex flex-col gap-1`}>
          <div>Tensor Server: {memoryWsUrl}</div>
          <div>HPC Server: {memoryHpcUrl}</div>
        </div>
      </div>
    </div>
  );
};

export default MemoryDashboardTile;