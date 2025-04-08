import { useState, useEffect, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useAssembly, useAssemblyLineage, useExplainMerge, useExplainActivation } from '@/lib/api'; // Base hooks
import { Assembly } from '@shared/schema'; // Import Assembly type

export function useAssemblyExplainabilityTabs(assemblyId: string | null | undefined, activeTab: string) {
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch core assembly data to check for valid memory IDs
  // Note: This assumes useAssembly hook is efficient and cached
  const { data: assemblyApiResponse } = useAssembly(assemblyId || null);
  const currentAssemblyMembers = assemblyApiResponse?.data?.memory_ids ?? [];

  // Initialize internal hooks
  const lineageQuery = useAssemblyLineage(assemblyId || null);
  const mergeQuery = useExplainMerge(assemblyId || null);
  const activationQuery = useExplainActivation(assemblyId || null, selectedMemoryId || undefined);

  // Handler to select memory and trigger activation refetch
  const selectMemory = useCallback((memoryId: string | null) => {
    console.log(`[ExplainabilityTabs] Memory selected: ${memoryId}`);
    setSelectedMemoryId(memoryId);

    // Trigger refetch immediately if tab is already 'activation'
    if (memoryId && activeTab === 'activation') {
      console.log(`[ExplainabilityTabs] Refetching activation for selected memory ${memoryId}`);
      // TODO: Phase 6.0+ - Consider integration with WebSocket/event subscriptions here
      if (!activationQuery.isFetching) {
        activationQuery.refetch();
      }
    }
  }, [activationQuery, activeTab]); // Dependencies: activationQuery, activeTab

  // Effect to trigger refetch on active tab change
  useEffect(() => {
    if (!assemblyId) {
      console.log('[ExplainabilityTabs] No assemblyId, skipping refetch.');
      return;
    }
    console.log(`[ExplainabilityTabs] Tab changed to: ${activeTab}`);

    const refetchBasedOnTab = async () => {
      try {
        // TODO: Phase 6.0+ - Consider integration with WebSocket/event subscriptions here
        switch (activeTab) {
          case 'lineage':
            if (!lineageQuery.isFetching) await lineageQuery.refetch();
            break;
          case 'merge':
            if (!mergeQuery.isFetching) await mergeQuery.refetch();
            break;
          case 'activation':
            // **Integrate Audit Point 3 (Memory Context Inheritance):**
            // Check if the currently selected memory is still valid for this assembly
            if (selectedMemoryId && !currentAssemblyMembers.includes(selectedMemoryId)) {
              console.warn(`[ExplainabilityTabs] Selected memory ${selectedMemoryId} no longer in assembly ${assemblyId}. Resetting selection.`);
              setSelectedMemoryId(null); // Reset if memory is gone
            } else if (selectedMemoryId && !activationQuery.isFetching) {
              // Only refetch activation if a valid memory IS selected
              await activationQuery.refetch();
            }
            break;
        }
      } catch (err) {
         console.error(`[ExplainabilityTabs] Error refetching for tab ${activeTab}:`, err);
      }
    };

    refetchBasedOnTab();
  }, [activeTab, assemblyId, selectedMemoryId, currentAssemblyMembers, lineageQuery, mergeQuery, activationQuery]); // Add selectedMemoryId & currentAssemblyMembers

  return {
    lineageQuery,
    mergeQuery,
    activationQuery,
    selectedMemoryId,
    selectMemory,
  };
}
