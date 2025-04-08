import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRuntimeConfig } from '../lib/api';
import { ApiResponse, RuntimeConfigResponse } from '@shared/schema';
import { toast } from '@/hooks/use-toast'; // Import toast

type FeaturesContextType = {
  explainabilityEnabled: boolean;
  isLoading: boolean;
  error: string | null;
  debugMode: boolean;
  usingFallbackConfig?: boolean; // Add fallback indicator
};

const FeaturesContext = createContext<FeaturesContextType>({
  explainabilityEnabled: false,
  isLoading: true,
  error: null,
  debugMode: false,
  usingFallbackConfig: false
});

export const FeaturesProvider = ({ children }: { children: ReactNode }) => {
  const { data: apiResponse, isLoading: configIsLoading, isError: configIsError, error: configError } = useRuntimeConfig('memory-core');
  
  // Start with a stable initial state - this prevents excessive re-renders
  const [featuresState, setFeaturesState] = useState<FeaturesContextType>({
    explainabilityEnabled: false, // Start disabled until config is loaded
    isLoading: true, // Start in loading state
    error: null,
    debugMode: false,
    usingFallbackConfig: false
  });
  
  // Only update the state ONCE when the config fetch settles
  useEffect(() => {
    console.log('[FeaturesContext] Config fetch status: isLoading:', configIsLoading, 'isError:', configIsError, 'has data:', !!apiResponse);

    // Only process when loading is complete (either success or error)
    if (!configIsLoading) {
      let newExplainability = false;
      let newDebug = false;
      let newError: string | null = null;
      let newIsFallback = false;
      let note = '';

      // CASE 1: We have a successful response with data
      if (apiResponse && apiResponse.success && apiResponse.data) {
        console.log('[FeaturesContext] Processing successful data:', apiResponse.data);
        
        const configData = apiResponse.data;
        
        // Check if we received fallback config (added by the proxy on error)
        if (configData._note?.includes('FALLBACK CONFIG')) {
          newIsFallback = true;
          note = configData._note;
          console.warn(`[FeaturesContext] Using fallback configuration: ${note}`);
          
          // For development, use fallback values (enable features)
          if (process.env.NODE_ENV === 'development') {
            newExplainability = true;
            newDebug = true;
            newError = `Using fallback configuration: ${note}`;
          }
        } else if (configData.config) {
          // Normal config data processing - use actual values
          newExplainability = configData.config.ENABLE_EXPLAINABILITY ?? false;
          newDebug = configData.config.DEBUG_MODE ?? false;
          newError = null; // Clear any previous error
          console.log('[FeaturesContext] Config loaded successfully. ENABLE_EXPLAINABILITY:', newExplainability);
        } else {
          // Response has unexpected structure - treat as fallback
          console.warn('[FeaturesContext] Successful response but unexpected data structure:', configData);
          newIsFallback = true;
          note = 'Response missing config field';
          
          // For development, enable features by default
          if (process.env.NODE_ENV === 'development') {
            newExplainability = true;
            newDebug = true;
          }
          newError = `Unexpected API response structure: ${note}`;
        }
      } 
      // CASE 2: We have an error or unsuccessful response
      else {
        const message = configIsError
          ? `Config Hook Error: ${configError?.message || 'Unknown error'}`
          : apiResponse ? `Config API Error: ${apiResponse.error || 'Unknown API error'}` : 'Config fetch failed';

        console.warn(`[FeaturesContext] Failed to load config: ${message}`);
        
        // Enable features by default in development even on error
        if (process.env.NODE_ENV === 'development') {
          newExplainability = true;
          newDebug = true;
          newIsFallback = true;
        }
        newError = message;
      }
      
      // Update state ONCE with all computed values
      const newState = {
        explainabilityEnabled: newExplainability,
        isLoading: false, // No longer loading
        error: newError,
        debugMode: newDebug,
        usingFallbackConfig: newIsFallback
      };
      
      console.log('[FeaturesContext] Setting new state:', newState);
      setFeaturesState(newState);
      
      // Display toast ONLY in development mode and only if using fallback
      if (newIsFallback && process.env.NODE_ENV === 'development') {
        toast({
          title: "Using Fallback Config",
          description: `Could not reach Memory Core config. Features ${newExplainability ? 'enabled' : 'disabled'} by default. Reason: ${newError || note}`,
          variant: "default",
          duration: 10000,
        });
      }
    }
    // Only depend on loading state and data, not internal state
  }, [configIsLoading, apiResponse, configIsError, configError]);
  
  return (
    <FeaturesContext.Provider value={featuresState}>
      {children}
      {/* Render a small persistent banner if using fallback in dev mode */}
      {featuresState.usingFallbackConfig && process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-2 left-2 bg-yellow-500 text-black text-xs p-1 rounded z-50">
          Using Fallback Config (404: /config/runtime/memory-core)
        </div>
      )}
    </FeaturesContext.Provider>
  );
};

export const useFeatures = () => useContext(FeaturesContext);
