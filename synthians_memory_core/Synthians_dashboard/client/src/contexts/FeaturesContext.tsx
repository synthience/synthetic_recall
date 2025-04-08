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
  const { data, isLoading, isError, error } = useRuntimeConfig('memory-core');
  const [explainabilityEnabled, setExplainabilityEnabled] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [usingFallbackConfig, setUsingFallbackConfig] = useState(false); // Track fallback usage
  
  useEffect(() => {
    console.log('[FeaturesContext] useEffect triggered. isLoading:', isLoading, 'isError:', isError, 'data:', data);

    if (!isLoading) {
      let isFallback = false;
      let note = '';

      // PRIORITIZE SUCCESSFUL DATA, even if 'isError' might be momentarily true from a failed retry
      if (data && (data as ApiResponse<RuntimeConfigResponse>).success && (data as ApiResponse<RuntimeConfigResponse>).data) {
        console.log('[FeaturesContext] Processing successful data:', (data as ApiResponse<RuntimeConfigResponse>).data);
        
        // For temporary solution: check if data.data directly contains config
        // or if it's health data with version/status
        let configData = (data as ApiResponse<RuntimeConfigResponse>).data;
        
        // If it looks like health data, create a temporary config
        // This is a temporary workaround until the proper config endpoint is fixed
        if (configData && 'status' in configData && 'version' in configData) {
          console.log('[FeaturesContext] Detected health data instead of config, creating temporary config');
          // Enable explainability features in development by default
          setExplainabilityEnabled(true);
          setDebugMode(true);
          setErrorMessage(null);
          isFallback = true;
          note = 'Using health data as config';
        } else if (configData?.config) {
          // Check for fallback config
          if (configData._note?.includes('FALLBACK CONFIG')) {
            isFallback = true;
            note = configData._note;
            console.warn(`[FeaturesContext] Using fallback configuration: ${note}`);
            
            // Display a dev-only toast/banner for fallback config
            if (process.env.NODE_ENV === 'development') {
              toast({
                title: "Using Fallback Config",
                description: `Could not reach Memory Core config. Explainability features enabled by default for dev. Reason: ${(configData as any)?._original_error || 'Unknown Proxy Error'}`,
                variant: "default", // Use default or a custom 'warning' variant
                duration: 10000, // Show for 10 seconds
              });
            }
          }

          // Normal config data processing
          const enabled = isFallback || (configData.config.ENABLE_EXPLAINABILITY ?? false);
          const debug = isFallback || (configData.config.DEBUG_MODE ?? false);
          console.log('[FeaturesContext] Config loaded successfully. ENABLE_EXPLAINABILITY:', enabled, 'Fallback:', isFallback);
          setExplainabilityEnabled(enabled);
          setDebugMode(debug);
          setErrorMessage(isFallback ? `Using fallback configuration: ${note}` : null);
        } else {
          console.warn('[FeaturesContext] Successful response but unexpected data structure:', configData);
          // Data exists but doesn't have expected structure - still better than nothing
          // Try to enable features for development if possible
          isFallback = typeof configData?._note === 'string' && (configData?._note?.includes('FALLBACK') ?? false);
          note = configData?._note || 'unknown structure';
          
          if (isFallback) {
            console.log('[FeaturesContext] Detected fallback data, enabling features for development');
            setExplainabilityEnabled(true);
            setDebugMode(true);
            setErrorMessage(`Using fallback configuration: ${note}`);
            
            // Display dev toast for fallback
            if (process.env.NODE_ENV === 'development') {
              toast({
                title: "Using Fallback Config",
                description: `Config has unexpected structure. Features enabled by default for dev.`,
                variant: "default",
                duration: 8000,
              });
            }
          } else {
            // Unknown data structure
            setExplainabilityEnabled(false);
            setDebugMode(false);
            setErrorMessage('Unexpected API response structure');
          }
        }
      } else {
        // Handle error or unsuccessful response ONLY IF data is not successfully present
        const message = isError
          ? `Hook Error: ${error?.message || 'Unknown error'}`
          : data ? `API Error: ${(data as ApiResponse<RuntimeConfigResponse>).error || 'Unknown API error'}` : 'Data fetch failed or response malformed';

        console.warn(`[FeaturesContext] Failed to load config OR data invalid: ${message}`);
        console.warn('[FeaturesContext] Using default configuration (explainability=false)');

        setExplainabilityEnabled(false);
        setDebugMode(false);
        setErrorMessage(message);
        isFallback = false;
      }
      
      // Update fallback status
      setUsingFallbackConfig(isFallback);
    }
  }, [data, isLoading, isError, error]);
  
  // Debug information
  useEffect(() => {
    console.log('[FeaturesContext] Current state:', {
      isLoading,
      isError,
      hasData: !!data,
      dataSuccess: data ? (data as ApiResponse<RuntimeConfigResponse>).success : undefined,
      explainabilityEnabled,
      debugMode,
      errorMessage,
      usingFallbackConfig
    });
  }, [isLoading, isError, data, explainabilityEnabled, debugMode, errorMessage, usingFallbackConfig]);
  
  return (
    <FeaturesContext.Provider value={{ 
      explainabilityEnabled, 
      isLoading, 
      error: errorMessage, 
      debugMode, 
      usingFallbackConfig 
    }}>
      {children}
      {/* Render a small persistent banner if using fallback in dev mode */}
      {usingFallbackConfig && process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-2 left-2 bg-yellow-500 text-black text-xs p-1 rounded z-50">
          Using Fallback Config
        </div>
      )}
    </FeaturesContext.Provider>
  );
};

export const useFeatures = () => useContext(FeaturesContext);
