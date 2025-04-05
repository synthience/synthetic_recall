import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRuntimeConfig } from '../lib/api';

type FeaturesContextType = {
  explainabilityEnabled: boolean;
  isLoading: boolean;
};

const FeaturesContext = createContext<FeaturesContextType>({
  explainabilityEnabled: false,
  isLoading: true,
});

export const FeaturesProvider = ({ children }: { children: ReactNode }) => {
  const { data, isLoading } = useRuntimeConfig('memory-core');
  const [explainabilityEnabled, setExplainabilityEnabled] = useState(false);
  
  useEffect(() => {
    if (data?.config) {
      // Check if the ENABLE_EXPLAINABILITY flag is present in the config
      setExplainabilityEnabled(!!data.config.ENABLE_EXPLAINABILITY);
    }
  }, [data]);
  
  return (
    <FeaturesContext.Provider value={{ explainabilityEnabled, isLoading }}>
      {children}
    </FeaturesContext.Provider>
  );
};

export const useFeatures = () => useContext(FeaturesContext);
