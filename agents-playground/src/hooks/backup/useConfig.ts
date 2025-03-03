"use client";

import { getCookie, setCookie } from "cookies-next";
import jsYaml from "js-yaml";
import { useRouter } from "next/navigation";
import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type UserSettings = {
  editable: boolean;
  theme_color: string;
  chat: boolean;
  inputs: {
    camera: boolean;
    mic: boolean;
  };
  outputs: {
    audio: boolean;
    video: boolean;
  };
  ws_url: string;
  token: string;
  memory_enabled: boolean;
  memory_ws_url: string;
  memory_hpc_url: string;
};

export type AppConfig = {
  title: string;
  description: string;
  github_link?: string;
  video_fit?: "cover" | "contain";
  settings: UserSettings;
  show_qr?: boolean;
};

// Default config
const defaultConfig: AppConfig = {
  title: "SYNTHIENCE Neural Playground",
  description: "A lucid neural presence—real-time, memory-driven, and always aware.",
  video_fit: "cover",
  settings: {
    editable: true,
    theme_color: "cyan",
    chat: true,
    inputs: {
      camera: true,
      mic: true,
    },
    outputs: {
      audio: true,
      video: true,
    },
    ws_url: "",
    token: "",
    memory_enabled: false,
    memory_ws_url: "ws://localhost:5001",
    memory_hpc_url: "ws://localhost:5005",
  },
  show_qr: false,
};

// Convert boolean to "1" or "0" string
const boolToString = (b: boolean): string => (b ? "1" : "0");

// Hook to get app config
const useAppConfig = (): AppConfig => {
  return useMemo(() => {
    if (process.env.NEXT_PUBLIC_APP_CONFIG) {
      try {
        const parsedConfig = jsYaml.load(process.env.NEXT_PUBLIC_APP_CONFIG) as AppConfig;
        
        // Set defaults for missing values
        if (!parsedConfig.settings) {
          parsedConfig.settings = { ...defaultConfig.settings };
        }
        
        if (parsedConfig.settings.editable === undefined) {
          parsedConfig.settings.editable = true;
        }
        
        if (parsedConfig.settings.memory_enabled === undefined) {
          parsedConfig.settings.memory_enabled = false;
        }
        
        if (parsedConfig.settings.memory_ws_url === undefined) {
          parsedConfig.settings.memory_ws_url = "ws://localhost:5001";
        }
        
        if (parsedConfig.settings.memory_hpc_url === undefined) {
          parsedConfig.settings.memory_hpc_url = "ws://localhost:5005";
        }
        
        if (parsedConfig.description === undefined) {
          parsedConfig.description = defaultConfig.description;
        }
        
        if (parsedConfig.video_fit === undefined) {
          parsedConfig.video_fit = "cover";
        }
        
        if (parsedConfig.show_qr === undefined) {
          parsedConfig.show_qr = false;
        }
        
        return parsedConfig;
      } catch (e) {
        console.error("Error parsing app config:", e);
      }
    }
    return { ...defaultConfig };
  }, []);
};

// Config context type
interface ConfigContextType {
  config: AppConfig;
  setUserSettings: (settings: UserSettings) => void;
}

// Create context
const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

// Provider component
export const ConfigProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const appConfig = useAppConfig();
  const router = useRouter();
  const [localColorOverride, setLocalColorOverride] = useState<string | null>(null);
  const [config, setConfig] = useState<AppConfig>(defaultConfig);
  
  // Get settings from URL parameters
  const getSettingsFromUrl = useCallback(() => {
    if (typeof window === "undefined" || !window.location.hash) {
      return null;
    }
    
    if (!appConfig.settings.editable) {
      return null;
    }
    
    const params = new URLSearchParams(window.location.hash.replace("#", ""));
    
    return {
      editable: true,
      theme_color: params.get("theme_color") || defaultConfig.settings.theme_color,
      chat: params.get("chat") === "1",
      inputs: {
        camera: params.get("cam") === "1",
        mic: params.get("mic") === "1",
      },
      outputs: {
        audio: params.get("audio") === "1",
        video: params.get("video") === "1",
      },
      ws_url: "",
      token: "",
      memory_enabled: params.get("memory") === "1",
      memory_ws_url: params.get("memory_ws") || "ws://localhost:5001",
      memory_hpc_url: params.get("memory_hpc") || "ws://localhost:5005",
    } as UserSettings;
  }, [appConfig.settings.editable]);
  
  // Get settings from cookies
  const getSettingsFromCookies = useCallback(() => {
    if (!appConfig.settings.editable) {
      return null;
    }
    
    const jsonSettings = getCookie("lk_settings");
    if (!jsonSettings) {
      return null;
    }
    
    try {
      return JSON.parse(jsonSettings as string) as UserSettings;
    } catch (e) {
      console.error("Error parsing settings from cookies:", e);
      return null;
    }
  }, [appConfig.settings.editable]);
  
  // Save settings to URL
  const setUrlSettings = useCallback((settings: UserSettings) => {
    const params = new URLSearchParams({
      cam: boolToString(settings.inputs.camera),
      mic: boolToString(settings.inputs.mic),
      video: boolToString(settings.outputs.video),
      audio: boolToString(settings.outputs.audio),
      chat: boolToString(settings.chat),
      theme_color: settings.theme_color || "cyan",
      memory: boolToString(settings.memory_enabled),
    });
    
    // Add memory URLs if memory is enabled
    if (settings.memory_enabled) {
      params.set("memory_ws", settings.memory_ws_url);
      params.set("memory_hpc", settings.memory_hpc_url);
    }
    
    router.replace("/#" + params.toString());
  }, [router]);
  
  // Save settings to cookies
  const setCookieSettings = useCallback((settings: UserSettings) => {
    try {
      const json = JSON.stringify(settings);
      setCookie("lk_settings", json);
    } catch (e) {
      console.error("Error saving settings to cookies:", e);
    }
  }, []);
  
  // Get config with settings from URL or cookies
  const getConfig = useCallback(() => {
    const result = { ...appConfig };
    
    // If settings are not editable, just set color override if any
    if (!result.settings.editable) {
      if (localColorOverride) {
        result.settings.theme_color = localColorOverride;
      }
      return result;
    }
    
    // Try to get settings from cookies or URL
    const cookieSettings = getSettingsFromCookies();
    const urlSettings = getSettingsFromUrl();
    
    // Sync settings between cookies and URL
    if (!cookieSettings && urlSettings) {
      setCookieSettings(urlSettings);
    }
    
    if (!urlSettings && cookieSettings) {
      setUrlSettings(cookieSettings);
    }
    
    // Get updated cookie settings
    const newSettings = getSettingsFromCookies();
    if (newSettings) {
      result.settings = newSettings;
    }
    
    return result;
  }, [
    appConfig,
    localColorOverride,
    getSettingsFromCookies,
    getSettingsFromUrl,
    setCookieSettings,
    setUrlSettings,
  ]);
  
  // Update user settings
  const setUserSettings = useCallback((settings: UserSettings) => {
    // If settings are not editable, just update color
    if (!appConfig.settings.editable) {
      setLocalColorOverride(settings.theme_color);
      return;
    }
    
    // Save settings to URL and cookies
    setUrlSettings(settings);
    setCookieSettings(settings);
    
    // Update local state
    setConfig((prev) => ({
      ...prev,
      settings: settings,
    }));
  }, [appConfig.settings.editable, setUrlSettings, setCookieSettings]);
  
  // Initialize config
  useEffect(() => {
    setConfig(getConfig());
  }, [getConfig]);
  
  // Create memoized context value
  const contextValue = useMemo(() => ({
    config,
    setUserSettings,
  }), [config, setUserSettings]);
  
  return (
    <ConfigContext.Provider value={contextValue}>
      {children}
    </ConfigContext.Provider>
  );
};

// Hook to use config
export const useConfig = () => {
  const context = useContext(ConfigContext);
  
  if (!context) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  
  return context;
};