// src/hooks/useConfig.tsx
"use client";

import { getCookie, setCookie } from "cookies-next";
import jsYaml from "js-yaml";
import { useRouter } from "next/navigation";
import React, {
  createContext,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

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
  // Memory system settings
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

// Fallback if NEXT_PUBLIC_APP_CONFIG is not set
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
    // Default memory settings
    memory_enabled: false,
    memory_ws_url: "ws://localhost:5001",
    memory_hpc_url: "ws://localhost:5005"
  },
  show_qr: false,
};

const useAppConfig = (): AppConfig => {
  return useMemo(() => {
    if (process.env.NEXT_PUBLIC_APP_CONFIG) {
      try {
        const parsedConfig = jsYaml.load(
          process.env.NEXT_PUBLIC_APP_CONFIG
        ) as AppConfig;
        // Default to missing values in parsed config
        if (parsedConfig.settings === undefined) {
          parsedConfig.settings = defaultConfig.settings;
        }
        if (parsedConfig.settings.editable === undefined) {
          parsedConfig.settings.editable = defaultConfig.settings.editable;
        }
        if (parsedConfig.settings.memory_enabled === undefined) {
          parsedConfig.settings.memory_enabled = defaultConfig.settings.memory_enabled;
        }
        if (parsedConfig.settings.memory_ws_url === undefined) {
          parsedConfig.settings.memory_ws_url = defaultConfig.settings.memory_ws_url;
        }
        if (parsedConfig.settings.memory_hpc_url === undefined) {
          parsedConfig.settings.memory_hpc_url = defaultConfig.settings.memory_hpc_url;
        }
        return parsedConfig;
      } catch (e) {
        console.error("Error parsing app config:", e);
      }
    }
    return defaultConfig;
  }, []);
};

type ConfigData = {
  config: AppConfig;
  setUserSettings: (settings: UserSettings) => void;
};

const ConfigContext = createContext<ConfigData | undefined>(undefined);

export const ConfigProvider = ({ children }: { children: React.ReactNode }) => {
  const appConfig = useAppConfig();
  const router = useRouter();
  const [localColorOverride, setLocalColorOverride] = useState<string | null>(
    null
  );

  const getSettingsFromUrl = useCallback(() => {
    if (typeof window === "undefined") {
      return null;
    }
    if (!window.location.hash) {
      return null;
    }
    const appConfigFromSettings = appConfig;
    if (appConfigFromSettings.settings.editable === false) {
      return null;
    }
    
    try {
      const params = new URLSearchParams(window.location.hash.replace("#", ""));
      
      // Create memory URL defaults if memory is enabled but URLs not specified
      const memoryEnabled = params.get("memory") === "1";
      const memoryWsUrl = params.get("memory_ws") || "ws://localhost:5001";
      const memoryHpcUrl = params.get("memory_hpc") || "ws://localhost:5005";
      
      console.log(`URL memory settings - enabled: ${memoryEnabled}, ws: ${memoryWsUrl}, hpc: ${memoryHpcUrl}`);
      
      return {
        editable: true,
        chat: params.get("chat") === "1",
        theme_color: params.get("theme_color") || "cyan", // Provide default value to avoid null
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
        // Memory settings from URL
        memory_enabled: memoryEnabled,
        memory_ws_url: memoryWsUrl,
        memory_hpc_url: memoryHpcUrl
      };
    } catch (error) {
      console.error("Error parsing URL parameters:", error);
      return null;
    }
  }, [appConfig]);

  const getSettingsFromCookies = useCallback(() => {
    if (typeof window === 'undefined') {
      // Return default settings during server-side rendering
      return appConfig.settings;
    }
    
    const appConfigFromSettings = appConfig;
    if (appConfigFromSettings.settings.editable === false) {
      return null;
    }
    const jsonSettings = getCookie("lk_settings");
    console.log("Initial cookie settings:", jsonSettings);
    
    if (jsonSettings) {
      try {
        const parsedSettings = JSON.parse(jsonSettings as string);
        console.log("Parsed settings from cookies:", parsedSettings);
        return parsedSettings as UserSettings;
      } catch (e) {
        console.error('Error parsing stored settings:', e);
        return appConfigFromSettings.settings;
      }
    }
    return appConfigFromSettings.settings;
  }, [appConfig]);

  const setUrlSettings = useCallback(
    (us: UserSettings) => {
      if (typeof window === 'undefined') {
        // Skip URL updates during server-side rendering
        return;
      }
      
      try {
        const obj = new URLSearchParams({
          cam: boolToString(us.inputs.camera),
          mic: boolToString(us.inputs.mic),
          video: boolToString(us.outputs.video),
          audio: boolToString(us.outputs.audio),
          chat: boolToString(us.chat),
          theme_color: us.theme_color || "cyan",
          memory: boolToString(us.memory_enabled || false)
        });
        
        // Log the URL parameters for debugging
        console.log("URL settings:", obj.toString());
        
        // Add memory URLs if memory is enabled
        if (us.memory_enabled) {
          obj.set('memory_ws', us.memory_ws_url);
          obj.set('memory_hpc', us.memory_hpc_url);
        }
        
        // Note: We don't set ws_url and token to the URL on purpose
        if (typeof window !== 'undefined') {
          // Use window.location directly instead of router to avoid SSR issues
          const currentUrl = window.location.pathname + '#' + obj.toString();
          window.history.replaceState({}, '', currentUrl);
        }
      } catch (error) {
        console.error("Error updating URL settings:", error);
      }
    },
    [] // Remove router dependency since we're not using it anymore
  );

  const setCookieSettings = useCallback((us: UserSettings) => {
    if (typeof window === 'undefined') {
      // Skip cookie operations during server-side rendering
      return;
    }
    
    try {
      const json = JSON.stringify(us);
      setCookie("lk_settings", json);
      console.log("Saved settings to cookies:", json);
    } catch (error) {
      console.error("Error saving settings to cookies:", error);
    }
  }, []);

  const getConfig = useCallback(() => {
    const appConfigFromSettings = appConfig;

    if (appConfigFromSettings.settings.editable === false) {
      if (localColorOverride) {
        appConfigFromSettings.settings.theme_color = localColorOverride;
      }
      return appConfigFromSettings;
    }
    const cookieSettigs = getSettingsFromCookies();
    const urlSettings = getSettingsFromUrl();
    if (!cookieSettigs) {
      if (urlSettings) {
        setCookieSettings(urlSettings);
      }
    }
    if (!urlSettings) {
      if (cookieSettigs) {
        setUrlSettings(cookieSettigs);
      }
    }
    const newCookieSettings = getSettingsFromCookies();
    if (!newCookieSettings) {
      return appConfigFromSettings;
    }
    appConfigFromSettings.settings = newCookieSettings;
    return { ...appConfigFromSettings };
  }, [
    appConfig,
    getSettingsFromCookies,
    getSettingsFromUrl,
    localColorOverride,
    setCookieSettings,
    setUrlSettings,
  ]);

  const setUserSettings = useCallback(
    (settings: UserSettings) => {
      const appConfigFromSettings = appConfig;
      if (appConfigFromSettings.settings.editable === false) {
        setLocalColorOverride(settings.theme_color);
        return;
      }
      
      // Ensure memory_enabled is a boolean
      if (settings.memory_enabled === undefined) {
        settings.memory_enabled = false;
      }
      
      setUrlSettings(settings);
      setCookieSettings(settings);
      _setConfig((prev) => {
        return {
          ...prev,
          settings: settings,
        };
      });
    },
    [appConfig, setCookieSettings, setUrlSettings]
  );

  const [config, _setConfig] = useState<AppConfig>(getConfig());

  useEffect(() => {
    // Compare user settings in cookie with current state
    if (typeof window === 'undefined') {
      return; // Skip during server-side rendering
    }
    
    // Provide defaults for memory settings if they're undefined
    const currentSettings = config.settings;
    if (currentSettings.memory_enabled === undefined) {
      console.log('Initializing undefined memory_enabled to false');
      _setConfig(prev => ({
        ...prev,
        settings: {
          ...prev.settings,
          memory_enabled: false
        }
      }));
    }
    
    const storedSettings = getCookie('userSettings');
    
    if (storedSettings) {
      try {
        const parsedSettings = JSON.parse(storedSettings);
        
        // If there's a change from default for memory settings, update app config
        if (parsedSettings.memory_enabled !== undefined && 
            parsedSettings.memory_enabled !== currentSettings.memory_enabled) {
          console.log(`Updating memory_enabled from cookie: ${parsedSettings.memory_enabled}`);
          
          _setConfig(prev => ({
            ...prev,
            settings: {
              ...prev.settings,
              memory_enabled: parsedSettings.memory_enabled
            }
          }));
        }
      } catch (e) {
        console.error('Error parsing stored settings:', e);
      }
    }
  }, []);

  // Run things client side because we use cookies
  useEffect(() => {
    if (typeof window !== 'undefined') {
      _setConfig(getConfig());
    }
  }, [getConfig]);

  return (
    <ConfigContext.Provider value={{ config, setUserSettings }}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfig = () => {
  const context = React.useContext(ConfigContext);
  if (context === undefined) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  return context;
};

const boolToString = (b: boolean) => (b ? "1" : "0");