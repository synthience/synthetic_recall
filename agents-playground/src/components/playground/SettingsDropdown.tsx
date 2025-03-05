// src/components/playground/SettingsDropdown.tsx
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useConfig } from "@/hooks/useConfig";
import React, { useState } from "react";

// Extend the base UserSettings type from useConfig
import type { UserSettings as BaseUserSettings } from "@/hooks/useConfig";

export interface UserSettings extends BaseUserSettings {
  memory_enabled: boolean;
  memory_ws_url: string;
  memory_hpc_url: string;
}

type SettingType = "chat" | "memory" | "inputs" | "outputs" | "separator";

type Setting = {
  title: string;
  type: SettingType;
  key: keyof UserSettings | "camera" | "mic" | "video" | "audio" | "separator_1" | "separator_2" | "separator_3";
};

const settings: Setting[] = [
  {
    title: "Show chat",
    type: "chat",
    key: "chat",
  },
  {
    title: "---",
    type: "separator",
    key: "separator_1",
  },
  {
    title: "Camera",
    type: "inputs",
    key: "camera",
  },
  {
    title: "Microphone",
    type: "inputs",
    key: "mic",
  },
  {
    title: "---",
    type: "separator",
    key: "separator_2",
  },
  {
    title: "Video output",
    type: "outputs",
    key: "video",
  },
  {
    title: "Audio output",
    type: "outputs",
    key: "audio",
  },
  {
    title: "---",
    type: "separator",
    key: "separator_3",
  },
  {
    title: "Memory System",
    type: "memory",
    key: "memory_enabled",
  }
];

// SVG icons as React components
const CheckIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
  >
    <g clipPath="url(#clip0_718_9977)">
      <path
        d="M1.5 7.5L4.64706 10L10.5 2"
        stroke="white"
        strokeWidth="1.5"
        strokeLinecap="square"
      />
    </g>
    <defs>
      <clipPath id="clip0_718_9977">
        <rect width="12" height="12" fill="white" />
      </clipPath>
    </defs>
  </svg>
);

const ChevronIcon: React.FC = () => (
  <svg
    width="16" 
    height="16"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className="w-3 h-3 fill-gray-200 transition-all group-hover:fill-white"
    style={{ transform: "rotate(0deg)", transition: "transform 0.2s ease" }}
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="m8 10.7.4-.3 4-4 .3-.4-.7-.7-.4.3L8 9.3 4.4 5.6 4 5.3l-.7.7.3.4 4 4 .4.3Z"
    />
  </svg>
);

export const SettingsDropdown = () => {
  const { config, setUserSettings } = useConfig();
  const [isOpen, setIsOpen] = useState(false);

  const getValue = (setting: Setting): boolean | string | undefined => {
    if (setting.type === "separator") return undefined;
    
    if (setting.type === "chat") {
      return config.settings.chat;
    }
    if (setting.type === "memory") {
      const key = setting.key as "memory_enabled" | "memory_ws_url" | "memory_hpc_url";
      // Ensure memory_enabled has a boolean value
      if (key === "memory_enabled") {
        const memoryEnabled = (config.settings as UserSettings)[key];
        return memoryEnabled !== undefined ? memoryEnabled : false;
      }
      return (config.settings as UserSettings)[key];
    }
    if (setting.type === "inputs") {
      const key = setting.key as "camera" | "mic";
      return config.settings.inputs[key];
    } 
    if (setting.type === "outputs") {
      const key = setting.key as "video" | "audio";
      return config.settings.outputs[key];
    }
    return undefined;
  };

  const handleChange = (setting: Setting, newValue: boolean | string) => {
    if (setting.type === "separator") return;
    
    const newSettings = { ...config.settings } as UserSettings;
    console.log("handleChange called with setting:", setting, "and new value:", newValue);
    if (setting.type === "chat") {
      newSettings.chat = newValue as boolean;
    } else if (setting.type === "memory") {
      const key = setting.key as "memory_enabled" | "memory_ws_url" | "memory_hpc_url";
      if (key === "memory_enabled") {
        // Ensure the value is a boolean
        const memoryEnabled = Boolean(newValue);
        newSettings.memory_enabled = memoryEnabled;
        console.log(`Memory system ${memoryEnabled ? 'enabled' : 'disabled'} in settings`, newSettings);
        
        // Update memory service state if the component is loaded
        try {
          const { getMemoryService } = require('@/lib/memoryService');
          const memoryService = getMemoryService();
          memoryService.setEnabled(memoryEnabled);
          console.log("Successfully updated memory service directly");
        } catch (e) {
          console.warn('Could not update memory service directly:', e);
        }
      } else if (key === "memory_ws_url" || key === "memory_hpc_url") {
        newSettings[key] = newValue as string;
      }
    } else if (setting.type === "inputs") {
      const key = setting.key as "camera" | "mic";
      newSettings.inputs[key] = newValue as boolean;
    } else if (setting.type === "outputs") {
      const key = setting.key as "video" | "audio";
      newSettings.outputs[key] = newValue as boolean;
    }
    console.log("New settings:", newSettings);
    setUserSettings(newSettings);
  };

  return (
    <DropdownMenu.Root modal={false} onOpenChange={setIsOpen}>
      <DropdownMenu.Trigger asChild>
        <button className="group inline-flex items-center gap-1 rounded-md py-1 px-2 text-gray-300 hover:bg-gray-800 hover:text-gray-100 transition-colors">
          <span className="text-sm">Settings</span>
          <svg
            width="16" 
            height="16"
            fill="currentColor"
            xmlns="http://www.w3.org/2000/svg"
            className="w-3 h-3 transition-transform duration-200"
            style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0)' }}
          >
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              d="m8 10.7.4-.3 4-4 .3-.4-.7-.7-.4.3L8 9.3 4.4 5.6 4 5.3l-.7.7.3.4 4 4 .4.3Z"
            />
          </svg>
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          sideOffset={5}
          align="start"
          collisionPadding={16}
          className="z-50 animate-fadeIn"
        >
          <div 
            style={{ minWidth: "240px" }} 
            className="bg-gray-900 rounded-md overflow-hidden shadow-xl border border-gray-800 backdrop-blur-sm"
          >
            <div className="py-1 px-2 text-xs uppercase tracking-wider text-cyan-400 border-b border-gray-800">
              System Settings
            </div>
            
            {settings.map((setting) => {
              if (setting.type === "separator") {
                return (
                  <div
                    key={setting.key}
                    className="h-[1px] bg-gray-800 mx-3 my-1"
                  />
                );
              }

              const value = getValue(setting);
              return (
                <DropdownMenu.Item
                  key={setting.key}
                  onSelect={(e) => {
                    e?.preventDefault();
                    if (typeof value === 'boolean') {
                      handleChange(setting, !value);
                    }
                  }}
                  className="flex items-center gap-3 px-3 py-2 text-sm hover:bg-gray-800 cursor-pointer outline-none text-gray-300 hover:text-cyan-300 transition-colors"
                >
                  <div className={`w-5 h-5 flex items-center justify-center border rounded-sm ${value ? 'bg-cyan-900/30 border-cyan-500/50' : 'bg-gray-900/50 border-gray-700'}`}>
                    {value && <CheckIcon />}
                  </div>
                  <span>{setting.title}</span>
                </DropdownMenu.Item>
              );
            })}
            
            {/* Advanced settings section */}
            {Boolean((config.settings as UserSettings).memory_enabled) && (
              <>
                <div className="h-[1px] bg-gray-800 mx-3 my-1" />
                <div className="py-1 px-2 text-xs uppercase tracking-wider text-cyan-400 border-b border-gray-800">
                  Memory Settings
                </div>
                <div className="p-3 text-xs text-gray-400">
                  Memory system is enabled. Configure connection settings in the Memory panel.
                </div>
              </>
            )}
            
            <div className="p-2 flex justify-end border-t border-gray-800">
              <a 
                href="https://docs.livekit.io/agents" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-xs text-cyan-500 hover:text-cyan-400 transition-colors"
              >
                Learn more
              </a>
            </div>
          </div>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
};