import { useMediaDeviceSelect } from "@livekit/components-react";
import { useEffect, useState } from "react";

type PlaygroundDeviceSelectorProps = {
  kind: MediaDeviceKind;
};

export const PlaygroundDeviceSelector = ({
  kind,
}: PlaygroundDeviceSelectorProps) => {
  const [showMenu, setShowMenu] = useState(false);
  const deviceSelect = useMediaDeviceSelect({ kind: kind });
  const [selectedDeviceName, setSelectedDeviceName] = useState("");

  useEffect(() => {
    deviceSelect.devices.forEach((device) => {
      if (device.deviceId === deviceSelect.activeDeviceId) {
        setSelectedDeviceName(device.label);
      }
    });
  }, [deviceSelect.activeDeviceId, deviceSelect.devices, selectedDeviceName]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showMenu) {
        setShowMenu(false);
      }
    };
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [showMenu]);

  return (
    <div className="relative z-20">
      <button
        className="flex gap-2 items-center px-2 py-1 bg-gray-900/60 text-cyan-400 border border-gray-800 
        rounded-sm hover:bg-gray-800/80 hover:border-cyan-500/30 transition-all duration-300"
        onClick={(e) => {
          setShowMenu(!showMenu);
          e.stopPropagation();
        }}
      >
        <span className="max-w-[80px] overflow-ellipsis overflow-hidden whitespace-nowrap font-mono text-xs tracking-wide">
          {selectedDeviceName || "Select Device"}
        </span>
        <ChevronSVG />
      </button>
      
      <div
        className="absolute right-0 top-8 bg-gray-900/95 text-gray-300 border border-cyan-500/20 
        rounded-sm z-10 w-48 backdrop-blur-sm overflow-hidden shadow-lg shadow-cyan-500/10"
        style={{
          display: showMenu ? "block" : "none",
        }}
      >
        {deviceSelect.devices.map((device, index) => {
          return (
            <div
              onClick={() => {
                deviceSelect.setActiveMediaDevice(device.deviceId);
                setShowMenu(false);
              }}
              className={`
                ${device.deviceId === deviceSelect.activeDeviceId ? "text-cyan-400 bg-gray-800/70" : "text-gray-500 bg-gray-900/80"} 
                text-xs py-2 px-3 cursor-pointer hover:bg-gray-800 hover:text-cyan-300 transition-colors
                border-b border-gray-800/50 last:border-b-0 font-mono
              `}
              key={index}
            >
              {device.label}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const ChevronSVG = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="12"
    height="12"
    viewBox="0 0 16 16"
    fill="none"
    className="opacity-70"
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M3 5H5V7H3V5ZM7 9V7H5V9H7ZM9 9V11H7V9H9ZM11 7V9H9V7H11ZM11 7V5H13V7H11Z"
      fill="currentColor"
      fillOpacity="0.8"
    />
  </svg>
);