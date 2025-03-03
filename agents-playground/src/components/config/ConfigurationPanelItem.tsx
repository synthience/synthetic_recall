import { ReactNode, useRef, useEffect } from "react";
import { PlaygroundDeviceSelector } from "@/components/playground/PlaygroundDeviceSelector";
import { TrackToggle } from "@livekit/components-react";
import { Track } from "livekit-client";
import { createTextFlicker } from "@/lib/animations";

type ConfigurationPanelItemProps = {
  title: string;
  children?: ReactNode;
  deviceSelectorKind?: MediaDeviceKind;
};

export const ConfigurationPanelItem: React.FC<ConfigurationPanelItemProps> = ({
  children,
  title,
  deviceSelectorKind,
}) => {
  const titleRef = useRef<HTMLHeadingElement>(null);
  
  useEffect(() => {
    if (titleRef.current) {
      // Add flickering effect to title for cyberpunk feel
      createTextFlicker(titleRef.current);
    }
  }, []);
  
  return (
    <div className="w-full text-gray-300 py-4 border-b border-b-gray-800/50 relative hud-panel mb-4">
      {/* Top edge cyberpunk accent */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-cyan-500/0 via-cyan-500/50 to-cyan-500/0"></div>
      
      <div className="flex flex-row justify-between items-center px-4 text-xs uppercase tracking-wider">
        <h3 
          ref={titleRef}
          className="text-cyan-400 font-mono relative text-glow tracking-widest py-1"
        >
          {title}
          
          {/* Line decoration under title */}
          <div className="absolute bottom-0 left-0 w-1/2 h-[1px] bg-cyan-500/30"></div>
        </h3>
        
        {deviceSelectorKind && (
          <span className="flex flex-row gap-2">
            <TrackToggle
              className="px-2 py-1 bg-gray-900/50 text-gray-300 border border-gray-800 rounded-sm hover:bg-gray-800/70 hover:border-cyan-500/30 transition-all"
              source={
                deviceSelectorKind === "audioinput"
                  ? Track.Source.Microphone
                  : Track.Source.Camera
              }
            />
            <PlaygroundDeviceSelector kind={deviceSelectorKind} />
          </span>
        )}
      </div>
      <div className="px-4 py-3 text-sm text-gray-400 leading-normal">
        {children}
      </div>
    </div>
  );
};