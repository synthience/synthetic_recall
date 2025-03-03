import { useConfig } from "@/hooks/useConfig";
import { CLOUD_ENABLED, CloudConnect } from "../cloud/CloudConnect";
import { Button } from "./button/Button";
import { useState, useEffect, useRef } from "react";
import { ConnectionMode } from "@/hooks/useConnection";
import { createGlitchEffect, createNeuralParticles } from "@/lib/animations";

type PlaygroundConnectProps = {
  accentColor: string;
  onConnectClicked: (mode: ConnectionMode) => void;
};

const ConnectTab = ({ active, onClick, children }: any) => {
  let className = "px-4 py-2 text-sm tracking-wide uppercase font-mono transition-all duration-300";

  if (active) {
    className += " border-b-2 border-cyan-500 text-cyan-400 text-glow";
  } else {
    className += " text-gray-500 border-b border-transparent hover:text-gray-300";
  }

  return (
    <button className={className} onClick={onClick}>
      {children}
    </button>
  );
};

const TokenConnect = ({
  accentColor,
  onConnectClicked,
}: PlaygroundConnectProps) => {
  const { setUserSettings, config } = useConfig();
  const [url, setUrl] = useState(config.settings.ws_url);
  const [token, setToken] = useState(config.settings.token);
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      // Add neural particles for visual effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 10);
      // Add glitch effect for cyberpunk feel
      const cleanupGlitch = createGlitchEffect(containerRef.current, 0.5);
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  return (
    <div 
      ref={containerRef}
      className="flex flex-col gap-6 p-8 bg-gray-950/80 w-full text-white border-t border-cyan-900/30 glass-panel relative overflow-hidden"
    >
      <div className="flex flex-col gap-4 relative z-10">
        <div className="text-xs text-cyan-400 uppercase tracking-wider mb-2 digital-flicker">Enter Neural Link Parameters</div>
        
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="text-cyan-300 text-sm bg-black/50 border border-cyan-800/30 rounded-sm px-4 py-3 
            focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/30
            font-mono placeholder-cyan-800/50 transition-all duration-300"
          placeholder="wss://neural.link.url"
        />
        
        <textarea
          value={token}
          onChange={(e) => setToken(e.target.value)}
          className="text-cyan-300 text-sm bg-black/50 border border-cyan-800/30 rounded-sm px-4 py-3 
            focus:border-cyan-500/50 focus:outline-none focus:ring-1 focus:ring-cyan-500/30 
            font-mono placeholder-cyan-800/50 min-h-[100px] transition-all duration-300"
          placeholder="Neural link authentication token..."
        />
      </div>
      
      <Button
        accentColor={accentColor}
        className="w-full py-3 text-lg tracking-wider"
        onClick={() => {
          const newSettings = { ...config.settings };
          newSettings.ws_url = url;
          newSettings.token = token;
          setUserSettings(newSettings);
          onConnectClicked("manual");
        }}
      >
        Initialize Neural Link
      </Button>
      
      <a
        href="https://kitt.livekit.io/"
        className={`text-xs text-${accentColor}-500 hover:text-${accentColor}-400 text-center transition-colors duration-300 underline`}
      >
        Don't have credentials? Try the KITT demo environment.
      </a>
      
      {/* Scan line effect */}
      <div className="absolute inset-0 scan-line"></div>
    </div>
  );
};

export const PlaygroundConnect = ({
  accentColor,
  onConnectClicked,
}: PlaygroundConnectProps) => {
  const [showCloud, setShowCloud] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      // Add neural particles for visual effect
      const cleanupParticles = createNeuralParticles(containerRef.current, 15);
      
      return () => {
        cleanupParticles();
      };
    }
  }, []);
  
  const copy = CLOUD_ENABLED
    ? "Initialize neural interface with LiveKit Cloud or manually with access credentials"
    : "Initialize neural interface with your access credentials";
    
  return (
    <div 
      ref={containerRef}
      className="flex left-0 top-0 w-full h-full bg-black/90 items-center justify-center text-center gap-2 relative overflow-hidden"
    >
      {/* Holographic grid overlay */}
      <div className="absolute inset-0 pointer-events-none holo-grid"></div>
      
      <div className="min-h-[540px] relative z-10">
        <div className="flex flex-col bg-gray-950/80 w-full max-w-[520px] rounded-md text-white border border-cyan-900/30 glass-panel overflow-hidden">
          <div className="flex flex-col gap-2">
            <div className="px-10 space-y-4 py-8">
              <h1 className="text-2xl text-cyan-400 tracking-wider font-mono digital-flicker">
                SYNTHIENCE.AI
              </h1>
              <p className="text-sm text-gray-400 tracking-wide">
                {copy}
              </p>
            </div>
            
            {CLOUD_ENABLED && (
              <div className="flex justify-center pt-2 gap-4 border-b border-t border-gray-900/70">
                <ConnectTab
                  active={showCloud}
                  onClick={() => {
                    setShowCloud(true);
                  }}
                >
                  Cloud Access
                </ConnectTab>
                <ConnectTab
                  active={!showCloud}
                  onClick={() => {
                    setShowCloud(false);
                  }}
                >
                  Manual Access
                </ConnectTab>
              </div>
            )}
          </div>
          
          <div className="flex flex-col bg-gray-900/30 flex-grow">
            {showCloud && CLOUD_ENABLED ? (
              <CloudConnect accentColor={accentColor} />
            ) : (
              <TokenConnect
                accentColor={accentColor}
                onConnectClicked={onConnectClicked}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
};