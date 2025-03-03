import { ReactNode, useState, useEffect, useRef } from "react";
import { createNeuralParticles } from "@/lib/animations";

const titleHeight = 32;

type PlaygroundTileProps = {
  title?: string;
  children?: ReactNode;
  className?: string;
  childrenClassName?: string;
  padding?: boolean;
  backgroundColor?: string;
};

export type PlaygroundTab = {
  title: string;
  content: ReactNode;
};

export type PlaygroundTabbedTileProps = {
  tabs: PlaygroundTab[];
  initialTab?: number;
} & PlaygroundTileProps;

export const PlaygroundTile: React.FC<PlaygroundTileProps> = ({
  children,
  title,
  className,
  childrenClassName,
  padding = true,
  backgroundColor = "transparent",
}) => {
  const contentPadding = padding ? 4 : 0;
  const tileRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (tileRef.current) {
      // Add neural particles for visual effect (low count for better performance)
      const cleanup = createNeuralParticles(tileRef.current, 5);
      return cleanup;
    }
  }, []);
  
  return (
    <div
      ref={tileRef}
      className={`
        flex flex-col relative glass-panel
        text-gray-300 bg-${backgroundColor}
        overflow-hidden
        transition-all duration-300
        hover:shadow-lg hover:shadow-cyan-500/10
        ${className || ""}
      `}
    >
      {title && (
        <div
          className="flex items-center justify-between text-xs uppercase py-2 px-4 border-b border-b-gray-800 tracking-wider text-cyan-400 text-glow"
          style={{
            height: `${titleHeight}px`,
            background: "rgba(0, 15, 30, 0.6)",
          }}
        >
          <h2 className="digital-flicker">{title}</h2>
          
          {/* Decorative elements for the cyberpunk HUD look */}
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 rounded-full bg-cyan-500/70 animate-pulse"></div>
            <div className="text-xs text-cyan-500/80 font-mono">SYN.ACTIVE</div>
          </div>
        </div>
      )}
      
      <div
        className={`
          flex flex-col items-center grow w-full 
          relative z-10
          ${childrenClassName || ""}
        `}
        style={{
          height: `calc(100% - ${title ? titleHeight + "px" : "0px"})`,
          padding: `${contentPadding * 4}px`,
        }}
      >
        {children}
      </div>
    </div>
  );
};

export const PlaygroundTabbedTile: React.FC<PlaygroundTabbedTileProps> = ({
  tabs,
  initialTab = 0,
  className,
  childrenClassName,
  backgroundColor = "transparent",
}) => {
  const contentPadding = 4;
  const [activeTab, setActiveTab] = useState(initialTab);
  const tileRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (tileRef.current) {
      // Add neural particles for visual effect
      const cleanup = createNeuralParticles(tileRef.current, 8);
      return cleanup;
    }
  }, []);
  
  if (activeTab >= tabs.length) {
    return null;
  }
  
  return (
    <div
      ref={tileRef}
      className={`
        flex flex-col h-full glass-panel 
        text-gray-400 bg-${backgroundColor}
        overflow-hidden
        transition-all duration-300
        ${className || ""}
      `}
    >
      <div
        className="flex items-center justify-start text-xs uppercase border-b border-b-gray-800 tracking-wider"
        style={{
          height: `${titleHeight}px`,
          background: "rgba(0, 15, 30, 0.6)",
        }}
      >
        {tabs.map((tab, index) => (
          <button
            key={index}
            className={`
              px-4 py-2 rounded-sm
              border-r border-r-gray-800
              transition-all duration-300
              ${
                index === activeTab
                  ? "bg-cyan-900/20 text-cyan-400 text-glow border-b-2 border-b-cyan-500"
                  : "bg-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800/30"
              }
            `}
            onClick={() => setActiveTab(index)}
          >
            {tab.title}
          </button>
        ))}
      </div>
      
      <div
        className={`w-full relative ${childrenClassName || ""}`}
        style={{
          height: `calc(100% - ${titleHeight}px)`,
          padding: `${contentPadding * 4}px`,
        }}
      >
        {tabs[activeTab].content}
      </div>
    </div>
  );
};