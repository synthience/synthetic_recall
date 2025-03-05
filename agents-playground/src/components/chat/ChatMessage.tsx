import React, { useEffect, useRef, memo } from "react";
import { addScanLineEffect } from "@/lib/animations";

type ChatMessageProps = {
  message: string;
  accentColor: string;
  name: string;
  isSelf: boolean;
  hideName?: boolean;
};

export const ChatMessage = memo(({
  name,
  message,
  accentColor,
  isSelf,
  hideName,
}: ChatMessageProps) => {
  const messageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messageRef.current) {
      // Add scan line effect to message
      const cleanupScanLine = addScanLineEffect(messageRef.current);
      return () => cleanupScanLine();
    }
  }, []);

  return (
    <div 
      className={`flex flex-col gap-1 ${hideName ? "pt-0" : "pt-4"}`}
    >
      {!hideName && (
        <div
          className={`text-${
            isSelf ? "gray-500" : accentColor + "-400"
          } uppercase text-xs tracking-wider`}
        >
          {name}
        </div>
      )}
      
      <div
        ref={messageRef}
        className={`
          relative glass-panel p-3 
          ${isSelf ? "bg-gray-900/30" : `bg-${accentColor}-900/20`}
          border ${isSelf ? "border-gray-700/30" : `border-${accentColor}-500/30`}
          shadow-sm ${isSelf ? "" : `shadow-${accentColor}-500/20`}
          hover:shadow-md hover:shadow-${accentColor}-500/30 
          transition-all duration-300
          max-w-[90%] ${isSelf ? "ml-auto" : "mr-auto"}
          overflow-hidden
        `}
      >
        <div
          className={`pr-4 text-${
            isSelf ? "gray-300" : accentColor + "-300"
          } text-sm whitespace-pre-line`}
        >
          {message}
        </div>
        
        {/* Holographic overlay */}
        <div 
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `linear-gradient(90deg, 
              transparent 0%, 
              rgba(${isSelf ? "255, 255, 255" : "0, 255, 255"}, 0.05) 25%, 
              transparent 50%
            )`,
            opacity: "0.3"
          }}
        ></div>
      </div>
    </div>
  );
});