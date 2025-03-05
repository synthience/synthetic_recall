import React, { ButtonHTMLAttributes, ReactNode, useEffect, useRef } from "react";
import { addGlowTrail, addDistortionPulse } from "@/lib/animations";

type ButtonProps = {
  accentColor: string;
  children: ReactNode;
  className?: string;
  disabled?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>;

export const Button: React.FC<ButtonProps> = ({
  accentColor,
  children,
  className = "",
  disabled = false,
  ...allProps
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (buttonRef.current && !disabled) {
      // Add interactive effects
      addGlowTrail(buttonRef.current);
      addDistortionPulse(buttonRef.current);
    }
  }, [disabled]);

  return (
    <button
      ref={buttonRef}
      className={`relative flex flex-row ${
        disabled ? "pointer-events-none opacity-50" : ""
      } text-gray-200 text-sm justify-center border border-${accentColor}-500/40 px-6 py-2 rounded-md transition-all duration-300 
      hover:shadow-lg hover:shadow-${accentColor}-500/30 hover:scale-105 
      active:scale-[0.98] ${className}`}
      disabled={disabled}
      {...allProps}
    >
      {/* Neon gradient background */}
      <span 
        className={`absolute inset-0 bg-gradient-to-r from-${accentColor}-500/10 via-transparent to-${accentColor}-500/10 rounded-md ${
          disabled ? "" : "animate-pulse"
        }`}
      ></span>
      
      {/* Button content */}
      <span className="relative z-10 flex items-center gap-2">
        {children}
      </span>
    </button>
  );
};