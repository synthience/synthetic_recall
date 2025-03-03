import { useState, useEffect, useRef } from "react";

type ColorPickerProps = {
  colors: string[];
  selectedColor: string;
  onSelect: (color: string) => void;
};

export const ColorPicker = ({
  colors,
  selectedColor,
  onSelect,
}: ColorPickerProps) => {
  const [isHovering, setIsHovering] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const onMouseEnter = () => {
    setIsHovering(true);
  };
  
  const onMouseLeave = () => {
    setIsHovering(false);
  };

  // Add a pulse effect to the selected color
  useEffect(() => {
    if (containerRef.current) {
      const selectedEl = containerRef.current.querySelector(
        `[data-color="${selectedColor}"]`
      ) as HTMLDivElement;
      
      if (selectedEl) {
        const pulseInterval = setInterval(() => {
          selectedEl.style.boxShadow = "0 0 12px var(--neon-cyan)";
          
          setTimeout(() => {
            selectedEl.style.boxShadow = "0 0 6px var(--neon-cyan)";
          }, 500);
        }, 2000);
        
        return () => clearInterval(pulseInterval);
      }
    }
  }, [selectedColor]);

  return (
    <div
      ref={containerRef}
      className="flex flex-row gap-2 py-3 flex-wrap glass-panel p-4"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      {colors.map((color) => {
        const isSelected = color === selectedColor;
        const saturation = !isHovering && !isSelected ? "saturate-[0.5] opacity-40" : "";
        const borderColor = isSelected
          ? `border border-${color}-400`
          : "border-gray-800/40";
          
        return (
          <div
            key={color}
            data-color={color}
            className={`
              ${saturation} rounded-md p-1 border-2 ${borderColor}
              cursor-pointer hover:opacity-100 transition-all duration-300
              hover:scale-110 hover:shadow-lg hover:shadow-${color}-500/30
              ${isSelected ? "scale-110 shadow-lg shadow-${color}-500/30" : ""}
            `}
            onClick={() => {
              onSelect(color);
            }}
          >
            <div 
              className={`w-6 h-6 bg-${color}-500 rounded-sm`}
              style={{
                boxShadow: isSelected ? "0 0 8px rgba(0, 255, 255, 0.5)" : "none",
              }}
            ></div>
          </div>
        );
      })}
    </div>
  );
};