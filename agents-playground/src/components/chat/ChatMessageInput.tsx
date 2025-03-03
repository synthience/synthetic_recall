import { useWindowResize } from "@/hooks/useWindowResize";
import { useCallback, useEffect, useRef, useState } from "react";

type ChatMessageInputProps = {
  placeholder: string;
  accentColor: string;
  height: number;
  onSend?: (message: string) => void;
};

export const ChatMessageInput: React.FC<ChatMessageInputProps> = ({
  placeholder,
  accentColor,
  height,
  onSend,
}) => {
  const [message, setMessage] = useState("");
  const [inputTextWidth, setInputTextWidth] = useState(0);
  const [inputWidth, setInputWidth] = useState(0);
  const hiddenInputRef = useRef<HTMLSpanElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const windowSize = useWindowResize();
  const [isTyping, setIsTyping] = useState(false);
  const [inputHasFocus, setInputHasFocus] = useState(false);

  const handleSend = useCallback(() => {
    if (!onSend || message === "") {
      return;
    }

    onSend(message);
    setMessage("");
  }, [onSend, message]);

  useEffect(() => {
    setIsTyping(true);
    const timeout = setTimeout(() => {
      setIsTyping(false);
    }, 500);

    return () => clearTimeout(timeout);
  }, [message]);

  useEffect(() => {
    if (hiddenInputRef.current) {
      setInputTextWidth(hiddenInputRef.current.clientWidth);
    }
  }, [hiddenInputRef, message]);

  useEffect(() => {
    if (inputRef.current) {
      setInputWidth(inputRef.current.clientWidth);
    }
  }, [hiddenInputRef, message, windowSize.width]);

  return (
    <div
      className="flex flex-col gap-2 border-t border-t-gray-800 relative glass-panel mt-2"
      style={{ height }}
    >
      {/* Scan line effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none scan-line"></div>
      
      <div className="flex flex-row pt-3 gap-2 items-center relative">
        {/* Digital cursor */}
        <div
          className={`
            w-2 h-4 
            bg-${inputHasFocus ? accentColor : "gray"}-${inputHasFocus ? "400" : "800"} 
            ${inputHasFocus ? "border-glow" : ""} 
            absolute left-2 
            ${!isTyping && inputHasFocus ? "cursor-animation" : ""}
            transition-all duration-200
          `}
          style={{
            transform: `translateX(${
              message.length > 0
                ? Math.min(inputTextWidth, inputWidth - 20) - 4
                : 0
            }px)`,
          }}
        ></div>
        
        {/* Input field */}
        <input
          ref={inputRef}
          className={`
            w-full text-sm caret-transparent bg-transparent 
            text-${accentColor}-100 p-2 pr-6 rounded-sm 
            focus:outline-none focus:ring-1 focus:ring-${accentColor}-700
            transition-all duration-300
            backdrop-blur-sm
            ${inputHasFocus ? "bg-gray-900/20" : "bg-gray-900/10"}
          `}
          style={{
            paddingLeft: message.length > 0 ? "12px" : "24px",
            caretShape: "block",
          }}
          placeholder={placeholder}
          value={message}
          onChange={(e) => {
            setMessage(e.target.value);
          }}
          onFocus={() => {
            setInputHasFocus(true);
          }}
          onBlur={() => {
            setInputHasFocus(false);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleSend();
            }
          }}
        ></input>
        
        {/* Hidden element to measure text width */}
        <span
          ref={hiddenInputRef}
          className="absolute top-0 left-0 text-sm pl-3 text-amber-500 pointer-events-none opacity-0"
        >
          {message.replaceAll(" ", "\u00a0")}
        </span>
        
        {/* Send button */}
        <button
          disabled={message.length === 0 || !onSend}
          onClick={handleSend}
          className={`
            text-xs uppercase tracking-wider
            text-${accentColor}-400 hover:text-${accentColor}-300
            hover:bg-${accentColor}-900/20 p-2 rounded-md
            opacity-${message.length > 0 ? "100" : "25"}
            pointer-events-${message.length > 0 ? "auto" : "none"}
            transition-all duration-300
            border border-transparent hover:border-${accentColor}-500/30
          `}
        >
          Send
        </button>
      </div>
    </div>
  );
};