import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatMessageInput } from "@/components/chat/ChatMessageInput";
import { ChatMessage as ComponentsChatMessage } from "@livekit/components-react";
import { useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Virtuoso } from 'react-virtuoso';

const inputHeight = 48;

export type ChatMessageType = {
  name: string;
  message: string;
  isSelf: boolean;
  timestamp: number;
};

type ChatTileProps = {
  messages: ChatMessageType[];
  accentColor: string;
  onSend?: (message: string) => Promise<ComponentsChatMessage>;
};

export const ChatTile = ({ messages, accentColor, onSend }: ChatTileProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Memoize message rendering function
  const renderMessage = useMemo(() => (message: ChatMessageType, index: number) => {
    const hideName = index >= 1 && messages[index - 1].name === message.name;
    
    return (
      <motion.div
        key={`${message.name}-${message.timestamp}-${index}`}
        initial={{ opacity: 0, y: 10, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, scale: 0.98 }}
        transition={{ 
          duration: 0.2,
          type: "spring",
          stiffness: 400,
          damping: 25
        }}
      >
        <ChatMessage
          hideName={hideName}
          name={message.name}
          message={message.message}
          isSelf={message.isSelf}
          accentColor={accentColor}
        />
      </motion.div>
    );
  }, [messages, accentColor]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [containerRef, messages]);

  return (
    <div className="flex flex-col gap-2 w-full h-full relative">
      {/* Holographic grid overlay */}
      <div className="absolute inset-0 pointer-events-none holo-grid opacity-10"></div>
      
      <div
        ref={containerRef}
        className="overflow-y-auto scrollbar-thin scrollbar-track-gray-900/20 scrollbar-thumb-cyan-500/20"
        style={{
          height: `calc(100% - ${inputHeight}px)`,
        }}
      >
        <Virtuoso
          style={{ height: '100%' }}
          data={messages}
          itemContent={(index, message) => renderMessage(message, index)}
          followOutput="smooth"
          alignToBottom
        />
      </div>
      
      <ChatMessageInput
        height={inputHeight}
        placeholder="Type a message"
        accentColor={accentColor}
        onSend={onSend ? (message) => onSend(message) : undefined}
      />
    </div>
  );
};