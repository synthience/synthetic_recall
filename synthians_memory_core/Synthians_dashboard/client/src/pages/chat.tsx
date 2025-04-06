import React, { useState, useRef, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar } from "@/components/ui/avatar";
import { Skeleton } from "@/components/ui/skeleton";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  metrics?: {
    variant?: string;
    surprise_level?: number;
    retrieved_memory_ids?: string[];
  };
  isLoading?: boolean;
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollArea = scrollAreaRef.current;
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }
  }, [messages]);
  
  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    
    // Add user message
    const userMessage: ChatMessage = {
      role: "user",
      content: inputValue,
      timestamp: new Date(),
    };
    
    setMessages([...messages, userMessage]);
    
    // Add loading state for assistant message
    const loadingMessage: ChatMessage = {
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isLoading: true
    };
    
    setMessages(prev => [...prev, loadingMessage]);
    
    // Clear input and set typing indicator
    setInputValue("");
    setIsTyping(true);
    
    // This is a placeholder for backend integration
    // In a real implementation, this would call the CCE service
    console.log("Sending message:", inputValue);
    
    // Simulate response after a delay
    setTimeout(() => {
      setIsTyping(false);
      
      // Replace loading message with placeholder response
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages.pop(); // Remove loading message
        
        // Add simulated response
        newMessages.push({
          role: "assistant",
          content: "Chat interface connected. Waiting for input... (Backend integration required)",
          timestamp: new Date(),
          metrics: {
            variant: "MAC",
            surprise_level: 0.42,
            retrieved_memory_ids: ["MEM-123456", "MEM-789012"]
          }
        });
        
        return newMessages;
      });
    }, 1500);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  const handleClearChat = () => {
    setMessages([]);
  };
  
  return (
    <>
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xl font-semibold text-white mb-1">Chat Interface</h2>
          <p className="text-sm text-gray-400">
            Direct interaction interface with an AI persona powered by the Synthians memory system
          </p>
        </div>
        
        <Button variant="outline" size="sm" onClick={handleClearChat}>
          <i className="fas fa-trash mr-2"></i>
          Clear Chat
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="lg:col-span-3">
          <CardHeader className="pb-2">
            <div className="flex items-center">
              <Avatar className="mr-2 h-8 w-8 bg-primary">
                <span className="text-xs">AI</span>
              </Avatar>
              <div>
                <CardTitle>Synthians Chat</CardTitle>
                <CardDescription>
                  Phase 6 Preparation - Placeholder for end-to-end testing
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {/* Chat Messages */}
            <ScrollArea className="h-[calc(100vh-320px)] p-4" ref={scrollAreaRef}>
              {messages.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <i className="fas fa-comments text-4xl mb-4 text-muted-foreground"></i>
                  <p>Send a message to start the conversation</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {messages.map((message, index) => (
                    <div 
                      key={index} 
                      className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    >
                      <div 
                        className={`max-w-[80%] rounded-lg p-3 ${
                          message.role === "user" 
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted"
                        }`}
                      >
                        {message.isLoading ? (
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 rounded-full bg-gray-400 animate-ping"></div>
                            <div className="w-2 h-2 rounded-full bg-gray-400 animate-ping" style={{ animationDelay: "0.2s" }}></div>
                            <div className="w-2 h-2 rounded-full bg-gray-400 animate-ping" style={{ animationDelay: "0.4s" }}></div>
                          </div>
                        ) : (
                          <>
                            <p className="mb-1">{message.content}</p>
                            <div className="text-xs opacity-70 mt-1 flex justify-between">
                              <span>{message.timestamp.toLocaleTimeString()}</span>
                              
                              {message.metrics && (
                                <span className="ml-2">
                                  {message.metrics.variant && (
                                    <Badge variant="outline" className="text-secondary border-secondary mr-1">
                                      {message.metrics.variant}
                                    </Badge>
                                  )}
                                </span>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
            
            {/* Input Area */}
            <div className="p-4 border-t border-border">
              <div className="flex space-x-2">
                <Input
                  placeholder="Type a message..."
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
                <Button onClick={handleSendMessage} disabled={!inputValue.trim()}>
                  <i className="fas fa-paper-plane mr-2"></i>
                  Send
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
        
        {/* Metrics Panel */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Interaction Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-500 mb-1">Active Variant</p>
                <Badge className="bg-muted text-secondary">MAC</Badge>
              </div>
              
              <div>
                <p className="text-sm text-gray-500 mb-1">Recent Surprise Level</p>
                <div className="h-2 w-full bg-muted rounded-full">
                  <div 
                    className="h-2 bg-gradient-to-r from-blue-500 to-primary rounded-full"
                    style={{ width: "42%" }}
                  ></div>
                </div>
                <p className="text-xs mt-1 text-right">0.42</p>
              </div>
              
              <div>
                <p className="text-sm text-gray-500 mb-1">Retrieved Memories</p>
                <div className="bg-muted p-2 rounded text-xs font-mono max-h-32 overflow-y-auto">
                  <div className="flex items-center mb-1">
                    <i className="fas fa-memory text-secondary mr-1"></i>
                    <span>MEM-123456</span>
                  </div>
                  <div className="flex items-center">
                    <i className="fas fa-memory text-secondary mr-1"></i>
                    <span>MEM-789012</span>
                  </div>
                </div>
              </div>
              
              <div className="pt-4 border-t border-border mt-4">
                <p className="text-sm text-gray-400 italic">
                  This is a placeholder interface. Backend integration with CCE is required for this feature to work.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
