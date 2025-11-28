"use client";

import type React from "react";
import { useEffect, useRef, useState } from "react";

// import RatingComponent from "@/components/RatingComponent";
import RichText from "@/components/RichText";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { RotateCcw, Scale, Send, User } from "lucide-react";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content:
        "Hello! I'm LegalBot PH, your Philippine Jurisprudence Assistant. What case would you like me to digest for you?",
      role: "assistant",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);
  // const [ratedMessages, setRatedMessages] = useState<Set<string>>(new Set());

  // 👇 ref for the dummy "end of messages" div
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // 👇 auto-scroll every time messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // 👇 cycling loading messages after initial delay
  useEffect(() => {
    if (!isTyping) return;

    const messages = [
      "Thinking...",
      "Researching jurisprudence",
      "Querying legal database",
      "Analyzing case law",
      "Generating case digest",
      "Compiling legal principles",
    ];

    // First message appears after 5 seconds, then changes every 10 seconds
    const firstMessageTimer = setTimeout(() => {
      setLoadingMessageIndex(0);
    }, 5000);

    const interval = setInterval(() => {
      setLoadingMessageIndex((prev) => (prev + 1) % messages.length);
    }, 10000);

    return () => {
      clearTimeout(firstMessageTimer);
      clearInterval(interval);
    };
  }, [isTyping]);

  // 👇 show loading message after 5 seconds
  const [showLoadingMessage, setShowLoadingMessage] = useState(false);

  useEffect(() => {
    if (!isTyping) {
      setShowLoadingMessage(false);
      return;
    }

    const timer = setTimeout(() => {
      setShowLoadingMessage(true);
    }, 5000);

    return () => clearTimeout(timer);
  }, [isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;
    sendQuery(input);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
  };

  // const handleRatingSubmitted = (messageId: string) => {
  //   setRatedMessages(prev => new Set([...prev, messageId]));
  // };

  const handleClearChat = async () => {
    // Reset to initial welcome message
    setMessages([
      {
        id: "welcome",
        content:
          "Hello! I'm LegalBot PH, your Philippine Jurisprudence Assistant. What case would you like me to digest for you?",
        role: "assistant",
        timestamp: new Date(),
      },
    ]);
    setInput("");
    // setRatedMessages(new Set());
    console.log("🔄 Chat history cleared - starting new session");
    
    // Clear case content cache on the backend for new session
    try {
      const response = await fetch("/api/admin/clear-cache/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ type: "session" }),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log("✅ Session cache cleared:", data.cleared_caches);
      } else {
        console.warn("⚠️ Failed to clear session cache:", response.statusText);
      }
    } catch (error) {
      console.warn("⚠️ Error clearing session cache:", error);
    }
  };

  const handleCaseNumberClick = (caseNumber: string, caseType: 'gr' | 'am') => {
    // Format the case number for the query
    let query = '';
    if (caseType === 'gr') {
      query = `G.R. No. ${caseNumber}`;
    } else if (caseType === 'am') {
      query = `A.M. No. ${caseNumber}`;
    }
    
    // Send the query directly
    sendQuery(query);
  };

  const sendQuery = async (queryText: string) => {
    if (!queryText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: queryText,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);
    setLoadingMessageIndex(0);

    try {
      // Build conversation history for context (exclude welcome message)
      const history = messages
        .filter(msg => msg.id !== "welcome") // Exclude welcome message
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));

      console.log("📡 Sending POST to Django backend with history...");
      console.log(`📜 History length: ${history.length} messages`);
      
      const res = await fetch("http://127.0.0.1:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: queryText,
          history: history 
        }),
      });

      const data = await res.json();
      console.log("✅ Response from backend:", data);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response ?? data.error ?? "No response",
        role: "assistant",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("❌ Error talking to backend:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 2).toString(),
          content: "Error connecting to backend",
          role: "assistant",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const extractCaseId = (content: string): string | undefined => {
    // Try to extract GR number or case ID from the response
    const grMatch = content.match(/G\.R\.?\s*NOS?\.?\s*([0-9\-]+)/i);
    if (grMatch) return grMatch[0];
    
    const spMatch = content.match(/SP\s*No\.?\s*([0-9\-]+)/i);
    if (spMatch) return spMatch[0];
    
    return undefined;
  };

  const loadingMessages = [
    "Thinking...",
    "Researching jurisprudence",
    "Querying legal database",
    "Analyzing case law",
    "Generating case digest",
    "Compiling legal principles",
  ];

  return (
    <div className="flex h-screen bg-white">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col items-center justify-center p-4">
        <Card className="w-full max-w-3xl flex flex-col h-full bg-white border rounded-xl shadow-sm">
        <CardHeader className="text-black">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full flex items-center justify-center">
                <Scale className="w-5 h-5" />
              </div>
              <div className="flex flex-col">
                <span className="text-xl font-bold">LegalBot PH</span>
                <span className="text-sm font-normal">
                  Philippine Jurisprudence Assistant
                </span>
              </div>
            </div>
            {messages.length > 1 && (
              <Button
                onClick={handleClearChat}
                variant="outline"
                size="sm"
                className="flex items-center gap-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100"
              >
                <RotateCcw className="w-4 h-4" />
                <span className="hidden sm:inline">New Chat</span>
              </Button>
            )}
          </CardTitle>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto px-4 py-2">
          <div className="space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.role === "assistant" && (
                  <Avatar className="w-10 h-10 border-2">
                    <AvatarFallback>
                      <Scale className="w-5 h-5" />
                    </AvatarFallback>
                  </Avatar>
                )}

                <div
                  className={`max-w-[75%] rounded-xl px-4 py-3 ${
                    message.role === "user"
                      ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white"
                      : "bg-white border text-slate-800"
                  }`}
                >
                  {message.role === "assistant" ? (
                    <RichText
                      content={message.content}
                      className="text-sm leading-relaxed space-y-2"
                      onCaseNumberClick={handleCaseNumberClick}
                    />
                  ) : (
                    <p className="text-sm leading-relaxed">{message.content}</p>
                  )}
                  <p
                    className={`text-xs mt-2 ${
                      message.role === "user"
                        ? "text-blue-100"
                        : "text-slate-500"
                    }`}
                  >
                    {formatTime(message.timestamp)}
                  </p>
                </div>

                {message.role === "user" && (
                  <Avatar className="w-10 h-10 border-2 border-blue-300">
                    <AvatarFallback className="bg-gradient-to-r from-blue-600 to-blue-700 text-white">
                      <User className="w-5 h-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-4 justify-start">
                <Avatar className="w-10 h-10 border-2">
                  <AvatarFallback>
                    <Scale className="w-5 h-5" />
                  </AvatarFallback>
                </Avatar>
                <div className="px-4 py-3 shadow-md bg-white border rounded-xl">
                  <div className="flex items-center space-x-2">
                    {showLoadingMessage && (
                      <span className="text-sm text-slate-600">
                        {loadingMessages[loadingMessageIndex]}
                      </span>
                    )}
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 👇 dummy div to scroll into view */}
            <div ref={messagesEndRef} />
          </div>
        </CardContent>

        <CardFooter className="border-t p-4 bg-white sticky bottom-0 z-10">
          <div className="flex w-full gap-3">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about jurisprudence..."
              className="flex-1 border-gray-300 focus:ring-0 focus:border-blue-500 bg-gray-50 text-sm"
              disabled={isTyping}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isTyping}
              size="icon"
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-lg"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </CardFooter>
        </Card>
      </div>

      {/* Rating Sidebar - COMMENTED OUT */}
      {/* {messages.length > 1 && messages[messages.length - 1].role === "assistant" && 
       !ratedMessages.has(messages[messages.length - 1].id) && (
        <div className="w-80 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
          <div className="sticky top-4">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-800 mb-2">Rate Response</h3>
              <p className="text-sm text-gray-600">
                Help us improve by rating the latest response
              </p>
            </div>
            <RatingComponent
              query={messages[messages.length - 2]?.content || ""}
              response={messages[messages.length - 1].content}
              caseId={extractCaseId(messages[messages.length - 1].content)}
              onRatingSubmitted={() => handleRatingSubmitted(messages[messages.length - 1].id)}
            />
          </div>
        </div>
      )} */}
    </div>
  );
}
