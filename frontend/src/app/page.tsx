"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";

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
import { Scale, Send, User } from "lucide-react";

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

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);
    setLoadingMessageIndex(0);

    try {
      console.log("📡 Sending POST to Django backend...");
      const res = await fetch("http://127.0.0.1:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
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

  const loadingMessages = [
    "Researching jurisprudence",
    "Querying legal database",
    "Analyzing case law",
    "Generating case digest",
    "Compiling legal principles",
  ];

  return (
    <div className="flex flex-col items-center justify-center h-screen p-4 bg-white">
      <Card className="w-full max-w-3xl flex flex-col h-full bg-white border rounded-xl shadow-sm">
        <CardHeader className="text-black">
          <CardTitle className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full flex items-center justify-center">
              <Scale className="w-5 h-5" />
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-bold">LegalBot PH</span>
              <span className="text-sm font-normal">
                Philippine Jurisprudence Assistant
              </span>
            </div>
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
  );
}
