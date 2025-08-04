"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Send, Scale, User, BookOpen } from "lucide-react";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Kumusta! I'm LegalBot PH, your Philippine jurisprudence assistant. I can help you with questions about Philippine laws, Supreme Court decisions, legal principles, and case citations. How may I assist you today?",
      role: "assistant",
      timestamp: new Date(Date.now() - 120000),
    },
    {
      id: "2",
      content:
        "Hi! Can you explain the doctrine of piercing the corporate veil under Philippine law?",
      role: "user",
      timestamp: new Date(Date.now() - 90000),
    },
    {
      id: "3",
      content:
        "The doctrine of piercing the corporate veil allows courts to disregard the separate juridical personality of a corporation. Under Philippine jurisprudence, this applies when the corporate fiction is used to defeat public convenience, justify wrong, protect fraud, or defend crime. Key cases include Concept Builders, Inc. v. NLRC (1994) and Lim Tong Lim v. Philippine Fishing Gear Industries (1992).",
      role: "assistant",
      timestamp: new Date(Date.now() - 60000),
    },
    {
      id: "4",
      content:
        "That's helpful! Can you also cite the specific G.R. numbers for those cases?",
      role: "user",
      timestamp: new Date(Date.now() - 30000),
    },
    {
      id: "5",
      content:
        "Concept Builders, Inc. v. NLRC is G.R. No. 108734, decided on May 26, 1994. Lim Tong Lim v. Philippine Fishing Gear Industries is G.R. No. 136448, decided on November 3, 1999. These cases established important precedents for when courts may pierce the corporate veil in the Philippines.",
      role: "assistant",
      timestamp: new Date(Date.now() - 15000),
    },
  ]);

  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    // Simulate AI response after a delay
    setTimeout(() => {
      const responses = [
        "That's an excellent legal question. Under Philippine jurisprudence, this principle is well-established through various Supreme Court decisions...",
        "Based on Philippine law and relevant case law, here's what you need to know about this legal concept...",
        "The Supreme Court of the Philippines has consistently held in several landmark cases that...",
        "According to the Civil Code of the Philippines and supporting jurisprudence, this legal principle operates as follows...",
        "This is a fundamental concept in Philippine law. Let me cite the relevant provisions and case law for you...",
        "The doctrine you're asking about has been refined through years of Philippine Supreme Court decisions. Here's the current state of the law...",
        "Under the Revised Penal Code and related jurisprudence, this legal principle is applied in the following manner...",
        "The Constitutional basis for this principle, as interpreted by the Philippine Supreme Court, establishes that...",
      ];

      const randomResponse =
        responses[Math.floor(Math.random() * responses.length)];

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: randomResponse,
        role: "assistant",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 2000);
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

  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-200 p-4">
      <Card className="w-full max-w-5xl  flex flex-col bg-slate-200 shadow-none">
        <CardHeader className="text-black ">
          <CardTitle className="flex items-center gap-3">
            <div className="w-10 h-10  rounded-full flex items-center justify-center ">
              <Scale className="w-5 h-5 " />
            </div>
            <div className="flex flex-col">
              <span className="text-xl font-bold">LegalBot PH</span>
              <span className="text-sm font-normal">
                Philippine Jurisprudence Assistant
              </span>
            </div>
          </CardTitle>
        </CardHeader>

        <CardContent className="flex-1 p-0 bg-slate-200">
          <ScrollArea className="h-full p-6">
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
                      <AvatarFallback className="">
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
                    <p className="text-sm leading-relaxed">{message.content}</p>
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
                    <AvatarFallback className="">
                      <Scale className="w-5 h-5" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="bg-white border rounded-xl px-4 py-3 shadow-md">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-slate-600">
                        Researching jurisprudence
                      </span>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        ></div>
                        <div
                          className="w-2 h-2 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>

        <CardFooter className="border-t border-amber-200/20 bg-gradient-to-r from-slate-50 to-blue-50 p-4">
          <div className="flex w-full gap-3">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Supreme Court cases, legal doctrines..."
              className="flex-1 focus:ring-amber-400/20"
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
