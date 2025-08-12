"use client";

import { useState } from "react";

export default function ChatBox() {
  const [query, setQuery] = useState("");

  async function send() {
    console.log("📡 Sending POST to Django backend...");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      console.log("✅ Response from backend:", data);
    } catch (error) {
      console.error("❌ Error talking to backend:", error);
    }
  }

  return (
    <div style={{ padding: "1rem" }}>
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Type your question..."
        rows={4}
        style={{ width: "100%", marginBottom: "0.5rem" }}
      />
      <button onClick={send}>Send</button>
    </div>
  );
}
