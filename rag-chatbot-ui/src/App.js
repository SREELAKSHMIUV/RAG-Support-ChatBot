import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import "./App.css";

function App() {

  const [messages, setMessages] = useState([]);

  const sendMessage = async (text) => {

    const userMessage = {
      sender: "user",
      text: text
    };

    setMessages(prev => [...prev, userMessage]);

    try {

      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: text })
      });

      const data = await response.json();

      const botMessage = {
        sender: "bot",
        text: data.answer
      };

      setMessages(prev => [...prev, botMessage]);

    } catch {

      setMessages(prev => [...prev, {
        sender: "bot",
        text: "Error connecting to server."
      }]);

    }

  };

  return (
    <div className="app">

      <div className="chat-container">

        <div className="chat-header">
          AI Support Assistant
        </div>

        <ChatWindow messages={messages} />

        <ChatInput sendMessage={sendMessage} />

      </div>

    </div>
  );
}

export default App;