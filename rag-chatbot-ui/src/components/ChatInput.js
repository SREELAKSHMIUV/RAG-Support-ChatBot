import React, { useState } from "react";

function ChatInput({ sendMessage }) {

  const [input, setInput] = useState("");

  const handleSend = () => {

    if (!input.trim()) return;

    sendMessage(input);
    setInput("");

  };

  return (
    <div className="chat-input">

      <input
        type="text"
        placeholder="Ask your question..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
      />

      <button onClick={handleSend}>
        Send
      </button>

    </div>
  );

}

export default ChatInput;