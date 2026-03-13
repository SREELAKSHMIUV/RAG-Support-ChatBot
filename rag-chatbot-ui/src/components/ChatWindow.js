import React from "react";
import Message from "./Message";

function ChatWindow({ messages }) {

  return (
    <div className="chat-window">

      {messages.map((msg, index) => (
        <Message key={index} sender={msg.sender} text={msg.text} />
      ))}

    </div>
  );

}

export default ChatWindow;