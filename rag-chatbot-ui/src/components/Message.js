import React from "react";
import ReactMarkdown from "react-markdown";
function Message({ sender, text }) {

  return (
    <div className={sender === "user" ? "user-message" : "bot-message"}>
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );

}

export default Message;