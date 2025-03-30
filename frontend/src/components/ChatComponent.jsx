import React, { useState } from "react";
import "./ChatComponent.css";

function ChatComponent() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),  
      });
      const data = await res.json();
      setResponse(data.answer);  
    } catch (error) {
      console.error("Error fetching response:", error);
      setResponse("Error fetching response. Please try again.");
    }
  };

  return (
    <div className="chat-component">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a clinical or UN food security question or anything else as well"
        />
        <button type="submit">Submit</button>
      </form>
      {response && (
        <div className="chat-response">
          <h4>Response:</h4>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}

export default ChatComponent;
