import React from "react";
import "./AgentNavBar.css";

function AgentNavBar() {
  return (
    <nav className="agent-nav">
      <div className="logo">Web UI Copilot</div>
      <ul>
        <li>General WebUI</li>
        <li>Clinical RAG</li>
        <li>Food Security</li>
      </ul>
    </nav>
  );
}

export default AgentNavBar;
