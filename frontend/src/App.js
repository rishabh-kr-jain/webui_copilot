import React from "react";
import AgentNavBar from "./components/AgentNavBar";
import Dashboard from "./components/Dashboard";
import ChatComponent from "./components/ChatComponent";
import "./App.css";

function App() {
  return (
    <div className="app-container">
      <AgentNavBar />
      <div className="content-container">
    
        <Dashboard />
      </div>
    </div>
  );
}

export default App;
