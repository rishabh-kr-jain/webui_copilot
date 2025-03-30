import React from "react";
import Widget from "./Widget";
import ChatComponent from "./ChatComponent";
import "./Dashboard.css";

/**
 * This component:
 * 1. Renders three radio inputs (hidden) to track which panel is selected (GDP, CO2, Agri).
 * 2. Has a left sidebar with icons for each dataset and a History section.
 * 3. Shows exactly one "panel" at a time (the selected dataset) above the chat box.
 */
function Dashboard() {
  return (
    <div className="dashboard-container">
      {/* Hidden radio buttons to toggle panels */}
      <input type="radio" name="dataset" id="radio-gdp" className="visually-hidden" />
      <input type="radio" name="dataset" id="radio-co2" className="visually-hidden" />
      <input type="radio" name="dataset" id="radio-agri" className="visually-hidden" />

      {/* LEFT SIDEBAR */}
      <div className="sidebar">
        <div className="icon-section">
          <label htmlFor="radio-gdp" className="icon-label">
            <i className="fa-solid fa-chart-line"></i>
            <span>GDP</span>
          </label>

          <label htmlFor="radio-co2" className="icon-label">
            <i className="fa-solid fa-cloud"></i>
            <span>CO₂</span>
          </label>

          <label htmlFor="radio-agri" className="icon-label">
            <i className="fa-solid fa-seedling"></i>
            <span>Agriculture</span>
          </label>
        </div>

        <div className="history-section">
          <h3>History</h3>
          {/* You can list previous chats or other items here */}
        </div>
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="main-content">
        <h2 className="main-header">Web Copilot</h2>

        {/* RESULTS AREA - holds the widget panels */}
        <div className="results-container">
          {/* GDP Panel */}
          <div className="panel panel-gdp">
            <Widget
              title="GDP (USA, 100 yrs)"
              apiUrl="http://127.0.0.1:8000/api/gdp-usa-100yrs"
              xKey="year"
              yKey="gdp"
              yLabel="GDP (Trillions USD)"
            />
          </div>

          {/* CO2 Panel */}
          <div className="panel panel-co2">
            <Widget
              title="CO₂ Emissions (World, 50 yrs)"
              apiUrl="http://127.0.0.1:8000/api/co2-world-50yrs"
              xKey="year"
              yKey="co2"
              yLabel="CO₂ Emissions (kT)"
            />
          </div>

          {/* Agricultural Panel */}
          <div className="panel panel-agri">
            <Widget
              title="Agricultural Land (World, 50 yrs)"
              apiUrl="http://127.0.0.1:8000/api/agri-land-world-50yrs"
              xKey="year"
              yKey="agriLand"
              yLabel="Area (hectares or sq km)"
            />
          </div>
        </div>

        {/* CHAT BOX AT THE BOTTOM */}
        <div className="chat-box">
          <ChatComponent />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
