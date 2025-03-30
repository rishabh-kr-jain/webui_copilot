import React, { useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function Widget({ title, apiUrl, xKey, yKey, yLabel }) {
  const [chartData, setChartData] = useState(null);

  // Same data-fetching logic, but now we call it automatically on mount
  const handleClick = async () => {
    try {
      const response = await fetch(apiUrl);
      const jsonData = await response.json();
      const dataArray = jsonData.data || [];

      const labels = dataArray.map((item) => item[xKey]);
      const values = dataArray.map((item) => item[yKey]);

      const data = {
        labels,
        datasets: [
          {
            label: yLabel,
            data: values,
            borderColor: "rgba(75,192,192,1)",
            fill: false,
          },
        ],
      };

      setChartData(data);
    } catch (error) {
      console.error("Error loading chart data:", error);
    }
  };

  // Automatically fetch chart data as soon as the component mounts
  useEffect(() => {
    handleClick();
    // eslint-disable-next-line
  }, []);

  return (
    <div className="widget-card">
      <h3>{title}</h3>
      {/* Remove the "Load Chart" button. The chart loads automatically. */}
      {chartData && (
        <Line
          data={chartData}
          options={{
            responsive: true,
            plugins: {
              legend: { position: "top" },
              title: { display: true, text: title },
            },
            scales: {
              x: { title: { display: true, text: xKey } },
              y: { title: { display: true, text: yLabel } },
            },
          }}
        />
      )}
    </div>
  );
}

export default Widget;
