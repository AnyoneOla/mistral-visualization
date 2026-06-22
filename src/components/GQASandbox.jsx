import React, { useState } from "react";
import { Info, Network, Cpu, Database } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function GQASandbox({ simData }) {
  const [selectedType, setSelectedType] = useState("GQA"); // MHA, GQA, MQA
  const [selectedQHead, setSelectedQHead] = useState(0);
  const [projTab, setProjTab] = useState("Q"); // Q, K, V
  const [hoveredKVHead, setHoveredKVHead] = useState(null);

  const numQHeads = 8;
  let numKVHeads = 2;
  if (selectedType === "MHA") numKVHeads = 8;
  if (selectedType === "MQA") numKVHeads = 1;
  if (selectedType === "GQA") numKVHeads = 2;

  const ratio = numQHeads / numKVHeads;
  const selectedKVHead = Math.floor(selectedQHead / ratio);

  // Read matrices and vectors from simData
  const activeEmbedding = simData.rmsnorm1.output; // [1 x 8]
  const W_Q = simData.gqa.W_Q[selectedQHead % simData.gqa.W_Q.length]; // [8 x 4]
  const W_K = simData.gqa.W_K[selectedKVHead % simData.gqa.W_K.length]; // [8 x 4]
  const W_V = simData.gqa.W_V[selectedKVHead % simData.gqa.W_V.length]; // [8 x 4]
  const qVec = simData.gqa.qVectors[selectedQHead % simData.gqa.qVectors.length]; // [1 x 4]

  const activeK = simData.gqa.kVectors[simData.activeTokenIdx];
  const kVec = activeK && activeK[selectedKVHead % activeK.length] 
    ? activeK[selectedKVHead % activeK.length] 
    : [0.1, 0.2, -0.3, 0.4];

  const activeV = simData.gqa.vVectors[simData.activeTokenIdx];
  const vVec = activeV && activeV[selectedKVHead % activeV.length]
    ? activeV[selectedKVHead % activeV.length]
    : [0.5, -0.1, 0.2, 0.3];

  // SVG dimensions
  const svgWidth = 220;
  const svgHeight = 320;
  const paddingY = 20;

  const getQHeadCoords = (idx) => {
    const step = (svgHeight - 2 * paddingY) / (numQHeads - 1 || 1);
    return { x: 20, y: paddingY + idx * step };
  };

  const getKVHeadCoords = (idx) => {
    const step = (svgHeight - 2 * paddingY) / (numKVHeads - 1 || 1);
    if (numKVHeads === 1) {
      return { x: svgWidth - 20, y: svgHeight / 2 };
    }
    return { x: svgWidth - 20, y: paddingY + idx * step };
  };

  const activeW = projTab === "Q" ? W_Q : projTab === "K" ? W_K : W_V;
  const activeOut = projTab === "Q" ? qVec : projTab === "K" ? kVec : vVec;
  const matrixLabel = projTab === "Q" ? `W_Q (Head ${selectedQHead})` : projTab === "K" ? `W_K (Group ${selectedKVHead})` : `W_V (Group ${selectedKVHead})`;
  const outputLabel = projTab === "Q" ? `Query Vector q` : projTab === "K" ? `Key Vector k` : `Value Vector v`;
  
  const getAccentColor = () => {
    if (projTab === "Q") return "var(--query-color)";
    if (projTab === "K") return "var(--key-color)";
    return "var(--value-color)";
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--key-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Network className="w-6 h-6 text-amber-400 animate-pulse-slow" />
          Step 3: QKV projections & Grouped-Query Attention (GQA)
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Attention maps inputs to three representations: **Queries (Q)**, **Keys (K)**, and **Values (V)** using projection matrices.
          Standard **Multi-Head Attention (MHA)** uses separate Key-Value heads for every Query head, taking massive memory. 
          Mistral uses **Grouped-Query Attention (GQA)**, grouping Query heads to share Key-Value heads. This yields <strong>8x memory savings</strong> during token generation.
        </p>
      </div>

      {/* Main Interactive Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1.1fr 1.3fr", gap: "24px", alignItems: "start" }}>
        
        {/* SVG Grouping Visualizer */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "white" }}>Head Map Visualizer</h3>
              
              {/* Attention Mode Selectors */}
              <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-surface)", padding: "3px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
                {["MHA", "GQA", "MQA"].map((type) => (
                  <button
                    key={type}
                    onClick={() => {
                      setSelectedType(type);
                      setSelectedQHead(0);
                    }}
                    style={{
                      padding: "4px 10px",
                      borderRadius: "6px",
                      border: "none",
                      backgroundColor: selectedType === type ? "var(--accent-color)" : "transparent",
                      color: selectedType === type ? "white" : "var(--text-secondary)",
                      fontSize: "0.75rem",
                      fontWeight: "700",
                      cursor: "pointer",
                      transition: "all 0.15s ease"
                    }}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Network SVG Graph Container */}
          <div style={{ display: "flex", justifyContent: "space-around", alignItems: "center", minHeight: "330px", padding: "10px", backgroundColor: "rgba(0,0,0,0.15)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
            
            {/* Left Q-head select list */}
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", height: `${svgHeight}px`, padding: `${paddingY}px 0` }}>
              {Array.from({ length: numQHeads }).map((_, idx) => {
                const isActive = selectedQHead === idx;
                const isGroupHovered = hoveredKVHead !== null && Math.floor(idx / ratio) === hoveredKVHead;
                return (
                  <div 
                    key={`gqa-q-lbl-${idx}`} 
                    onClick={() => setSelectedQHead(idx)}
                    style={{ 
                      fontSize: "0.75rem", 
                      color: isActive ? "var(--query-color)" : isGroupHovered ? "white" : "var(--text-muted)",
                      fontWeight: isActive || isGroupHovered ? "700" : "400",
                      cursor: "pointer",
                      textAlign: "right",
                      paddingRight: "8px",
                      transition: "all 0.15s ease"
                    }}
                  >
                    Query Head Q<sub>{idx}</sub>
                  </div>
                );
              })}
            </div>

            {/* SVG area */}
            <svg width={svgWidth} height={svgHeight} style={{ overflow: "visible" }}>
              {/* Links */}
              {Array.from({ length: numQHeads }).map((_, qIdx) => {
                const targetKVIdx = Math.floor(qIdx / ratio);
                const qCoords = getQHeadCoords(qIdx);
                const kvCoords = getKVHeadCoords(targetKVIdx);
                
                const isSelected = selectedQHead === qIdx;
                const isGroupHovered = hoveredKVHead === targetKVIdx;
                
                const cp1x = qCoords.x + 60;
                const cp1y = qCoords.y;
                const cp2x = kvCoords.x - 60;
                const cp2y = kvCoords.y;

                const pathData = `M ${qCoords.x} ${qCoords.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${kvCoords.x} ${kvCoords.y}`;

                let strokeColor = "rgba(255, 255, 255, 0.05)";
                if (isSelected) strokeColor = "var(--query-color)";
                else if (isGroupHovered) strokeColor = "var(--key-color)";

                return (
                  <g key={`gqa-p-${qIdx}`} style={{ transition: "all 0.2s ease" }}>
                    <path
                      d={pathData}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={isSelected ? 3.5 : isGroupHovered ? 2 : 1}
                      className="gqa-line"
                    />
                    {isSelected && (
                      <circle r="4" fill="var(--query-color)">
                        <animateMotion dur="1.2s" repeatCount="indefinite" path={pathData} />
                      </circle>
                    )}
                  </g>
                );
              })}

              {/* Q Nodes */}
              {Array.from({ length: numQHeads }).map((_, idx) => {
                const coords = getQHeadCoords(idx);
                const isSelected = selectedQHead === idx;
                return (
                  <circle
                    key={`gqa-q-n-${idx}`}
                    cx={coords.x}
                    cy={coords.y}
                    r={isSelected ? 7 : 5}
                    fill={isSelected ? "var(--query-color)" : "hsl(190, 40%, 15%)"}
                    stroke="var(--bg-surface)"
                    strokeWidth="1.5"
                    onClick={() => setSelectedQHead(idx)}
                    style={{ cursor: "pointer", transition: "all 0.2s ease", filter: isSelected ? "drop-shadow(0 0 6px var(--query-color))" : "none" }}
                  />
                );
              })}

              {/* KV Nodes */}
              {Array.from({ length: numKVHeads }).map((_, idx) => {
                const coords = getKVHeadCoords(idx);
                const isMappedToSelected = Math.floor(selectedQHead / ratio) === idx;
                return (
                  <circle
                    key={`gqa-kv-n-${idx}`}
                    cx={coords.x}
                    cy={coords.y}
                    r={isMappedToSelected ? 8 : 6}
                    fill={isMappedToSelected ? "var(--key-color)" : "hsl(38, 40%, 15%)"}
                    stroke="var(--bg-surface)"
                    strokeWidth="1.5"
                    onMouseEnter={() => setHoveredKVHead(idx)}
                    onMouseLeave={() => setHoveredKVHead(null)}
                    style={{ cursor: "help", transition: "all 0.2s ease", filter: isMappedToSelected ? "drop-shadow(0 0 8px var(--key-color))" : "none" }}
                  />
                );
              })}
            </svg>

            {/* Right KV-head group labels */}
            <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", height: `${svgHeight}px`, padding: `${paddingY}px 0` }}>
              {Array.from({ length: numKVHeads }).map((_, idx) => {
                const isMapped = Math.floor(selectedQHead / ratio) === idx;
                return (
                  <div 
                    key={`gqa-kv-lbl-${idx}`} 
                    onMouseEnter={() => setHoveredKVHead(idx)}
                    onMouseLeave={() => setHoveredKVHead(null)}
                    style={{ 
                      fontSize: "0.75rem", 
                      color: isMapped ? "var(--key-color)" : "var(--text-muted)",
                      fontWeight: isMapped ? "700" : "400",
                      paddingLeft: "8px",
                      cursor: "help"
                    }}
                  >
                    KV Group KV<sub>{idx}</sub>
                  </div>
                );
              })}
            </div>

          </div>

          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontStyle: "italic", textAlign: "center" }}>
            💡 Click a Query head to select it, or hover over a KV head to see groupings.
          </div>

        </div>

        {/* Right Panel: Projection multipliers */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            
            {/* Header select */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.85rem", fontWeight: "700", color: "white" }}>Linear Projection Matrix Multiply</span>
              
              <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-surface)", padding: "2px", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
                {["Q", "K", "V"].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setProjTab(tab)}
                    style={{
                      padding: "3px 8px",
                      borderRadius: "4px",
                      border: "none",
                      backgroundColor: projTab === tab ? getAccentColor() : "transparent",
                      color: projTab === tab ? "black" : "var(--text-secondary)",
                      fontSize: "0.7rem",
                      fontWeight: "700",
                      cursor: "pointer"
                    }}
                  >
                    {tab} Proj
                  </button>
                ))}
              </div>
            </div>

            {/* Matrix multiplication execution */}
            <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
              <InteractiveMatrixMul
                leftVal={activeEmbedding}
                rightVal={activeW}
                outputVal={activeOut}
                leftLabel="RMSNorm 1 Out Vector"
                rightLabel={matrixLabel}
                outputLabel={outputLabel}
                leftColor="var(--query-color)"
                rightColor="var(--key-color)"
                outputColor={getAccentColor()}
                leftRowNames={[`"x_norm"`]}
                rightColNames={Array.from({ length: activeOut.length }, (_, i) => `d${i}`)}
              />
            </div>

            <div style={{ 
              fontSize: "0.75rem", 
              color: "var(--text-secondary)", 
              lineHeight: "1.4",
              padding: "10px 12px",
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-color)",
              borderRadius: "8px"
            }}>
              🚀 <strong>Head dimension:</strong> The projection outputs a vector of size 4 (representing head size 4). In our simulation, the full hidden dimension (8) maps down to head states. Under GQA, multiple query heads share the same Key/Value projection matrices, cutting cache load.
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}
