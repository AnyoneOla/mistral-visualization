import React, { useState } from "react";
import { Info, Network, Cpu, Database, Sparkles } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function GQASandbox({ simData, attentionType, setAttentionType }) {
  const [selectedQHead, setSelectedQHead] = useState(0);
  const [projTab, setProjTab] = useState("viz"); // "viz" (default diagram) or "Q", "K", "V" (math matrices)
  const [hoveredKVHead, setHoveredKVHead] = useState(null);

  const numQHeads = simData.dimensions.numQHeads;
  const numKVHeads = simData.dimensions.numKVHeads;

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

  const isMathActive = ["Q", "K", "V"].includes(projTab);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--query-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Network className="w-6 h-6 text-sky-400 animate-pulse-slow" />
          Step 3: Grouped-Query Attention Projections
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Attention maps inputs to three representations: <strong>Query (Q)</strong> (what we are looking for), <strong>Key (K)</strong> (what information we contain), and <strong>Value (V)</strong> (the actual content).
          <br />
          In standard <strong>Multi-Head Attention (MHA)</strong>, every Query head has a dedicated Key and Value head. 
          To reduce memory cache overhead, Mistral uses <strong>Grouped-Query Attention (GQA)</strong>, grouping multiple Query heads (e.g., 4) to share a single Key-Value pair.
        </p>
      </div>

      {/* Main Layout Grid */}
      <div className="responsive-grid">
        
        {/* Left Panel: Head Map Visualizer */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--text-primary)" }}>Head Map Visualizer</h3>
            
            {/* Attention Mode Selectors */}
            <div className="tab-container">
              {["MHA", "GQA", "MQA"].map((type) => (
                <button
                  key={type}
                  onClick={() => {
                    setAttentionType(type);
                    setSelectedQHead(0);
                  }}
                  className={`tab-button ${attentionType === type ? "active" : ""}`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Network SVG Graph Container */}
          <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "8px", padding: "10px 0" }}>
            <div style={{ display: "flex", alignItems: "center", position: "relative" }}>
              
              {/* Left query-head labels */}
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", height: `${svgHeight}px`, padding: `${paddingY}px 0`, textAlign: "right", paddingRight: "8px" }}>
                {Array.from({ length: numQHeads }).map((_, idx) => (
                  <div 
                    key={`gqa-q-lbl-${idx}`} 
                    onClick={() => setSelectedQHead(idx)}
                    style={{ 
                      fontSize: "0.75rem", 
                      color: selectedQHead === idx ? "var(--query-color)" : "var(--text-secondary)",
                      fontWeight: selectedQHead === idx ? "700" : "400",
                      cursor: "pointer",
                      padding: "2px 6px",
                      borderRadius: "4px",
                      backgroundColor: selectedQHead === idx ? "var(--query-glow)" : "transparent",
                      transition: "all 0.2s"
                    }}
                  >
                    Query Head Q<sub>{idx}</sub>
                  </div>
                ))}
              </div>

              {/* Central connecting mapping lines */}
              <svg width={svgWidth} height={svgHeight} style={{ overflow: "visible" }}>
                <defs>
                  <linearGradient id="grad-q-kv" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="var(--key-color)" stopOpacity="0.8" />
                  </linearGradient>
                </defs>

                {/* Draw all lines first */}
                {Array.from({ length: numQHeads }).map((_, qIdx) => {
                  const kvIdx = Math.floor(qIdx / ratio);
                  const start = getQHeadCoords(qIdx);
                  const end = getKVHeadCoords(kvIdx);
                  const isSel = qIdx === selectedQHead;
                  const isHovered = hoveredKVHead === kvIdx;

                  return (
                    <path 
                      key={`map-line-${qIdx}`}
                      d={`M ${start.x} ${start.y} C ${start.x + 80} ${start.y}, ${end.x - 80} ${end.y}, ${end.x} ${end.y}`}
                      fill="none"
                      stroke={isSel ? "var(--accent-color)" : "var(--text-muted)"}
                      strokeWidth={isSel ? "2.5" : isHovered ? "1.5" : "1.2"}
                      opacity={isSel ? 0.95 : isHovered ? 0.75 : 0.45}
                      strokeDasharray={isSel ? "5 4" : "none"}
                      style={{ 
                        transition: "all 0.3s",
                        animation: isSel ? "flowMarquee 0.8s linear infinite" : "none"
                      }}
                    />
                  );
                })}

                {/* Draw Query Nodes */}
                {Array.from({ length: numQHeads }).map((_, idx) => {
                  const pt = getQHeadCoords(idx);
                  const isSel = idx === selectedQHead;
                  return (
                    <circle 
                      key={`q-nd-${idx}`}
                      cx={pt.x} cy={pt.y} r={isSel ? "6.5" : "4.5"}
                      fill={isSel ? "var(--query-color)" : "var(--bg-card)"}
                      stroke={isSel ? "white" : "var(--border-color)"}
                      strokeWidth="1.5"
                      onClick={() => setSelectedQHead(idx)}
                      style={{ cursor: "pointer", transition: "all 0.2s" }}
                    />
                  );
                })}

                {/* Draw KV Nodes */}
                {Array.from({ length: numKVHeads }).map((_, idx) => {
                  const pt = getKVHeadCoords(idx);
                  const isMapped = Math.floor(selectedQHead / ratio) === idx;
                  const isHovered = hoveredKVHead === idx;

                  return (
                    <circle 
                      key={`kv-nd-${idx}`}
                      cx={pt.x} cy={pt.y} r={isMapped ? "7" : isHovered ? "6" : "5"}
                      fill={isMapped ? "var(--key-color)" : "var(--bg-card)"}
                      stroke={isMapped ? "white" : "var(--border-color)"}
                      strokeWidth="1.5"
                      style={{ transition: "all 0.2s" }}
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
          </div>

          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontStyle: "italic", textAlign: "center" }}>
            💡 Click a Query head to select it, or hover over a KV head to see groupings.
          </div>
        </div>

        {/* Right Panel: Projection multipliers with Tabs */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px", minHeight: isMathActive ? "auto" : "390px" }}>
            
            {/* Header with Tabs */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
              <span style={{ fontSize: "0.85rem", fontWeight: "700", color: "var(--text-primary)" }}>Linear Projection</span>
              
              <div className="tab-container">
                <button 
                  onClick={() => setProjTab("viz")}
                  className={`tab-button ${projTab === "viz" ? "active" : ""}`}
                >
                  Projection Flow
                </button>
                <button 
                  onClick={() => setProjTab("Q")}
                  className={`tab-button ${isMathActive ? "active" : ""}`}
                >
                  Projection Math
                </button>
              </div>
            </div>

            {/* TAB 1: VISUAL PROJECTION FLOW (Default) */}
            {projTab === "viz" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                  The single normalized input vector <strong>x_norm</strong> (size 8) branches into three parallel weights matrices (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>) simultaneously, producing the Q, K, and V vectors:
                </div>

                <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "10px", padding: "10px 0" }}>
                  <svg width="340" height="230" viewBox="0 0 340 230" style={{ overflow: "visible" }}>
                    <defs>
                      <linearGradient id="grad-q-flow" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="var(--query-color)" stopOpacity="0.8" />
                      </linearGradient>
                      <linearGradient id="grad-k-flow" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="var(--key-color)" stopOpacity="0.8" />
                      </linearGradient>
                      <linearGradient id="grad-v-flow" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                        <stop offset="100%" stopColor="var(--value-color)" stopOpacity="0.8" />
                      </linearGradient>
                    </defs>

                    {/* Left: Input Vector x */}
                    <g transform="translate(10, 95)">
                      <text x="35" y="-8" fill="var(--text-muted)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Input x_norm (d=8)</text>
                      <rect x="0" y="0" width="70" height="24" rx="4" fill="rgba(255,255,255,0.02)" stroke="var(--query-color)" strokeWidth="1.5" />
                      {Array.from({ length: 8 }).map((_, i) => (
                        <line key={i} x1={i * 8.7 + 8.7} y1="0" x2={i * 8.7 + 8.7} y2="24" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
                      ))}
                    </g>

                    {/* Branching paths to three lens matrices */}
                    {/* Q path */}
                    <path d="M 80 107 L 110 107 L 110 45 L 140 45" fill="none" stroke="var(--query-color)" strokeWidth="1.5" />
                    {/* K path */}
                    <path d="M 80 107 L 140 107" fill="none" stroke="var(--key-color)" strokeWidth="1.5" strokeDasharray="3 3" />
                    {/* V path */}
                    <path d="M 80 107 L 110 107 L 110 170 L 140 170" fill="none" stroke="var(--value-color)" strokeWidth="1.5" />

                    {/* Lens 1: W_Q */}
                    <g transform="translate(140, 20)">
                      <polygon points="0,5 40,0 40,50 0,45" fill="var(--bg-card)" stroke="var(--query-color)" strokeWidth="1.5" />
                      <text x="20" y="24" fill="var(--text-primary)" fontSize="0.5rem" textAnchor="middle" fontWeight="bold">Lens W_Q</text>
                      <text x="20" y="34" fill="var(--query-color)" fontSize="0.4rem" textAnchor="middle">[8x4]</text>
                    </g>
                    {/* Lens 2: W_K */}
                    <g transform="translate(140, 82)">
                      <polygon points="0,5 40,0 40,50 0,45" fill="var(--bg-card)" stroke="var(--key-color)" strokeWidth="1.5" />
                      <text x="20" y="24" fill="var(--text-primary)" fontSize="0.5rem" textAnchor="middle" fontWeight="bold">Lens W_K</text>
                      <text x="20" y="34" fill="var(--key-color)" fontSize="0.4rem" textAnchor="middle">[8x4]</text>
                    </g>
                    {/* Lens 3: W_V */}
                    <g transform="translate(140, 145)">
                      <polygon points="0,5 40,0 40,50 0,45" fill="var(--bg-card)" stroke="var(--value-color)" strokeWidth="1.5" />
                      <text x="20" y="24" fill="var(--text-primary)" fontSize="0.5rem" textAnchor="middle" fontWeight="bold">Lens W_V</text>
                      <text x="20" y="34" fill="var(--value-color)" fontSize="0.4rem" textAnchor="middle">[8x4]</text>
                    </g>

                    {/* Connect lenses to outputs */}
                    <path d="M 180 45 L 230 45" stroke="var(--query-color)" strokeWidth="1.5" />
                    <path d="M 180 107 L 230 107" stroke="var(--key-color)" strokeWidth="1.5" />
                    <path d="M 180 170 L 230 170" stroke="var(--value-color)" strokeWidth="1.5" />

                    {/* Output 1: Vector q */}
                    <g transform="translate(230, 33)">
                      <rect x="0" y="0" width="50" height="24" rx="4" fill="var(--bg-card)" stroke="var(--query-color)" strokeWidth="1.5" />
                      <text x="25" y="15" fill="var(--text-primary)" fontSize="0.52rem" textAnchor="middle" fontFamily="monospace">vector q</text>
                    </g>
                    {/* Output 2: Vector k */}
                    <g transform="translate(230, 95)">
                      <rect x="0" y="0" width="50" height="24" rx="4" fill="var(--bg-card)" stroke="var(--key-color)" strokeWidth="1.5" />
                      <text x="25" y="15" fill="var(--text-primary)" fontSize="0.52rem" textAnchor="middle" fontFamily="monospace">vector k</text>
                    </g>
                    {/* Output 3: Vector v */}
                    <g transform="translate(230, 158)">
                      <rect x="0" y="0" width="50" height="24" rx="4" fill="var(--bg-card)" stroke="var(--value-color)" strokeWidth="1.5" />
                      <text x="25" y="15" fill="var(--text-primary)" fontSize="0.52rem" textAnchor="middle" fontFamily="monospace">vector v</text>
                    </g>

                    {/* Flow Particles */}
                    <circle cx="80" cy="107" r="3.5" fill="var(--query-color)">
                      <animate attributeName="cx" values="80;110;110;140;180;230" keyTimes="0;0.2;0.2;0.5;0.75;1" dur="2.5s" repeatCount="indefinite" />
                      <animate attributeName="cy" values="107;107;45;45;45;45" keyTimes="0;0.2;0.2;0.5;0.75;1" dur="2.5s" repeatCount="indefinite" />
                    </circle>
                    <circle cx="80" cy="107" r="3.5" fill="var(--key-color)">
                      <animate attributeName="cx" values="80;140;180;230" keyTimes="0;0.5;0.75;1" dur="2.5s" repeatCount="indefinite" />
                      <animate attributeName="cy" values="107;107;107;107" keyTimes="0;0.5;0.75;1" dur="2.5s" repeatCount="indefinite" />
                    </circle>
                    <circle cx="80" cy="107" r="3.5" fill="var(--value-color)">
                      <animate attributeName="cx" values="80;110;110;140;180;230" keyTimes="0;0.2;0.2;0.5;0.75;1" dur="2.5s" repeatCount="indefinite" />
                      <animate attributeName="cy" values="107;107;170;170;170;170" keyTimes="0;0.2;0.2;0.5;0.75;1" dur="2.5s" repeatCount="indefinite" />
                    </circle>
                  </svg>
                </div>
              </div>
            )}

            {/* TAB 2: INTERACTIVE PROJECTION MATH */}
            {isMathActive && (
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                
                {/* Tab specific Q/K/V toggle */}
                <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-surface)", padding: "2px", borderRadius: "6px", border: "1px solid var(--border-color)", alignSelf: "flex-end" }}>
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
            )}

          </div>

        </div>

      </div>

    </div>
  );
}
