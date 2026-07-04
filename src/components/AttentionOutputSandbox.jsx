import React, { useState } from "react";
import { Network, ArrowDown, HelpCircle, Layers, Cpu, Sparkles } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function AttentionOutputSandbox({ simData }) {
  const [selectedHead, setSelectedHead] = useState(0);
  const [activeTab, setActiveTab] = useState("viz"); // "viz" or "math"

  const { concatenatedHeads, attentionOutputProj, W_O, attentionWeights, vVectors } = simData.gqa;
  const tokens = simData.allTokens;
  const seqLen = tokens.length;
  const numHeads = 8;
  const headDim = 4;
  const ratio = 4; // GQA ratio: 8 Query heads, 2 KV heads (4:1)

  const activeKVHead = Math.floor(selectedHead / ratio);

  // Read weights and values for Selected Head
  const weights = attentionWeights[selectedHead] || Array(seqLen).fill(0);
  
  // Pack Value vectors for this head group: shape [seqLen x headDim (4)]
  const valuesStack = tokens.map((_, tIdx) => vVectors[tIdx]?.[activeKVHead] || Array(headDim).fill(0));

  // Compute head output vector [1 x 4]
  const headOutput = Array(headDim).fill(0);
  for (let d = 0; d < headDim; d++) {
    for (let t = 0; t < seqLen; t++) {
      headOutput[d] += weights[t] * valuesStack[t][d];
    }
    headOutput[d] = parseFloat(headOutput[d].toFixed(3));
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--accent-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Network className="w-6 h-6 text-orange-400 animate-pulse-slow" />
          Step 7: Value Weighted Sum & Output Projection (O-Proj)
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          The final self-attention stage aggregates information and projects it back to the hidden dimension:
          <br />
          <strong>Stage 1 (Weighted Values Sum):</strong> The attention weights [1 x T] scale the Value vectors [T x 4] of all prior tokens. Adding these gives the output of each head [1 x 4].
          <br />
          <strong>Stage 2 (Output Projection):</strong> The outputs of all 8 heads are <strong>concatenated</strong> into a single vector of size 32, and multiplied by the projection matrix <strong>W<sub>O</sub></strong> (size 32 &times; 8) to yield the final attention state.
        </p>
      </div>

      {/* Tabs Selector */}
      <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "16px", minHeight: activeTab === "viz" ? "420px" : "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Cpu className="w-4 h-4 text-orange-400 animate-pulse" />
            <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--text-primary)", margin: 0 }}>
              Aggregation & Projection Engine
            </h3>
          </div>
          
          <div className="tab-container">
            <button 
              onClick={() => setActiveTab("viz")}
              className={`tab-button ${activeTab === "viz" ? "active" : ""}`}
            >
              Merge & Projection Diagram
            </button>
            <button 
              onClick={() => setActiveTab("math")}
              className={`tab-button ${activeTab === "math" ? "active" : ""}`}
            >
              Concatenation Matrix Math
            </button>
          </div>
        </div>

        {/* TAB 1: VISUAL DIAGRAM */}
        {activeTab === "viz" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              The output vectors from all <strong>8 attention heads</strong> (each dimension 4) are stacked side-by-side to form a 32-element vector, which is projected back down to 8 dimensions by <strong>W_O</strong>.
            </div>

            <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "10px", padding: "15px 0" }}>
              <svg width="580" height="230" viewBox="0 0 580 230" style={{ overflow: "visible" }}>
                <defs>
                  <linearGradient id="grad-merge" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="var(--accent-color)" stopOpacity="0.8" />
                  </linearGradient>
                </defs>

                {/* 1. Heads inputs stack */}
                <g transform="translate(10, 10)">
                  <text x="35" y="0" fill="var(--text-muted)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">8 Heads (each d=4)</text>
                  {Array.from({ length: 8 }).map((_, hIdx) => {
                    const y = hIdx * 24 + 10;
                    const isSel = hIdx === selectedHead;
                    return (
                      <g key={hIdx} transform={`translate(0, ${y})`} style={{ cursor: "pointer" }} onClick={() => setSelectedHead(hIdx)}>
                        <rect 
                          x="0" y="0" width="70" height="18" rx="3" 
                          fill={isSel ? "var(--query-glow)" : "rgba(255,255,255,0.02)"} 
                          stroke={isSel ? "var(--query-color)" : "rgba(255,255,255,0.08)"} 
                          strokeWidth={isSel ? "1.5" : "0.5"} 
                        />
                        <text x="35" y="12" fill={isSel ? "var(--text-primary)" : "var(--text-secondary)"} fontSize="0.6rem" textAnchor="middle" fontWeight="bold">Head Q{hIdx}</text>
                      </g>
                    );
                  })}
                </g>

                {/* 2. Concatenation strip */}
                <g transform="translate(140, 20)">
                  <rect x="0" y="0" width="30" height="190" rx="4" fill="rgba(255,255,255,0.02)" stroke="var(--accent-color)" strokeWidth="1.5" />
                  <text x="15" y="-5" fill="var(--accent-color)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Concatenated [32]</text>
                  
                  {/* Visual segments */}
                  {Array.from({ length: 8 }).map((_, hIdx) => {
                    const y = hIdx * 23 + 2;
                    const isSel = hIdx === selectedHead;
                    return (
                      <rect 
                        key={hIdx} 
                        x="2" y={y} width="26" height="19" rx="2" 
                        fill={isSel ? "var(--accent-color)" : "rgba(255,255,255,0.1)"} 
                        style={{ transition: "fill 0.3s" }} 
                      />
                    );
                  })}
                </g>

                {/* Connecting converging lines */}
                {Array.from({ length: 8 }).map((_, hIdx) => {
                  const yStart = hIdx * 24 + 29; // 10 (parent y) + hIdx*24+10 (local y) + 9 (height/2)
                  const yEnd = hIdx * 23 + 31.5; // 20 (parent y) + hIdx*23+2 (local y) + 9.5 (height/2)
                  const isSel = hIdx === selectedHead;
                  return (
                    <line 
                      key={hIdx} 
                      x1="80" y1={yStart} x2="142" y2={yEnd} 
                      stroke={isSel ? "var(--accent-color)" : "var(--text-muted)"} 
                      strokeWidth={isSel ? "2.2" : "0.85"} 
                      opacity={isSel ? 0.95 : 0.15}
                      strokeDasharray={isSel ? "4 3" : "none"}
                      style={{ 
                        animation: isSel ? "flowMarquee 0.8s linear infinite" : "none" 
                      }}
                    />
                  );
                })}

                {/* Projection Lens (W_O) */}
                <g transform="translate(230, 85)">
                  <polygon points="0,0 80,-20 80,80 0,60" fill="var(--bg-card)" stroke="var(--attention-color)" strokeWidth="1.5" style={{ filter: "drop-shadow(0 0 5px var(--attention-glow))" }} />
                  <text x="40" y="32" fill="var(--text-primary)" fontSize="0.65rem" textAnchor="middle" fontWeight="bold">Lens W_O</text>
                  <text x="40" y="45" fill="var(--attention-color)" fontSize="0.55rem" textAnchor="middle">[32 x 8]</text>
                </g>

                {/* Connection lines from Concatenation to Lens */}
                <path d="M 170 115 L 230 115" stroke="var(--accent-color)" strokeWidth="2" strokeDasharray="3 3" />

                {/* 3. Output Projected Vector */}
                <g transform="translate(380, 75)">
                  <rect x="0" y="0" width="160" height="60" rx="8" fill="var(--bg-surface)" stroke="var(--success-color)" strokeWidth="2" />
                  <text x="80" y="-8" fill="var(--success-color)" fontSize="0.65rem" textAnchor="middle" fontWeight="bold">Projected Attention Output [1x8]</text>
                  
                  {/* Grid of values */}
                  <g transform="translate(10, 18)">
                    {attentionOutputProj.map((val, d) => (
                      <g key={d} transform={`translate(${d * 18}, 0)`}>
                        <rect x="0" y="0" width="16" height="24" rx="2" fill="var(--bg-card)" stroke="var(--border-color)" strokeWidth="0.5" />
                        <text x="8" y="15" fill="var(--text-primary)" fontSize="0.52rem" textAnchor="middle" fontFamily="monospace">{val.toFixed(1)}</text>
                      </g>
                    ))}
                  </g>
                </g>

                {/* Connecting lines from Lens to Output */}
                <path d="M 310 115 L 380 105" stroke="var(--success-color)" strokeWidth="2" strokeDasharray="3 3" />
                
                {/* Flow Particles */}
                <circle cx="170" cy="115" r="4" fill="var(--accent-color)">
                  <animate attributeName="cx" values="170;230" dur="1.5s" repeatCount="indefinite" />
                </circle>
                <circle cx="310" cy="115" r="4" fill="var(--success-color)">
                  <animate attributeName="cx" values="310;380" dur="1.5s" repeatCount="indefinite" />
                  <animate attributeName="cy" values="115;105" dur="1.5s" repeatCount="indefinite" />
                </circle>
              </svg>
            </div>
          </div>
        )}

        {/* TAB 2: CONCATENATION & MATRIX MATH */}
        {activeTab === "math" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
            
            {/* Select Head Tabs */}
            <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: "600" }}>Inspect Query Head:</span>
              <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-surface)", padding: "3px", borderRadius: "8px", border: "1px solid var(--border-color)", flexWrap: "wrap" }}>
                {Array.from({ length: numHeads }).map((_, hIdx) => (
                  <button
                    key={`h-btn-${hIdx}`}
                    onClick={() => setSelectedHead(hIdx)}
                    style={{
                      padding: "4px 8px",
                      borderRadius: "6px",
                      border: "none",
                      backgroundColor: selectedHead === hIdx ? "var(--accent-color)" : "transparent",
                      color: selectedHead === hIdx ? "white" : "var(--text-secondary)",
                      fontSize: "0.75rem",
                      fontWeight: "700",
                      cursor: "pointer",
                      transition: "all 0.15s ease"
                    }}
                  >
                    Head Q{hIdx}
                  </button>
                ))}
              </div>
            </div>

            {/* Stage 1: Weighted Sum */}
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--query-color)", fontWeight: "700" }}>
                Stage 1: Attention Head Q{selectedHead} Value Aggregation [1 &times; T] &times; [T &times; 4]
              </span>
              <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
                <InteractiveMatrixMul
                  leftVal={weights}
                  rightVal={valuesStack}
                  outputVal={headOutput}
                  leftLabel={`Head Q${selectedHead} Softmax Weights`}
                  rightLabel={`Value Vectors (KV Group ${activeKVHead})`}
                  outputLabel="Head Output Vector"
                  leftColor="var(--attention-color)"
                  rightColor="var(--value-color)"
                  outputColor="var(--success-color)"
                  leftRowNames={[`"Weights"`]}
                  rightColNames={["d0", "d1", "d2", "d3"]}
                />
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", margin: "-6px 0" }}>
              <ArrowDown className="w-5 h-5 text-text-muted" />
            </div>

            {/* Stage 2: Output Projection */}
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--success-color)", fontWeight: "700" }}>
                Stage 2: Concatenate & Output Projection W_O [1 &times; 32] &times; [32 &times; 8]
              </span>
              <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
                <InteractiveMatrixMul
                  leftVal={concatenatedHeads}
                  rightVal={W_O}
                  outputVal={attentionOutputProj}
                  leftLabel="Concatenated Head Outputs"
                  rightLabel="Weight Matrix W_O"
                  outputLabel="Projected Hidden Vector"
                  leftColor="var(--accent-color)"
                  rightColor="var(--key-color)"
                  outputColor="var(--success-color)"
                  leftRowNames={["Concatenated"]}
                  rightColNames={Array.from({ length: 8 }, (_, i) => `d${i}`)}
                />
              </div>
            </div>

          </div>
        )}

      </div>

    </div>
  );
}
