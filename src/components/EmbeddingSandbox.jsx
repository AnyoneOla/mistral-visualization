import React, { useState } from "react";
import { Brain, Search, Info } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function EmbeddingSandbox({ simData }) {
  const { vocab, activeWord, activeIndex, embeddingWeights, outputVector } = simData.embeddingLookup;
  const [searchTerm, setSearchTerm] = useState("");
  
  // Reference for scrolling the active vocab item into view locally
  const activeVocabRef = React.useRef(null);
  const containerRef = React.useRef(null);
  
  React.useEffect(() => {
    if (activeVocabRef.current && containerRef.current) {
      const container = containerRef.current;
      const element = activeVocabRef.current;
      const topPos = element.offsetTop - container.offsetTop;
      container.scrollTo({ top: topPos - container.clientHeight / 2 + element.clientHeight / 2, behavior: "smooth" });
    }
  }, [activeIndex]);

  // Localized slice for the Matrix Multiplication visualizer (keep it 8x8 max so it doesn't overflow)
  const matrixRange = 8;
  const startIdx = Math.max(0, Math.min(vocab.length - matrixRange, activeIndex - Math.floor(matrixRange / 2)));
  
  const localVocab = vocab.slice(startIdx, startIdx + matrixRange);
  const localWeights = embeddingWeights.slice(startIdx, startIdx + matrixRange);
  
  const localOneHot = Array(matrixRange).fill(0);
  const localActiveIdx = activeIndex - startIdx;
  if (localActiveIdx >= 0 && localActiveIdx < matrixRange) {
    localOneHot[localActiveIdx] = 1;
  }

  // Filter vocabulary for searching
  const filteredVocab = vocab
    .map((word, index) => ({ word, index }))
    .filter(item => item.word.toLowerCase().includes(searchTerm.toLowerCase()));

  const [activeTab, setActiveTab] = useState("viz"); // "viz" or "math"

  // Y positions for table rows in the diagram
  const tableRows = Array.from({ length: 7 }, (_, i) => {
    const idx = activeIndex - 3 + i;
    const isValid = idx >= 0 && idx < vocab.length;
    return {
      idx,
      word: isValid ? vocab[idx] : "---",
      isCurrent: idx === activeIndex
    };
  });

  const activeRowIndex = tableRows.findIndex(r => r.isCurrent);
  const activeRowY = 10 + (activeRowIndex >= 0 ? activeRowIndex : 3) * 28 + 12;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--query-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Brain className="w-6 h-6 text-indigo-400 animate-pulse-slow" />
          Step 1: Input & Token Embedding Lookup
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Before an LLM can perform mathematical equations, text is split into subwords called <strong>tokens</strong>. 
          Each token has a unique vocabulary index. We look up this index in the <strong>Embedding Weights Matrix (W<sub>embed</sub>)</strong> to retrieve the dense vector representing that token.
        </p>
        <div style={{ 
          display: "flex", 
          alignItems: "flex-start", 
          gap: "8px", 
          fontSize: "0.8rem", 
          color: "var(--text-muted)", 
          backgroundColor: "rgba(255,255,255,0.02)", 
          padding: "10px", 
          borderRadius: "8px",
          border: "1px dashed var(--border-color)"
        }}>
          <Info className="w-4.5 h-4.5 text-accent-color flex-shrink-0" style={{ marginTop: "2px" }} />
          <span>
            <strong>Educational Insight:</strong> An embedding lookup is mathematically identical to multiplying a <strong>one-hot encoded index vector</strong> (a vector of zeros with a single '1' at the token's vocabulary index) by the weight matrix. Hover over the cells below to see this dot product in action!
          </span>
        </div>
      </div>

      {/* Visual Workspace Grid */}
      <div className="responsive-grid">
        
        {/* Left Panel: Vocabulary & Search */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: "700", color: "var(--query-color)" }}>Vocabulary Index</span>
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>({vocab.length} items)</span>
          </div>

          {/* Search box */}
          <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
            <Search width={14} height={14} className="text-text-muted" style={{ position: "absolute", left: "10px", pointerEvents: "none" }} />
            <input 
              type="text"
              placeholder="Search vocabulary..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: "100%",
                padding: "6px 10px 6px 30px",
                backgroundColor: "var(--bg-surface)",
                border: "1px solid var(--border-color)",
                borderRadius: "6px",
                fontSize: "0.75rem",
                color: "var(--text-primary)",
                outline: "none"
              }}
            />
          </div>

          {/* Vocab row list */}
          <div ref={containerRef} style={{ display: "flex", flexDirection: "column", gap: "4px", maxHeight: "330px", overflowY: "auto", paddingRight: "4px" }}>
            {filteredVocab.map(({ word, index }) => {
              const isActive = index === activeIndex;
              return (
                <div 
                  key={`vocab-row-${index}`}
                  ref={isActive ? activeVocabRef : null}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "6px 10px",
                    backgroundColor: isActive ? "var(--query-glow)" : "rgba(255,255,255,0.01)",
                    border: `1px solid ${isActive ? "var(--query-color)" : "transparent"}`,
                    borderRadius: "6px",
                    fontSize: "0.75rem",
                    transition: "all 0.2s ease"
                  }}
                >
                  <div style={{ display: "flex", gap: "8px", overflow: "hidden", textOverflow: "ellipsis" }}>
                    <span style={{ fontFamily: "monospace", color: "var(--text-muted)" }}>#{index}</span>
                    <span style={{ fontWeight: isActive ? "700" : "500", color: isActive ? "var(--text-primary)" : "var(--text-secondary)" }}>"{word}"</span>
                  </div>
                  {isActive && (
                    <span style={{ fontSize: "0.55rem", padding: "1px 5px", backgroundColor: "var(--accent-glow)", border: "1px solid var(--accent-color)", color: "var(--accent-color)", borderRadius: "4px", fontWeight: "700" }}>
                      ACTIVE
                    </span>
                  )}
                </div>
              );
            })}
            {filteredVocab.length === 0 && (
              <div style={{ textAlignment: "center", padding: "20px", color: "var(--text-muted)", fontSize: "0.75rem" }}>
                No tokens found.
              </div>
            )}
          </div>
        </div>

        {/* Right Panel with Tabs: Visual Lookup vs. Matrix Multiply */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px", minHeight: activeTab === "viz" ? "390px" : "auto" }}>
          
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
            <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--text-primary)", margin: 0 }}>
              Embedding Lookup Engine
            </h3>
            
            <div className="tab-container">
              <button 
                onClick={() => setActiveTab("viz")}
                className={`tab-button ${activeTab === "viz" ? "active" : ""}`}
              >
                Lookup Diagram
              </button>
              <button 
                onClick={() => setActiveTab("math")}
                className={`tab-button ${activeTab === "math" ? "active" : ""}`}
              >
                Matrix Multiplication Math
              </button>
            </div>
          </div>

          {/* TAB 1: VISUAL DIAGRAM */}
          {activeTab === "viz" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                Active word <strong>"{activeWord}"</strong> triggers an immediate memory lookup from the embedding weights table (W_embed).
              </div>

              <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "10px", padding: "15px 0" }}>
                <svg width="580" height="230" viewBox="0 0 580 230" style={{ overflow: "visible" }}>
                  <defs>
                    <linearGradient id="grad-lookup-line" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                      <stop offset="50%" stopColor="var(--accent-color)" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="var(--success-color)" stopOpacity="0.8" />
                    </linearGradient>
                  </defs>

                  {/* 1. Token Input Block */}
                  <g transform="translate(15, 80)">
                    <rect x="0" y="0" width="130" height="60" rx="8" fill="var(--bg-surface)" stroke="var(--query-color)" strokeWidth="2" style={{ filter: "drop-shadow(0 0 5px var(--query-glow))" }} />
                    <text x="65" y="24" fill="var(--text-primary)" fontSize="0.75rem" textAnchor="middle" fontWeight="bold">Active Token</text>
                    <text x="65" y="44" fill="var(--query-color)" fontSize="0.85rem" textAnchor="middle" fontWeight="800">"{activeWord}"</text>
                    <text x="65" y="75" fill="var(--text-muted)" fontSize="0.65rem" textAnchor="middle">Index: #{activeIndex}</text>
                  </g>

                  {/* 2. Vocabulary Table */}
                  <g transform="translate(195, 10)">
                    <text x="45" y="-3" fill="var(--text-muted)" fontSize="0.6rem" textAnchor="middle" fontWeight="bold">W_embed Matrix</text>
                    {tableRows.map((row, i) => {
                      const y = i * 28;
                      return (
                        <g key={`tab-row-${row.idx}`} transform={`translate(0, ${y})`}>
                          {/* Row background */}
                          <rect 
                            x="0" y="0" width="160" height="24" rx="4" 
                            fill={row.isCurrent ? "var(--query-glow)" : "rgba(255,255,255,0.02)"} 
                            stroke={row.isCurrent ? "var(--query-color)" : "rgba(255,255,255,0.05)"} 
                            strokeWidth={row.isCurrent ? "1.5" : "0.5"} 
                          />
                          {/* Index */}
                          <text x="8" y="15" fill="var(--text-muted)" fontSize="0.6rem" fontFamily="monospace">#{row.idx}</text>
                          {/* Word */}
                          <text x="42" y="15" fill={row.isCurrent ? "var(--accent-color)" : "var(--text-secondary)"} fontSize="0.68rem" fontWeight={row.isCurrent ? "800" : "normal"}>
                            "{row.word}"
                          </text>
                          {/* Value bars */}
                          <g transform="translate(100, 7)">
                            {Array.from({ length: 4 }).map((_, d) => (
                              <rect key={d} x={d * 12} y="0" width="8" height="10" rx="1.5" fill={row.isCurrent ? "var(--accent-color)" : "rgba(255,255,255,0.15)"} />
                            ))}
                          </g>
                        </g>
                      );
                    })}
                  </g>

                  {/* 3. Connecting Scan Line */}
                  <path d={`M 145 110 L 195 ${activeRowY}`} fill="none" stroke="var(--query-color)" strokeWidth="2" strokeDasharray="3 3" opacity="0.8" />

                  {/* 4. Output Vector Block */}
                  <g transform="translate(425, 60)">
                    <rect x="-10" y="-20" width="160" height="100" rx="8" fill="var(--bg-surface)" stroke="var(--success-color)" strokeWidth="1.5" />
                    <text x="70" y="-6" fill="var(--success-color)" fontSize="0.7rem" textAnchor="middle" fontWeight="bold">Output Embedding x [1x8]</text>
                    
                    {/* Grid of values */}
                    <g transform="translate(0, 15)">
                      {outputVector.slice(0, 4).map((val, d) => (
                        <g key={d} transform={`translate(${d * 34}, 0)`}>
                          <rect x="0" y="0" width="30" height="24" rx="4" fill="var(--bg-card)" stroke="var(--border-color)" strokeWidth="0.5" />
                          <text x="15" y="15" fill="var(--text-primary)" fontSize="0.62rem" textAnchor="middle" fontFamily="monospace">{val.toFixed(2)}</text>
                        </g>
                      ))}
                      {outputVector.slice(4, 8).map((val, d) => (
                        <g key={d + 4} transform={`translate(${d * 34}, 30)`}>
                          <rect x="0" y="0" width="30" height="24" rx="4" fill="var(--bg-card)" stroke="var(--border-color)" strokeWidth="0.5" />
                          <text x="15" y="15" fill="var(--text-primary)" fontSize="0.62rem" textAnchor="middle" fontFamily="monospace">{val.toFixed(2)}</text>
                        </g>
                      ))}
                    </g>
                  </g>

                  {/* Flow Particles */}
                  <path d={`M 355 ${activeRowY} L 425 110`} fill="none" stroke="url(#grad-lookup-line)" strokeWidth="1.5" strokeDasharray="5 5">
                    <animate attributeName="strokeDashoffset" values="30;0" dur="1.2s" repeatCount="indefinite" />
                  </path>
                  <circle cx="355" cy={activeRowY} r="4" fill="var(--success-color)">
                    <animate attributeName="cx" values="355;425" dur="1.2s" repeatCount="indefinite" />
                    <animate attributeName="cy" values={`${activeRowY};110`} dur="1.2s" repeatCount="indefinite" />
                  </circle>

                </svg>
              </div>
            </div>
          )}

          {/* TAB 2: MATRIX MULTIPLICATION */}
          {activeTab === "math" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
                <InteractiveMatrixMul
                  leftVal={localOneHot}
                  rightVal={localWeights}
                  outputVal={outputVector}
                  leftLabel="One-hot Token Vector"
                  rightLabel="Embedding Matrix W_embed"
                  outputLabel="Token Vector x"
                  leftColor="var(--query-color)"
                  rightColor="var(--key-color)"
                  outputColor="var(--success-color)"
                  leftRowNames={[`"${activeWord}"`]}
                  rightColNames={Array.from({ length: outputVector.length }, (_, i) => `d${i}`)}
                />
              </div>

              <div style={{ 
                fontSize: "0.75rem", 
                color: "var(--text-secondary)", 
                lineHeight: "1.4",
                padding: "12px",
                backgroundColor: "var(--bg-surface)",
                borderRadius: "8px",
                border: "1px solid var(--border-color)",
                marginTop: "10px"
              }}>
                💡 Notice how the <strong>One-hot Vector</strong> has all 0s except a single 1 at index <strong>#{activeIndex}</strong>. 
                When multiplied by the weight matrix, it acts as a selective filter that extracts exactly row #{activeIndex} (representing token <strong>"{activeWord}"</strong>) to yield the <strong>Token Vector x [1 &times; 8]</strong>.
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
