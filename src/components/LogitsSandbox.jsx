import React, { useState, useMemo } from "react";
import { Sparkles, Flame, Sliders, Info, Zap, Cpu } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function LogitsSandbox({ 
  simData, 
  temperature, 
  setTemperature, 
  topK, 
  setTopK, 
  topP, 
  setTopP 
}) {
  const [hoveredLogit, setHoveredLogit] = useState(null); // index of hovered candidate
  const [activeTab, setActiveTab] = useState("viz"); // "viz" or "math"

  const { vocab, W_unembed, rawLogits, correctWord, correctIndex, validWords } = simData.logitsData;
  const finalHiddenState = simData.swiglu.finalHiddenState;
  const hiddenDim = simData.dimensions.hiddenDim;

  const W_unembed_transposed = useMemo(() => {
    return Array.from({ length: hiddenDim }, (_, r) => 
      Array.from({ length: vocab.length }, (_, c) => {
        const val = W_unembed[c]?.[r] ?? 0;
        return parseFloat((val / Math.max(0.05, temperature)).toFixed(3));
      })
    );
  }, [W_unembed, hiddenDim, vocab.length, temperature]);

  const scaledLogitsForMatrix = useMemo(() => {
    return rawLogits.map(l => parseFloat((l / Math.max(0.05, temperature)).toFixed(3)));
  }, [rawLogits, temperature]);

  // Recalculate candidate steps (Sort -> Top-K -> Top-P Softmax)
  const candidateSteps = useMemo(() => {
    const scaledLogits = rawLogits.map(l => l / Math.max(0.05, temperature));
    
    const items = vocab.map((word, idx) => ({
      word,
      rawLogit: rawLogits[idx],
      scaledLogit: scaledLogits[idx],
      index: idx,
      isValidPath: validWords.includes(word),
      isCorrect: idx === correctIndex
    }));
    
    // Sort descending
    items.sort((a, b) => b.scaledLogit - a.scaledLogit);
    
    // Mark Top-K
    const kMapped = items.map((item, idx) => ({
      ...item,
      rank: idx + 1,
      status: idx < topK ? "Candidate" : "Masked (Top-K)"
    }));
    
    // Calculate Softmax on Top-K active elements
    const topKItems = kMapped.filter(x => x.status === "Candidate");
    const maxLogit = Math.max(...topKItems.map(x => x.scaledLogit));
    const exps = topKItems.map(x => Math.exp(x.scaledLogit - maxLogit));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => sumExp > 0 ? e / sumExp : 0);
    
    let cumulative = 0;
    const finalItems = kMapped.map(item => {
      if (item.status === "Masked (Top-K)") {
        return {
          ...item,
          prob: 0,
          cumProb: 0,
          status: "Masked (Top-K)"
        };
      }
      
      const kIndex = topKItems.findIndex(x => x.index === item.index);
      const prob = probs[kIndex];
      cumulative += prob;
      
      const prevCum = cumulative - prob;
      const inTopP = prevCum < topP;
      
      return {
        ...item,
        inTopP,
        prob,
        cumProb: cumulative,
        status: inTopP ? "Active" : "Masked (Top-P)"
      };
    });
    
    const activeItems = finalItems.filter(item => item.inTopP);
    const activeSum = activeItems.reduce((a, b) => a + b.prob, 0);
    
    return finalItems.map(item => {
      if (!item.inTopP) return { ...item, finalProb: 0 };
      return {
        ...item,
        finalProb: activeSum > 0 ? item.prob / activeSum : 0
      };
    });
  }, [vocab, rawLogits, correctIndex, temperature, topK, topP, validWords]);

  const topCandidates = useMemo(() => {
    return candidateSteps.slice(0, 5);
  }, [candidateSteps]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--attention-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse-slow" />
          Step 10: Unembedding (LM Head) & Softmax Sampler
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          At the final layer, the hidden state vector h (1 &times; 8) is multiplied by the transposed unembedding weights matrix <strong>W<sub>unembed</sub></strong> (LM Head) to yield raw scores called <strong>Logits</strong>.
          These logits are scaled by <strong>Temperature (T)</strong>, filtered using <strong>Top-K</strong> and <strong>Top-P (Nucleus)</strong> algorithms, and normalized through a <strong>Softmax</strong> probability distribution to sample the next token.
        </p>
      </div>

      {/* Main Grid */}
      <div className="responsive-grid">
        
        {/* Left: LM Head Unembedding tabbed visualizer */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px", minHeight: activeTab === "viz" ? "390px" : "auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Cpu className="w-4 h-4 text-sky-400 animate-pulse" />
              <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--text-primary)", margin: 0 }}>
                LM Head Projection
              </h3>
            </div>

            <div className="tab-container">
              <button 
                onClick={() => setActiveTab("viz")}
                className={`tab-button ${activeTab === "viz" ? "active" : ""}`}
              >
                Prediction Diagram
              </button>
              <button 
                onClick={() => setActiveTab("math")}
                className={`tab-button ${activeTab === "math" ? "active" : ""}`}
              >
                Unembedding Matrix Math
              </button>
            </div>
          </div>

          {/* TAB 1: VISUAL PREDICTION DIAGRAM */}
          {activeTab === "viz" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                The final hidden vector <strong>h</strong> (8 values) is mapped via unembedding weights to scores for the whole vocabulary.
              </div>

              <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "10px", padding: "15px 0" }}>
                <svg width="340" height="230" viewBox="0 0 340 230" style={{ overflow: "visible" }}>
                  <defs>
                    <linearGradient id="grad-unembed-flow" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="var(--attention-color)" stopOpacity="0.8" />
                    </linearGradient>
                  </defs>

                  {/* 1. Final Hidden State */}
                  <g transform="translate(10, 30)">
                    <text x="12" y="-8" fill="var(--text-muted)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Vector h (d=8)</text>
                    {finalHiddenState.map((val, idx) => {
                      const y = idx * 19;
                      return (
                        <g key={idx} transform={`translate(0, ${y})`}>
                          <rect x="0" y="0" width="24" height="15" rx="2" fill="var(--bg-card)" stroke="var(--query-color)" strokeWidth="0.5" />
                          <text x="12" y="10" fill="var(--text-primary)" fontSize="0.55rem" textAnchor="middle" fontFamily="monospace">{val.toFixed(1)}</text>
                        </g>
                      );
                    })}
                  </g>

                  {/* 2. Projection lens LM Head */}
                  <g transform="translate(70, 70)">
                    <polygon points="0,0 60,-15 60,85 0,70" fill="var(--bg-card)" stroke="var(--query-color)" strokeWidth="1.5" />
                    <text x="30" y="32" fill="var(--text-primary)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">LM Head</text>
                    <text x="30" y="44" fill="var(--text-muted)" fontSize="0.45rem" textAnchor="middle">[8 x Vocab]</text>
                  </g>

                  <path d="M 34 105 L 70 105" stroke="var(--query-color)" strokeWidth="1" strokeDasharray="2 2" />

                  {/* 3. Top vocabulary logits display */}
                  <g transform="translate(160, 20)">
                    <text x="65" y="-5" fill="var(--attention-color)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Vocabulary Logits Selection</text>
                    
                    {topCandidates.map((cand, idx) => {
                      const y = idx * 36;
                      const isCorrect = cand.index === correctIndex;
                      return (
                        <g key={idx} transform={`translate(0, ${y})`}>
                          {/* Label */}
                          <text x="0" y="12" fill={isCorrect ? "var(--success-color)" : "var(--text-primary)"} fontSize="0.65rem" fontWeight={isCorrect ? "800" : "normal"}>
                            "{cand.word}"
                          </text>
                          {/* Probability bar */}
                          <rect x="0" y="18" width="130" height="6" rx="3" fill="var(--bg-surface)" />
                          <rect 
                            x="0" y="18" 
                            width={130 * Math.max(0.05, cand.finalProb)} 
                            height="6" rx="3" 
                            fill={isCorrect ? "var(--success-color)" : "var(--attention-color)"} 
                          />
                          <text x="135" y="24" fill="var(--text-muted)" fontSize="0.55rem" fontFamily="monospace">{(cand.finalProb * 100).toFixed(0)}%</text>
                        </g>
                      );
                    })}
                  </g>

                  <path d="M 130 105 L 160 105" stroke="var(--attention-color)" strokeWidth="1" strokeDasharray="2 2" />

                  {/* Animation particles */}
                  <circle cx="34" cy="105" r="3" fill="var(--query-color)">
                    <animate attributeName="cx" values="34;70" dur="1.5s" repeatCount="indefinite" />
                  </circle>
                  <circle cx="130" cy="105" r="3" fill="var(--attention-color)">
                    <animate attributeName="cx" values="130;160" dur="1.5s" repeatCount="indefinite" />
                  </circle>

                </svg>
              </div>
            </div>
          )}

          {/* TAB 2: INTERACTIVE MATRIX MULTIPLICATION */}
          {activeTab === "math" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
                <InteractiveMatrixMul
                  leftVal={finalHiddenState}
                  rightVal={W_unembed_transposed}
                  outputVal={scaledLogitsForMatrix}
                  leftLabel="Final Hidden Vector h"
                  rightLabel="Weights W_unembed (Scaled 1/T)"
                  outputLabel="Scaled Logits vector"
                  leftColor="var(--query-color)"
                  rightColor="var(--key-color)"
                  outputColor="var(--success-color)"
                  leftRowNames={["h"]}
                  rightColNames={vocab}
                />
              </div>
            </div>
          )}

        </div>

        {/* Right: Softmax Sampler Panel */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "20px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--attention-color)", display: "flex", alignItems: "center", gap: "6px", margin: 0 }}>
            <Flame className="w-5 h-5 text-orange-400 animate-pulse" />
            Softmax Sampler Configuration
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            
            {/* Temperature slider */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "4px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Temperature (T):</span>
                <span style={{ fontWeight: "700", color: "var(--success-color)", fontFamily: "monospace" }}>T = {temperature.toFixed(2)}</span>
              </div>
              <input
                type="range" min="0.1" max="2.0" step="0.05" value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="custom-slider"
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.6rem", color: "var(--text-muted)", marginTop: "2px" }}>
                <span>Greedy (0.1)</span>
                <span>Random (2.0)</span>
              </div>
            </div>

            {/* Top-K slider */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "4px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Top-K (K):</span>
                <span style={{ fontWeight: "700", color: "var(--query-color)", fontFamily: "monospace" }}>K = {topK}</span>
              </div>
              <input
                type="range" min="1" max="8" step="1" value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="custom-slider"
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.6rem", color: "var(--text-muted)", marginTop: "2px" }}>
                <span>Top 1</span>
                <span>All 8</span>
              </div>
            </div>

            {/* Top-P slider */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "4px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Top-P (Nucleus P):</span>
                <span style={{ fontWeight: "700", color: "var(--attention-color)", fontFamily: "monospace" }}>P = {topP.toFixed(2)}</span>
              </div>
              <input
                type="range" min="0.1" max="1.0" step="0.05" value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                className="custom-slider"
              />
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.6rem", color: "var(--text-muted)", marginTop: "2px" }}>
                <span>Narrow (0.1)</span>
                <span>Full (1.0)</span>
              </div>
            </div>

          </div>

          {/* Sorted candidate probability layout */}
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "10px" }}>
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Vocabulary Candidate Distribution
            </span>

            <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxHeight: "250px", overflowY: "auto" }}>
              {candidateSteps.map((candidate, idx) => {
                const isActive = candidate.status === "Active";
                const isMaskedK = candidate.status === "Masked (Top-K)";
                const isMaskedP = candidate.status === "Masked (Top-P)";
                const percent = (candidate.finalProb * 100).toFixed(1);

                return (
                  <div
                    key={`c-row-${idx}`}
                    onMouseEnter={() => setHoveredLogit(candidate.index)}
                    onMouseLeave={() => setHoveredLogit(null)}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: "4px",
                      padding: "8px",
                      backgroundColor: hoveredLogit === candidate.index ? "rgba(255,255,255,0.03)" : "var(--bg-surface)",
                      borderRadius: "8px",
                      border: "1px solid var(--border-color)",
                      opacity: isActive ? 1 : 0.4,
                      transition: "all 0.15s ease"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", alignItems: "center" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span style={{ color: "var(--text-muted)", fontSize: "0.65rem", fontFamily: "monospace" }}>#{candidate.rank}</span>
                        <span style={{ fontWeight: "700", color: "var(--text-primary)", fontFamily: "monospace" }}>"{candidate.word}"</span>
                        {candidate.isValidPath && (
                          <span style={{ fontSize: "0.55rem", padding: "1px 4px", backgroundColor: "rgba(16, 185, 129, 0.15)", border: "1px solid var(--success-color)", color: "var(--success-color)", borderRadius: "4px", fontWeight: "700" }}>
                            {candidate.isCorrect ? "BEST PATH" : "ALT PATH"}
                          </span>
                        )}
                      </div>

                      {/* Status Tag */}
                      <div>
                        {isActive && (
                          <span style={{ fontSize: "0.55rem", padding: "1px 4px", backgroundColor: "rgba(34, 211, 238, 0.15)", border: "1px solid var(--query-color)", color: "var(--query-color)", borderRadius: "4px", fontWeight: "700" }}>
                            ACTIVE
                          </span>
                        )}
                        {isMaskedK && (
                          <span style={{ fontSize: "0.55rem", padding: "1px 4px", backgroundColor: "rgba(239, 68, 68, 0.15)", border: "1px solid rgb(239, 68, 68)", color: "rgb(239, 68, 68)", borderRadius: "4px", fontWeight: "700" }}>
                            MASKED K
                          </span>
                        )}
                        {isMaskedP && (
                          <span style={{ fontSize: "0.55rem", padding: "1px 4px", backgroundColor: "rgba(245, 158, 11, 0.15)", border: "1px solid var(--key-color)", color: "var(--key-color)", borderRadius: "4px", fontWeight: "700" }}>
                            MASKED P
                          </span>
                        )}
                      </div>
                    </div>

                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                      <span>Raw: {candidate.rawLogit} (Scaled: {candidate.scaledLogit.toFixed(2)})</span>
                      {isActive && <span>Cum: {(candidate.cumProb * 100).toFixed(0)}%</span>}
                    </div>

                    {/* Progress bar */}
                    {isActive && (
                      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "2px" }}>
                        <div style={{ flex: 1, height: "6px", backgroundColor: "rgba(255,255,255,0.05)", borderRadius: "3px", overflow: "hidden" }}>
                          <div
                            style={{
                              width: `${percent}%`,
                              height: "100%",
                              backgroundColor: candidate.isValidPath ? "var(--success-color)" : "var(--value-color)",
                              boxShadow: candidate.isValidPath ? "0 0 6px var(--success-glow)" : "0 0 6px var(--value-glow)",
                              transition: "width 0.2s ease"
                            }}
                          />
                        </div>
                        <span style={{ fontSize: "0.75rem", fontFamily: "monospace", fontWeight: "700", color: candidate.isValidPath ? "var(--success-color)" : "white", width: "40px", textAlign: "right" }}>
                          {percent}%
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
