import React, { useState, useMemo } from "react";
import { Sparkles, Flame, Sliders, Info, Zap } from "lucide-react";
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
      inTopK: idx < topK,
      rank: idx + 1
    }));
    
    // Mark Top-P
    const topKItems = kMapped.filter(item => item.inTopK);
    const maxLogit = Math.max(...topKItems.map(c => c.scaledLogit));
    const exps = topKItems.map(c => Math.exp(c.scaledLogit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExps);
    
    let cumulative = 0;
    const finalItems = kMapped.map(item => {
      if (!item.inTopK) {
        return {
          ...item,
          inTopP: false,
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
  }, [vocab, rawLogits, correctIndex, temperature, topK, topP]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--attention-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse-slow" />
          Step 10: Unembedding (LM Head) & Softmax Sampler
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          At the final layer, the hidden state vector $h$ (1 &times; 8) is multiplied by the transposed unembedding weights matrix **{"$W_{unembed}$"}** (LM Head) to yield raw scores called **Logits**.
          These logits are scaled by **Temperature** ($T$), filtered using **Top-K** and **Top-P (Nucleus)** algorithms, and normalise through a **Softmax** probability distribution to sample the next token.
        </p>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "24px", alignItems: "start" }}>
        
        {/* Left: LM Head Matrix Multiplication */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "var(--query-color)" }}>
            LM Head Unembedding Matrix Multiply
          </h3>

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

        {/* Right: Softmax Sampler Panel */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "var(--attention-color)", display: "flex", alignItems: "center", gap: "6px" }}>
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

            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
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
                        <span style={{ fontWeight: "700", color: "white", fontFamily: "monospace" }}>"{candidate.word}"</span>
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
