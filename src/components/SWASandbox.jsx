import React, { useState } from "react";
import { Cpu, Eye, Info } from "lucide-react";

export default function SWASandbox({ simData }) {
  const [windowSize, setWindowSize] = useState(4);
  const [hoveredCell, setHoveredCell] = useState(null); // {qIdx, kIdx}
  const [activeLayerToken, setActiveLayerToken] = useState(null);

  const tokens = simData.allTokens;
  const seqLen = tokens.length;
  
  // Custom attention weights mapped for the sliding window visualizer
  const getAttentionWeight = (qIdx, kIdx) => {
    if (kIdx > qIdx) return 0;
    const isMasked = kIdx < qIdx - windowSize + 1;
    if (isMasked) return 0;

    const headWeights = simData.swa.attentionWeights[0];
    if (headWeights) {
      // Re-normalize scores for the current windowSize (on the fly simulation)
      let rawScores = [];
      let maxScore = -Infinity;
      for (let t = 0; t <= qIdx; t++) {
        const masked = t < qIdx - windowSize + 1;
        const score = masked ? -100 : (simData.gqa.rawScores[0]?.[t] || (0.5 - Math.abs(qIdx - t) * 0.15));
        rawScores.push(score);
        if (score > maxScore) maxScore = score;
      }
      
      let sumExp = 0;
      const exps = [];
      for (let t = 0; t <= qIdx; t++) {
        if (rawScores[t] <= -99) {
          exps.push(0);
        } else {
          const val = Math.exp(rawScores[t] - maxScore);
          exps.push(val);
          sumExp += val;
        }
      }
      return sumExp > 0 ? parseFloat((exps[kIdx] / sumExp).toFixed(3)) : 0;
    }

    return 0.1;
  };

  // Get raw attention score for the cell inspector
  const getRawScore = (qIdx, kIdx) => {
    const rawVal = simData.gqa.rawScores[0]?.[kIdx] || (0.5 - Math.abs(qIdx - kIdx) * 0.15);
    return parseFloat(rawVal.toFixed(3));
  };

  // Receptive field propagation helper
  const getReceptiveField = (targetIdx, layerDepth) => {
    const visible = new Set([targetIdx]);
    
    for (let l = 0; l < layerDepth; l++) {
      const currentSnapshot = Array.from(visible);
      currentSnapshot.forEach(idx => {
        const minVal = Math.max(0, idx - windowSize + 1);
        for (let i = minVal; i <= idx; i++) {
          visible.add(i);
        }
      });
    }
    return Array.from(visible).sort((a, b) => a - b);
  };

  const targetTokenIdxForRF = activeLayerToken !== null ? activeLayerToken : Math.min(seqLen - 1, 7);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--attention-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Cpu className="w-6 h-6 text-pink-400 animate-pulse-slow" />
          Step 5: Sliding Window Self-Attention (SWA)
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Standard self-attention computes dot products for all tokens, scaling quadratically: $O(T^2)$. 
          Mistral limits this using a **Sliding Window of size W** (e.g., W = 4096). A token at position $i$ only attends to positions $[i - W + 1, i]$. 
          This results in a band-diagonal pattern in the attention matrix, saving computation.
        </p>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "24px", alignItems: "start" }}>
        
        {/* Left: Attention Matrix Grid */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
            <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "white" }}>
              Attention Weights Grid
            </h3>
            
            {/* Slider to adjust W */}
            <div style={{ display: "flex", alignItems: "center", gap: "10px", fontSize: "0.85rem" }}>
              <span style={{ color: "var(--text-secondary)" }}>Window W:</span>
              <input
                type="range"
                min="2"
                max="8"
                value={windowSize}
                onChange={(e) => setWindowSize(parseInt(e.target.value))}
                className="custom-slider"
                style={{ width: "90px" }}
              />
              <span style={{ fontWeight: "700", color: "var(--attention-color)" }}>{windowSize}</span>
            </div>
          </div>

          {/* Matrix canvas */}
          <div style={{ overflowX: "auto" }}>
            <div style={{ 
              minWidth: "360px", 
              padding: "16px", 
              backgroundColor: "var(--bg-surface)", 
              borderRadius: "12px", 
              border: "1px solid var(--border-color)"
            }}>
              
              <div style={{
                display: "grid",
                gridTemplateColumns: `70px repeat(${seqLen}, 1fr)`,
                gap: "3px"
              }}>
                {/* Empty corner */}
                <div></div>
                {/* Column Tokens (Keys) */}
                {tokens.map((tok, idx) => (
                  <div 
                    key={`swa-col-${idx}`} 
                    style={{ 
                      fontSize: "0.65rem", 
                      color: hoveredCell?.kIdx === idx ? "var(--key-color)" : "var(--text-muted)", 
                      textAlign: "center", 
                      textOverflow: "ellipsis", 
                      overflow: "hidden", 
                      whiteSpace: "nowrap",
                      fontWeight: hoveredCell?.kIdx === idx ? "700" : "500"
                    }}
                  >
                    {tok.text}
                  </div>
                ))}

                {/* Rows */}
                {tokens.map((rowTok, qIdx) => (
                  <React.Fragment key={`swa-row-${qIdx}`}>
                    {/* Row Token (Query) */}
                    <div style={{ 
                      fontSize: "0.65rem", 
                      color: hoveredCell?.qIdx === qIdx ? "var(--query-color)" : "var(--text-muted)", 
                      display: "flex", 
                      alignItems: "center", 
                      justifyContent: "flex-end", 
                      paddingRight: "6px", 
                      overflow: "hidden", 
                      textOverflow: "ellipsis", 
                      whiteSpace: "nowrap",
                      fontWeight: hoveredCell?.qIdx === qIdx ? "700" : "500"
                    }}>
                      {rowTok.text}
                    </div>

                    {/* Cells */}
                    {tokens.map((colTok, kIdx) => {
                      const isCausalMask = kIdx > qIdx;
                      const isSWAMask = kIdx < qIdx - windowSize + 1;
                      const isMasked = isCausalMask || isSWAMask;
                      const weight = getAttentionWeight(qIdx, kIdx);
                      
                      const cellColor = isMasked 
                        ? "transparent" 
                        : `hsla(322, 90%, 60%, ${0.05 + weight * 0.95})`;

                      return (
                        <div
                          key={`swa-cell-${qIdx}-${kIdx}`}
                          className={`matrix-cell ${isMasked ? "masked" : ""}`}
                          style={{
                            backgroundColor: cellColor,
                            border: isMasked ? "1px solid rgba(255,255,255,0.01)" : "1px solid rgba(255,255,255,0.07)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: "0.6rem",
                            color: "white",
                            fontWeight: "700",
                            aspectRatio: "1"
                          }}
                          onMouseEnter={() => setHoveredCell({ qIdx, kIdx })}
                          onMouseLeave={() => setHoveredCell(null)}
                        >
                          {!isMasked && weight > 0.05 && weight.toFixed(2)}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>

            </div>
          </div>
        </div>

        {/* Right: Math inspection & depth stack */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          
          {/* Causal and Window Masking Math Inspector */}
          <div className="glass-panel" style={{ padding: "16px" }}>
            <h3 style={{ fontSize: "1.0rem", marginBottom: "12px", fontFamily: "Outfit", color: "var(--attention-color)" }}>
              Attention Score Masking Inspector
            </h3>
            
            {hoveredCell ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.8rem" }}>
                <div>
                  <span style={{ color: "var(--text-secondary)" }}>Query Token (i = {hoveredCell.qIdx}):</span>{" "}
                  <strong style={{ color: "var(--query-color)" }}>"{tokens[hoveredCell.qIdx]?.text}"</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-secondary)" }}>Key Token (j = {hoveredCell.kIdx}):</span>{" "}
                  <strong style={{ color: "var(--key-color)" }}>"{tokens[hoveredCell.kIdx]?.text}"</strong>
                </div>

                {(() => {
                  const isCausalMask = hoveredCell.kIdx > hoveredCell.qIdx;
                  const isSWAMask = hoveredCell.kIdx < hoveredCell.qIdx - windowSize + 1;
                  const isMasked = isCausalMask || isSWAMask;
                  const rawScore = getRawScore(hoveredCell.qIdx, hoveredCell.kIdx);
                  const maskVal = isCausalMask ? "-\u221E (future mask)" : isSWAMask ? "-\u221E (window limit)" : "0 (active)";
                  const finalInput = isMasked ? "-\u221E" : rawScore;
                  const prob = getAttentionWeight(hoveredCell.qIdx, hoveredCell.kIdx);

                  const isActiveRow = hoveredCell.qIdx === seqLen - 1;
                  const qVec = simData.gqa.qVectors[0] || [0.2, -0.5, 0.8, -0.1];
                  const kVec = simData.gqa.kVectors[hoveredCell.kIdx]?.[0] || [0.3, 0.4, -0.1, 0.2];

                  return (
                    <div style={{ 
                      padding: "10px", 
                      backgroundColor: "var(--bg-surface)", 
                      borderRadius: "8px", 
                      border: "1px solid var(--border-color)",
                      fontFamily: "monospace",
                      fontSize: "0.72rem",
                      display: "flex",
                      flexDirection: "column",
                      gap: "5px"
                    }}>
                      {isActiveRow && (
                        <div style={{ borderBottom: "1px dashed rgba(255,255,255,0.05)", paddingBottom: "6px", marginBottom: "4px" }}>
                          <div style={{ color: "var(--text-muted)", fontSize: "0.65rem", marginBottom: "2px" }}>Q &middot; K Vector dot product:</div>
                          <div style={{ color: "white" }}>
                            Dot = {qVec.map((qv, idx) => `(${qv.toFixed(2)}&times;${kVec[idx].toFixed(2)})`).join("+")} = <strong>{(qVec.reduce((s, qv, idx) => s + qv * kVec[idx], 0)).toFixed(3)}</strong>
                          </div>
                        </div>
                      )}
                      <div>Score (Scaled): <strong>{rawScore}</strong></div>
                      <div>Mask Offset M<sub>ij</sub>: <span style={{ color: isMasked ? "var(--attention-color)" : "var(--success-color)" }}>{maskVal}</span></div>
                      <div>Input to Softmax (Score + M): <strong>{finalInput}</strong></div>
                      <div style={{ fontWeight: "700", color: isMasked ? "var(--text-muted)" : "var(--success-color)", marginTop: "4px" }}>
                        Attention Weight: {(prob * 100).toFixed(1)}%
                      </div>
                    </div>
                  );
                })()}
              </div>
            ) : (
              <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", fontStyle: "italic", margin: 0 }}>
                Hover over grid cells to inspect mathematical masking operations.
              </p>
            )}
          </div>

          {/* Layer Receptive Field Stack */}
          <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "10px" }}>
            <h3 style={{ fontSize: "1.0rem", fontFamily: "Outfit", color: "var(--success-color)" }}>
              Receptive Field Depth
            </h3>
            
            <div style={{ display: "flex", flexWrap: "wrap", gap: "4px", margin: "4px 0" }}>
              {tokens.map((t, idx) => (
                <button
                  key={`rf-btn-${idx}`}
                  onClick={() => setActiveLayerToken(idx)}
                  style={{
                    padding: "2px 6px",
                    borderRadius: "4px",
                    border: "1px solid var(--border-color)",
                    backgroundColor: targetTokenIdxForRF === idx ? "var(--success-color)" : "var(--bg-surface)",
                    color: targetTokenIdxForRF === idx ? "black" : "var(--text-secondary)",
                    fontSize: "0.7rem",
                    cursor: "pointer",
                    fontWeight: "600"
                  }}
                >
                  {t.text}
                </button>
              ))}
            </div>

            <div style={{ 
              display: "flex", 
              flexDirection: "column", 
              gap: "8px", 
              backgroundColor: "var(--bg-surface)", 
              padding: "10px", 
              borderRadius: "8px", 
              border: "1px solid var(--border-color)" 
            }}>
              {[3, 2, 1, 0].map((layerDepth) => {
                const visibleIdxs = getReceptiveField(targetTokenIdxForRF, layerDepth);
                return (
                  <div key={`rf-layer-${layerDepth}`} style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-muted)" }}>
                      <span>Layer {layerDepth}</span>
                      <span>{visibleIdxs.length} visible tokens</span>
                    </div>
                    
                    <div style={{ display: "flex", gap: "2px" }}>
                      {tokens.map((_, tIdx) => {
                        const isVisible = visibleIdxs.includes(tIdx);
                        const isTarget = tIdx === targetTokenIdxForRF;
                        return (
                          <div
                            key={`rf-dot-${layerDepth}-${tIdx}`}
                            style={{
                              flex: 1,
                              height: "10px",
                              borderRadius: "2px",
                              backgroundColor: isTarget 
                                ? "var(--query-color)" 
                                : isVisible 
                                  ? "var(--success-color)" 
                                  : "rgba(255,255,255,0.05)",
                              boxShadow: isTarget 
                                ? "0 0 4px var(--query-color)" 
                                : isVisible 
                                  ? "0 0 2px var(--success-glow)" 
                                  : "none",
                              transition: "all 0.2s ease"
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

            <p style={{ color: "var(--text-muted)", fontSize: "0.7rem", lineHeight: "1.3", margin: 0 }}>
              💡 Destructive stacked sliding windows allow tokens to look deep back. Across 32 stacked layers in Mistral, the effective receptive field grows to **131,072 tokens**!
            </p>
          </div>

        </div>

      </div>

    </div>
  );
}
