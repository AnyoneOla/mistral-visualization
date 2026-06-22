import React, { useState } from "react";
import { Network, ArrowDown, HelpCircle, Layers } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function AttentionOutputSandbox({ simData }) {
  const [selectedHead, setSelectedHead] = useState(0);

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

  // Row labels for Stage 1 left input vector
  const weightRowNames = [`"Head ${selectedHead} weights"`];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--accent-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Network className="w-6 h-6 text-orange-400 animate-pulse-slow" />
          Step 7: Value Weighted Sum & Output Projection (O-Proj)
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          The final self-attention stage aggregates information and projects it back to the hidden dimension:
          <br />
          <strong>Stage 1 (Weighted Values Sum):</strong> The attention weights [1 x T] scale the Value vectors [T x 4] of all prior tokens. Adding these gives the output of each head [1 x 4].
          <br />
          <strong>Stage 2 (Output Projection):</strong> The outputs of all 8 heads are **concatenated** into a single vector of size 32, and multiplied by the projection matrix **$W_O$** (size 32 &times; 8) to yield the final attention state.
        </p>
      </div>

      {/* Select Head Tabs */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
        <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: "600" }}>Inspect Query Head:</span>
        <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-surface)", padding: "3px", borderRadius: "8px", border: "1px solid var(--border-color)", flexWrap: "wrap" }}>
          {Array.from({ length: numHeads }).map((_, hIdx) => (
            <button
              key={`h-btn-${hIdx}`}
              onClick={() => setSelectedHead(hIdx)}
              style={{
                padding: "6px 12px",
                borderRadius: "6px",
                border: "none",
                backgroundColor: selectedHead === hIdx ? "var(--accent-color)" : "transparent",
                color: selectedHead === hIdx ? "white" : "var(--text-secondary)",
                fontSize: "0.8rem",
                fontWeight: "700",
                cursor: "pointer",
                transition: "all 0.15s ease"
              }}
            >
              Head Q{hIdx} (KV{Math.floor(hIdx / ratio)})
            </button>
          ))}
        </div>
      </div>

      {/* Stacked Panels for Stage 1 and Stage 2 */}
      <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
        
        {/* Stage 1: Weighted Sum */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "var(--query-color)", display: "flex", alignItems: "center", gap: "6px" }}>
            <Layers className="w-5 h-5 text-indigo-400" />
            Stage 1: Attention Head Q{selectedHead} Value Aggregation [1 &times; T] &times; [T &times; 4]
          </h3>
          
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

        <div style={{ display: "flex", justifyContent: "center", margin: "-12px 0" }}>
          <ArrowDown className="w-5 h-5 text-text-muted" />
        </div>

        {/* Stage 2: Output Projection */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "var(--success-color)", display: "flex", alignItems: "center", gap: "6px" }}>
            <Network className="w-5 h-5 text-orange-400" />
            Stage 2: Concatenate & Output Projection W_O [1 &times; 32] &times; [32 &times; 8]
          </h3>

          <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", margin: 0 }}>
            💡 All 8 head outputs (8 &times; 4 = 32 values) are concatenated. The active head Q{selectedHead} occupies slots <strong>[{selectedHead * 4} to {selectedHead * 4 + 3}]</strong> of this large vector.
          </p>
          
          <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
            <InteractiveMatrixMul
              leftVal={concatenatedHeads}
              rightVal={W_O}
              outputVal={attentionOutputProj}
              leftLabel="Concatenated Attention Head Outputs [1 x 32]"
              rightLabel="Projection Weight Matrix W_O [32 x 8]"
              outputLabel="Projected Hidden Vector [1 x 8]"
              leftColor="var(--accent-color)"
              rightColor="var(--key-color)"
              outputColor="var(--success-color)"
              leftRowNames={["Concatenated"]}
              rightColNames={Array.from({ length: 8 }, (_, i) => `d${i}`)}
            />
          </div>
        </div>

      </div>

    </div>
  );
}
