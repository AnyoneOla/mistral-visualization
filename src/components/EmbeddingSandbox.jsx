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

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--query-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Brain className="w-6 h-6 text-indigo-400 animate-pulse-slow" />
          Step 1: Input & Token Embedding Lookup
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Before an LLM can perform mathematical equations, text is split into subwords called <strong>tokens</strong>. 
          Each token has a unique vocabulary index. We look up this index in the **Embedding Weights Matrix ({"$W_{embed}$"})** to retrieve the dense vector representing that token.
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
      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: "24px", alignItems: "start" }}>
        
        {/* Left Panel: Vocabulary & Search */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "0.85rem", fontWeight: "700", color: "var(--query-color)" }}>Vocabulary Index</span>
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>({vocab.length} items)</span>
          </div>

          {/* Search box */}
          <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
            <Search className="w-3.5 h-3.5 text-text-muted" style={{ position: "absolute", left: "10px", pointerEvents: "none" }} />
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
                color: "white",
                outline: "none"
              }}
            />
          </div>

          {/* Vocab row list */}
          <div ref={containerRef} style={{ display: "flex", flexDirection: "column", gap: "4px", maxHeight: "300px", overflowY: "auto", paddingRight: "4px" }}>
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
                    <span style={{ fontWeight: isActive ? "700" : "500", color: isActive ? "white" : "var(--text-secondary)" }}>"{word}"</span>
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
              <div style={{ textAlign: "center", padding: "20px", color: "var(--text-muted)", fontSize: "0.75rem" }}>
                No tokens found.
              </div>
            )}
          </div>
        </div>

        {/* Right Panel: One-hot Vector * Weights Matrix = Output Vector */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "white" }}>
            Lookup Multiplier Engine
          </h3>
          
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

      </div>

    </div>
  );
}
