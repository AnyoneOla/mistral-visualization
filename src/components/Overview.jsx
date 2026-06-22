import React from "react";
import { 
  Brain, 
  Cpu, 
  Disc, 
  Key, 
  Network, 
  RefreshCw, 
  ArrowRight, 
  Sparkles,
  Zap
} from "lucide-react";
import { PRESETS } from "../utils/mistralSim";

export default function Overview({ 
  presetIdx, 
  simData, 
  setActiveTab 
}) {
  
  // Pipeline block definitions matching all 10 sequential stages
  const pipelineBlocks = [
    {
      id: "embedding",
      title: "1. Token Embedding Lookup",
      description: "Converts active token text into a dense vector using the embedding lookup matrix.",
      icon: <Brain className="w-5 h-5 text-indigo-400" />,
      details: `Active Token: "${simData.activeToken?.text}" → Vector: [${simData.embeddingLookup.outputVector?.slice(0, 4).map(v => v.toFixed(2)).join(", ")}... ]`
    },
    {
      id: "rmsnorm1",
      title: "2. RMSNorm 1 & Dropout",
      description: "Normalizes the activation vector using Root Mean Square, preparing it for attention query/key/value projections.",
      icon: <Cpu className="w-5 h-5 text-teal-400" />,
      details: `RMS: ${simData.rmsnorm1.rms.toFixed(4)} → Normed Output: [${simData.rmsnorm1.output?.slice(0, 3).map(v => v.toFixed(2)).join(", ")}...]`
    },
    {
      id: "gqa",
      title: "3. QKV GQA Projections",
      description: "Projects query, key, and value vectors. Mistral groups query heads to share key-value states.",
      icon: <Network className="w-5 h-5 text-amber-400" />,
      details: `${simData.dimensions.numQHeads} Query heads map to ${simData.dimensions.numKVHeads} KV groups (GQA 4:1 compression)`
    },
    {
      id: "rope",
      title: "4. Rotary Position Embedding (RoPE)",
      description: "Rotates query & key dimensions in 2D planes based on position to inject relative distance information.",
      icon: <RefreshCw className="w-5 h-5 text-sky-400" />,
      details: `Pos: ${simData.activeTokenIdx} → rotQ: [${simData.rope[simData.activeTokenIdx]?.rotQ?.slice(0, 3).map(v => v.toFixed(2)).join(", ")}...]`
    },
    {
      id: "swa",
      title: "5. Sliding Window Attention",
      description: "Limits attention range. The active token only scores attention for the W recent tokens.",
      icon: <Cpu className="w-5 h-5 text-pink-400" />,
      details: `Masks tokens older than W = ${simData.swa.windowSize} steps`
    },
    {
      id: "cache",
      title: "6. Rolling Buffer KV Cache",
      description: "Writes key-value vectors to slot index (position mod W) of circular queue cache.",
      icon: <Disc className="w-5 h-5 text-purple-400" />,
      details: `Overwriting Slot ${simData.rollingCache.activeSlot} with active token key/value states`
    },
    {
      id: "oproj",
      title: "7. Attention Output Projection (O-Proj)",
      description: "Concatenates individual head attention values and projects them back using matrix weights.",
      icon: <Network className="w-5 h-5 text-orange-400" />,
      details: `Proj Out: [${simData.gqa.attentionOutputProj?.slice(0, 4).map(v => v.toFixed(2)).join(", ")}...]`
    },
    {
      id: "resnorm2",
      title: "8. Residual 1 & RMSNorm 2",
      description: "Adds attention output back to embedding input (residual connection) and normalizes for MLP block.",
      icon: <Cpu className="w-5 h-5 text-emerald-400" />,
      details: `Residual added → RMSNorm 2: [${simData.rmsnorm2.output?.slice(0, 3).map(v => v.toFixed(2)).join(", ")}...]`
    },
    {
      id: "swiglu",
      title: "9. SwiGLU MLP Block",
      description: "Gated feed-forward network stream combining SiLU gate activation and up projection.",
      icon: <Key className="w-5 h-5 text-cyan-400" />,
      details: `Formula: (SiLU(xWg) ⊗ xWu) Wd → [${simData.swiglu.ffnOutput?.slice(0, 4).map(v => v.toFixed(2)).join(", ")}...]`
    },
    {
      id: "logits",
      title: "10. LM Head Softmax & Sampling",
      description: "Multiplies final layer hidden vector by vocabulary weights to output next token probability.",
      icon: <Sparkles className="w-5 h-5 text-yellow-400" />,
      details: `Next Token: "${simData.logitsData.correctWord}" with Logit score: ${simData.logitsData.rawLogits[simData.logitsData.correctIndex]?.toFixed(2)}`
    }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
      
      {/* Preset Details Banner */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--query-color)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Zap className="w-5 h-5 text-accent-color animate-pulse-slow" />
          <span style={{ fontSize: "0.7rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-muted)" }}>
            Current Simulation Prompt
          </span>
        </div>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", color: "var(--text-primary)", fontWeight: "800" }}>
          {PRESETS[presetIdx].title}
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          {PRESETS[presetIdx].description}
        </p>
      </div>

      {/* Model Architecture Pipeline Block Grid */}
      <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontSize: "1.3rem", fontFamily: "Outfit", color: "white" }}>
            Mistral 7B Transformer Layer Pipeline Flow
          </h2>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase" }}>
            Click node to inspect details below
          </span>
        </div>
        
        {/* Pipeline container using a flex-col list with SVG flow lines in between */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "stretch", gap: "10px" }}>
          {pipelineBlocks.map((block, idx) => (
            <React.Fragment key={block.id}>
              
              {/* Node Card */}
              <div 
                className="glass-panel pipeline-node"
                style={{
                  display: "grid",
                  gridTemplateColumns: "60px 1.5fr 2.5fr 50px",
                  alignItems: "center",
                  gap: "16px",
                  padding: "16px 24px",
                  cursor: "pointer",
                  backgroundColor: "rgba(10, 15, 30, 0.45)"
                }}
                onClick={() => setActiveTab(block.id)}
              >
                {/* Step Number & Icon */}
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
                  <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace", fontWeight: "700" }}>
                    STEP {idx + 1}
                  </span>
                  <div style={{
                    padding: "8px",
                    borderRadius: "8px",
                    backgroundColor: "var(--bg-surface)",
                    border: "1px solid var(--border-color)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center"
                  }}>
                    {block.icon}
                  </div>
                </div>

                {/* Title & Description */}
                <div>
                  <h3 style={{ fontSize: "1.0rem", fontFamily: "Outfit", color: "white", marginBottom: "4px" }}>{block.title}</h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "0.8rem", lineHeight: "1.3" }}>
                    {block.description}
                  </p>
                </div>

                {/* Live simulation state readout */}
                <div style={{
                  backgroundColor: "hsl(224, 71%, 3%)",
                  borderRadius: "6px",
                  border: "1px solid rgba(255,255,255,0.04)",
                  padding: "10px 14px",
                  fontFamily: "monospace",
                  fontSize: "0.72rem",
                  color: "var(--text-muted)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  display: "flex",
                  alignItems: "center",
                  gap: "10px"
                }}>
                  <div style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    backgroundColor: "var(--accent-color)",
                    boxShadow: "0 0 6px var(--accent-color)"
                  }}></div>
                  <span style={{ color: "var(--text-secondary)" }}>{block.details}</span>
                </div>

                {/* Click action indicator */}
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <div style={{
                    width: "28px",
                    height: "28px",
                    borderRadius: "50%",
                    backgroundColor: "rgba(255,255,255,0.02)",
                    border: "1px solid var(--border-color)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.2s ease"
                  }}>
                    <ArrowRight className="w-3.5 h-3.5 text-accent-color" />
                  </div>
                </div>
              </div>

              {/* Connecting line between nodes (except last) */}
              {idx < pipelineBlocks.length - 1 && (
                <div style={{ display: "flex", justifyContent: "center", margin: "-6px 0", height: "30px" }}>
                  <svg width="40" height="30" style={{ overflow: "visible" }}>
                    <line x1="20" y1="0" x2="20" y2="30" stroke="var(--border-color)" strokeWidth="2" strokeDasharray="4 3" />
                    <circle cx="20" cy="15" r="3.5" fill="var(--accent-color)" style={{ opacity: 0.6 }} />
                  </svg>
                </div>
              )}

            </React.Fragment>
          ))}
        </div>

      </div>

    </div>
  );
}
