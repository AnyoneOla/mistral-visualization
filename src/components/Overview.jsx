import React, { useState } from "react";
import { 
  Brain, 
  Cpu, 
  Disc, 
  Key, 
  Network, 
  RefreshCw, 
  Sparkles,
  Zap,
  Info
} from "lucide-react";
import { PRESETS } from "../utils/mistralSim";

export default function Overview({ 
  presetIdx, 
  simData, 
  setActiveTab 
}) {
  const [hoveredNode, setHoveredNode] = useState(null);

  const nodes = [
    { id: "embedding", label: "Embed Lookup", x: 50, y: 110, desc: "Token text is looked up in the embedding matrix W_embed to yield an 8-dim vector.", icon: <Brain className="w-4 h-4 text-indigo-700" /> },
    { id: "rmsnorm1", label: "RMSNorm 1", x: 130, y: 110, desc: "Root Mean Square Normalization prepares activations for self-attention.", icon: <Cpu className="w-4 h-4 text-teal-700" /> },
    
    // Attention group (raised y position)
    { id: "gqa", label: "QKV Proj (GQA)", x: 220, y: 50, desc: "Projects vector to Query, Key, and Value states with GQA (4:1 Query to KV head groups).", icon: <Network className="w-4 h-4 text-amber-700" /> },
    { id: "rope", label: "RoPE", x: 300, y: 50, desc: "Rotary Position Embeddings rotate Q and K vectors in 2D planes based on position index.", icon: <RefreshCw className="w-4 h-4 text-sky-700" /> },
    { id: "swa", label: "SWA Mask", x: 380, y: 50, desc: "Sliding Window Attention limits attention window range to W recent tokens.", icon: <Cpu className="w-4 h-4 text-pink-700" /> },
    { id: "cache", label: "Rolling KV Cache", x: 460, y: 50, desc: "Writes Key/Value states into circular buffer slot (position modulo window size).", icon: <Disc className="w-4 h-4 text-purple-700" /> },
    { id: "oproj", label: "O-Proj (Merge)", x: 540, y: 50, desc: "Concatenates all head values and projects back to hidden dimension via W_O.", icon: <Network className="w-4 h-4 text-orange-700" /> },
    
    { id: "resnorm2", label: "Add & RMSNorm 2", x: 630, y: 110, desc: "Sums attention output with skip connection (x_embed) and normalizes for MLP block.", icon: <Cpu className="w-4 h-4 text-emerald-700" /> },
    
    // MLP group (lowered y position)
    { id: "swiglu", label: "SwiGLU MLP", x: 710, y: 170, desc: "Feed-forward network with SiLU gate activation, expanding dimension to 12.", icon: <Key className="w-4 h-4 text-cyan-700" /> },
    
    { id: "logits", label: "LM Head", x: 800, y: 110, desc: "Unembedding projection maps the final vector back to raw scores for 8 vocabulary words.", icon: <Sparkles className="w-4 h-4 text-yellow-700" /> }
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Preset Details Banner */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--query-color)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Zap className="w-4 h-4 text-cyan-700 animate-pulse" />
          <span style={{ fontSize: "0.65rem", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-secondary)" }}>
            Current Simulation Prompt
          </span>
        </div>
        <h2 style={{ fontSize: "1.3rem", fontFamily: "inherit", color: "var(--text-primary)", fontWeight: "800", letterSpacing: "-0.02em", margin: 0 }}>
          {PRESETS[presetIdx].title}
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", lineHeight: "1.5", margin: 0 }}>
          {PRESETS[presetIdx].description}
        </p>
      </div>

      {/* Interactive Architecture Flow Graph */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px", padding: "24px 20px", position: "relative" }}>
        
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
          <h3 style={{ fontSize: "1rem", fontFamily: "inherit", color: "var(--text-primary)", margin: 0 }}>
            Mistral 7B Layer Architecture Pipeline
          </h3>
          <span style={{ fontSize: "0.65rem", color: "var(--text-secondary)", fontWeight: "700", textTransform: "uppercase", letterSpacing: "0.04em", fontFamily: "monospace" }}>
            Click node to scroll to details
          </span>
        </div>

        <div style={{ overflowX: "auto", padding: "10px 0", position: "relative" }}>
          <svg width="860" height="230" viewBox="0 0 860 230" style={{ overflow: "visible", display: "block", margin: "0 auto" }}>
            <defs>
              <linearGradient id="grad-active-stream" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="var(--query-color)" />
                <stop offset="50%" stopColor="var(--attention-color)" />
                <stop offset="100%" stopColor="var(--success-color)" />
              </linearGradient>
            </defs>

            {/* Pipeline Connections (Base Lines) */}
            {/* Main trunk */}
            <path d="M 50 110 L 130 110" stroke="var(--border-color)" strokeWidth="2" />
            <path d="M 130 110 L 220 50 L 540 50 L 630 110" stroke="var(--border-color)" strokeWidth="2" fill="none" />
            <path d="M 630 110 L 710 170 L 800 110" stroke="var(--border-color)" strokeWidth="2" fill="none" />

            {/* Skip Connection 1 (Attention block bypass) */}
            <path d="M 130 110 Q 380 150, 630 110" fill="none" stroke="var(--border-color)" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.6" />

            {/* Skip Connection 2 (MLP block bypass) */}
            <path d="M 630 110 Q 715 70, 800 110" fill="none" stroke="var(--border-color)" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.6" />

            {/* Animated Pulses running through streams */}
            {/* Trunk stream */}
            <path d="M 50 110 L 130 110 L 220 50 L 540 50 L 630 110 L 710 170 L 800 110" fill="none" stroke="url(#grad-active-stream)" strokeWidth="1.5" strokeDasharray="6 6">
              <animate attributeName="strokeDashoffset" values="40;0" dur="2s" repeatCount="indefinite" />
            </path>

            {/* Skip 1 Stream */}
            <path d="M 130 110 Q 380 150, 630 110" fill="none" stroke="var(--query-color)" strokeWidth="1" strokeDasharray="4 4" opacity="0.3">
              <animate attributeName="strokeDashoffset" values="30;0" dur="2.5s" repeatCount="indefinite" />
            </path>

            {/* Skip 2 Stream */}
            <path d="M 630 110 Q 715 70, 800 110" fill="none" stroke="var(--success-color)" strokeWidth="1" strokeDasharray="4 4" opacity="0.3">
              <animate attributeName="strokeDashoffset" values="20;0" dur="2.2s" repeatCount="indefinite" />
            </path>

            {/* Node markers */}
            {nodes.map((node, idx) => {
              const isHovered = hoveredNode === node.id;
              return (
                <g 
                  key={node.id} 
                  transform={`translate(${node.x}, ${node.y})`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  onClick={() => setActiveTab(node.id)}
                >
                  {/* Outer glow ring on hover */}
                  <circle cx="0" cy="0" r={isHovered ? "20" : "15"} fill="rgba(0,0,0,0.01)" stroke={isHovered ? "var(--query-color)" : "var(--border-color)"} strokeWidth={isHovered ? "2" : "1.2"} style={{ transition: "all 0.2s cubic-bezier(0.22, 1, 0.36, 1)" }} />
                  
                  {/* Inner node color */}
                  <circle cx="0" cy="0" r="10" fill="var(--bg-deep)" stroke="var(--border-color)" strokeWidth="1.5" />
                  
                  {/* Miniature index badge */}
                  <g transform="translate(0, -22)">
                    <rect x="-10" y="-7" width="20" height="12" fill="var(--bg-card)" stroke="var(--border-color)" strokeWidth="1" />
                    <text x="0" y="2.5" fill="var(--text-primary)" fontSize="0.45rem" textAnchor="middle" fontFamily="monospace" fontWeight="700">#{idx + 1}</text>
                  </g>

                  {/* Icon placement inside node */}
                  <g transform="translate(-8, -8)">
                    {React.cloneElement(node.icon, { width: 16, height: 16, style: { transform: "scale(0.85)", transformOrigin: "8px 8px" } })}
                  </g>

                  {/* Text label underneath */}
                  <text x="0" y="32" fill={isHovered ? "var(--query-color)" : "var(--text-secondary)"} fontSize="0.6rem" textAnchor="middle" fontWeight={isHovered ? "700" : "500"} style={{ transition: "all 0.2s" }}>
                    {node.label}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Detail Tooltip Box below diagram for seamless explanation */}
        <div style={{ 
          minHeight: "75px", 
          backgroundColor: "var(--bg-surface)", 
          borderRadius: "12px", 
          padding: "12px 16px",
          border: "1px solid var(--border-color)",
          display: "flex",
          alignItems: "flex-start",
          gap: "10px"
        }}>
          <Info className="w-4 h-4 text-cyan-700 flex-shrink-0" style={{ marginTop: "2px" }} />
          {hoveredNode ? (
            <div>
              <span style={{ fontSize: "0.85rem", fontWeight: "700", color: "var(--text-primary)" }}>
                {nodes.find(n => n.id === hoveredNode).label}
              </span>
              <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)", margin: "4px 0 0 0", lineHeight: "1.4" }}>
                {nodes.find(n => n.id === hoveredNode).desc}
              </p>
            </div>
          ) : (
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontStyle: "italic" }}>
              💡 Hover over any step node in the Mistral Layer diagram to read about that operation, or click it to scroll directly to the corresponding interactive sandbox.
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
