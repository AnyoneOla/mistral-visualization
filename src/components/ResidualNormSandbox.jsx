import React, { useState } from "react";
import { Cpu, ArrowDown, Info, GitFork } from "lucide-react";

export default function ResidualNormSandbox({ simData }) {
  const [activeTab, setActiveTab] = useState("viz"); // "viz" or "math"

  const x_embed = simData.embeddingLookup.outputVector;
  const x_attn_out = simData.gqa.attentionOutputProj;
  const { rms, normalized, gamma, output } = simData.rmsnorm2;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--accent-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Cpu className="w-6 h-6 text-emerald-400 animate-pulse-slow" />
          Step 8: Skip Connection (Residual 1) & RMSNorm 2
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          To prevent gradients from vanishing or exploding in deep transformer networks, a <strong>Residual Skip Connection</strong> is applied: the original input embedding vector x<sub>embed</sub> is added directly to the attention output vector x<sub>attn_out</sub>.
          This sum is then normalized by a second <strong>RMSNorm</strong> layer before entering the feedforward SwiGLU MLP network.
        </p>
      </div>

      {/* Tab Panel */}
      <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "16px", minHeight: "390px" }}>
        
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <GitFork className="w-4 h-4 text-emerald-400 animate-pulse" />
            <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--text-primary)", margin: 0 }}>
              Residual Connection & Normalization
            </h3>
          </div>

          <div className="tab-container">
            <button 
              onClick={() => setActiveTab("viz")}
              className={`tab-button ${activeTab === "viz" ? "active" : ""}`}
            >
              Residual Stream Diagram
            </button>
            <button 
              onClick={() => setActiveTab("math")}
              className={`tab-button ${activeTab === "math" ? "active" : ""}`}
            >
              Addition Math details
            </button>
          </div>
        </div>

        {/* TAB 1: VISUAL DIAGRAM */}
        {activeTab === "viz" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              The original token input bypasses the attention computation completely, combining at the summing junction (⊕) to maintain gradient flow.
            </div>

            <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "10px", padding: "15px 0" }}>
              <svg width="540" height="200" viewBox="0 0 540 200" style={{ overflow: "visible" }}>
                {/* Lines */}
                <path d="M 120 95 L 145 95 L 145 35 L 175 35" fill="none" stroke="var(--query-color)" strokeWidth="1.5" />
                <path d="M 145 95 L 145 155 L 320 155 L 320 110" fill="none" stroke="var(--accent-color)" strokeWidth="1.5" strokeDasharray="3 3" />
                <path d="M 285 35 L 320 35 L 320 80" fill="none" stroke="var(--query-color)" strokeWidth="1.5" />
                <path d="M 335 95 L 365 95" stroke="var(--success-color)" strokeWidth="1.5" />
                <path d="M 455 95 L 490 95" stroke="var(--success-color)" strokeWidth="1.5" />

                {/* Particles (Drawn before boxes to hide under them) */}
                <circle r="3.5" fill="var(--query-color)">
                  <animate attributeName="cx" values="120;145;145;320;320;490" keyTimes="0;0.05;0.17;0.52;0.64;1" dur="3s" repeatCount="indefinite" />
                  <animate attributeName="cy" values="95;95;35;35;95;95" keyTimes="0;0.05;0.17;0.52;0.64;1" dur="3s" repeatCount="indefinite" />
                </circle>
                <circle r="3.5" fill="var(--accent-color)">
                  <animate attributeName="cx" values="120;145;145;320;320;490" keyTimes="0;0.05;0.17;0.52;0.64;1" dur="3s" repeatCount="indefinite" />
                  <animate attributeName="cy" values="95;95;155;155;95;95" keyTimes="0;0.05;0.17;0.52;0.64;1" dur="3s" repeatCount="indefinite" />
                </circle>

                {/* 1. Input Box */}
                <g transform="translate(15, 75)">
                  <rect x="0" y="0" width="105" height="40" rx="6" fill="var(--bg-surface)" stroke="var(--query-color)" strokeWidth="1.5" />
                  <text x="52.5" y="24" fill="var(--text-primary)" fontSize="0.7rem" textAnchor="middle" fontWeight="bold">Input x_embed</text>
                </g>

                {/* 2. Attention block */}
                <g transform="translate(175, 15)">
                  <rect x="0" y="0" width="110" height="40" rx="6" fill="var(--bg-card)" stroke="var(--border-color)" strokeWidth="1" />
                  <text x="55" y="24" fill="var(--text-secondary)" fontSize="0.65rem" textAnchor="middle">Attention Block</text>
                </g>

                {/* 3. Summing junction ⊕ */}
                <g transform="translate(305, 80)">
                  <circle cx="15" cy="15" r="15" fill="var(--bg-surface)" stroke="var(--success-color)" strokeWidth="2" />
                  <text x="15" y="20" fill="var(--success-color)" fontSize="1.1rem" textAnchor="middle" fontWeight="bold">+</text>
                </g>

                {/* 4. RMSNorm 2 block */}
                <g transform="translate(365, 75)">
                  <rect x="0" y="0" width="90" height="40" rx="6" fill="var(--bg-surface)" stroke="var(--success-color)" strokeWidth="1.5" />
                  <text x="45" y="24" fill="var(--text-primary)" fontSize="0.7rem" textAnchor="middle" fontWeight="bold">RMSNorm 2</text>
                </g>

                {/* 5. Output Vector */}
                <g transform="translate(490, 80)">
                  <circle cx="15" cy="15" r="15" fill="var(--success-glow)" stroke="var(--success-color)" strokeWidth="2" />
                  <text x="15" y="19" fill="var(--text-primary)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Out</text>
                </g>
              </svg>
            </div>
          </div>
        )}

        {/* TAB 2: STEP-BY-STEP MATH */}
        {activeTab === "math" && (
          <div className="responsive-grid">
            
            {/* Left Column: Skip Addition */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <h4 style={{ fontSize: "0.9rem", color: "var(--query-color)", margin: 0 }}>Residual Addition</h4>
              
              <div style={{ display: "flex", flexDirection: "column", gap: "12px", backgroundColor: "var(--bg-surface)", padding: "16px", borderRadius: "10px", border: "1px solid var(--border-color)", fontFamily: "monospace", fontSize: "0.75rem" }}>
                
                {/* Vector 1: Embedding Input */}
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span style={{ color: "var(--text-muted)", fontWeight: "600" }}>Embedding Input vector (x_embed):</span>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {x_embed.map((val, idx) => (
                      <div key={`resin-x-${idx}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.65rem", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "6px", color: "var(--text-secondary)" }}>
                        {val.toFixed(2)}
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "center", fontSize: "1.1rem", fontWeight: "700", color: "var(--text-muted)", margin: "-4px 0" }}>+</div>

                {/* Vector 2: Attention Projection Output */}
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span style={{ color: "var(--text-muted)", fontWeight: "600" }}>Attention Out (x_attn_out):</span>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {x_attn_out.map((val, idx) => (
                      <div key={`resin-ao-${idx}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.65rem", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "6px", color: "var(--text-secondary)" }}>
                        {val.toFixed(2)}
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "center", fontSize: "1.1rem", fontWeight: "700", color: "var(--text-muted)", margin: "-4px 0" }}>=</div>

                {/* Vector 3: Sum */}
                <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                  <span style={{ color: "var(--accent-color)", fontWeight: "700" }}>Residual Hidden State Vector:</span>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {gamma.map((_, idx) => {
                      const sumVal = parseFloat((x_embed[idx] + x_attn_out[idx]).toFixed(3));
                      return (
                        <div key={`resin-sum-${idx}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.65rem", textAlign: "center", backgroundColor: "var(--accent-glow)", border: "1px solid var(--accent-color)", color: "var(--text-primary)", borderRadius: "6px", fontWeight: "700" }}>
                          {sumVal.toFixed(2)}
                        </div>
                      );
                    })}
                  </div>
                </div>

              </div>
            </div>

            {/* Right Column: RMSNorm 2 */}
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <h4 style={{ fontSize: "0.9rem", color: "var(--success-color)", margin: 0 }}>RMSNorm 2 Block</h4>
              
              <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontFamily: "monospace", fontSize: "0.75rem" }}>
                
                {/* RMS calculations */}
                <div style={{
                  backgroundColor: "var(--bg-surface)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "8px",
                  padding: "12px 14px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "4px"
                }}>
                  <span style={{ color: "var(--text-muted)" }}>RMS calculation:</span>
                  <div style={{ fontSize: "0.75rem", fontFamily: "monospace", color: "var(--text-secondary)" }}>
                    RMS(x) = &radic;<span style={{ borderTop: "1px solid var(--text-secondary)" }}> (1/d) &Sigma; x<sub>i</sub><sup>2</sup> + &epsilon; </span>
                  </div>
                  <div style={{ fontSize: "0.75rem", fontFamily: "monospace", color: "var(--query-color)", overflowX: "auto", whiteSpace: "nowrap", paddingBottom: "4px" }}>
                    RMS = &radic;<span style={{ borderTop: "1px solid var(--query-color)" }}> 1/8 ({gamma.map((_, idx) => ((x_embed[idx] + x_attn_out[idx]) ** 2).toFixed(2)).join(" + ")}) + 1e-5 </span> = <strong>{rms.toFixed(4)}</strong>
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "center", margin: "2px 0" }}>
                  <ArrowDown className="w-4 h-4 text-text-muted" />
                </div>

                {/* Normalized x / RMS */}
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Normalized state:</span>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {normalized.map((val, idx) => (
                      <div key={`resnorm2-n-${idx}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.65rem", textAlign: "center", backgroundColor: "var(--query-glow)", border: "1px solid var(--query-color)", borderRadius: "4px", color: "var(--text-primary)" }}>
                        {val.toFixed(1)}
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "center", fontSize: "1.1rem", color: "var(--text-muted)", margin: "2px 0" }}>&times;</div>

                {/* Scale parameter gamma */}
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <span style={{ color: "var(--text-muted)" }}>Gamma scaling weights (&gamma;):</span>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {gamma.map((val, idx) => (
                      <div key={`resnorm2-g-${idx}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.65rem", textAlign: "center", backgroundColor: "var(--key-glow)", border: "1px solid var(--key-color)", borderRadius: "4px", color: "var(--text-primary)" }}>
                        {val.toFixed(1)}
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ display: "flex", justifyContent: "center", fontSize: "1.1rem", color: "var(--text-muted)", margin: "2px 0" }}>=</div>

                {/* Normalizer output */}
                <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  <span style={{ color: "var(--success-color)", fontWeight: "700" }}>RMSNorm 2 Output:</span>
                  <div style={{ display: "flex", gap: "3px" }}>
                    {output.map((val, idx) => (
                      <div key={`resnorm2-out-${idx}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.65rem", textAlign: "center", backgroundColor: "var(--success-glow)", border: "1px solid var(--success-color)", borderRadius: "6px", color: "var(--text-primary)", fontWeight: "700" }}>
                        {val.toFixed(1)}
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </div>

          </div>
        )}

      </div>

    </div>
  );
}
