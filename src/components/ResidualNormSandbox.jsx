import React from "react";
import { Cpu, ArrowDown, Info } from "lucide-react";

export default function ResidualNormSandbox({ simData }) {
  const x_embed = simData.embeddingLookup.outputVector;
  const x_attn_out = simData.gqa.attentionOutputProj;
  const { rms, normalized, gamma, output } = simData.rmsnorm2;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--accent-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Cpu className="w-6 h-6 text-emerald-400 animate-pulse-slow" />
          Step 8: Skip Connection (Residual 1) & RMSNorm 2
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          To prevent gradients from vanishing or exploding in deep transformer networks, a **Residual Skip Connection** is applied: the original input embedding vector {"$x_{embed}$"} is added directly to the attention output vector {"$x_{attn_out}$"}.
          This sum is then normalized by a second **RMSNorm** layer before entering the feedforward SwiGLU MLP network.
        </p>
      </div>

      {/* Main Layout Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "24px", alignItems: "start" }}>
        
        {/* Left Panel: Skip Connection Visual Math */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "var(--query-color)" }}>
            Residual Addition
          </h3>
          
          <div style={{ display: "flex", flexDirection: "column", gap: "12px", backgroundColor: "rgba(0,0,0,0.15)", padding: "16px", borderRadius: "10px", border: "1px solid var(--border-color)", fontFamily: "monospace", fontSize: "0.75rem" }}>
            
            {/* Vector 1: Embedding Input */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ color: "var(--text-muted)", fontWeight: "600" }}>Embedding Input vector (x_embed):</span>
              <div style={{ display: "flex", gap: "3px" }}>
                {x_embed.map((val, idx) => (
                  <div key={`resin-x-${idx}`} style={{ flex: 1, padding: "8px 0", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "6px", color: "var(--text-secondary)" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", fontSize: "1.2rem", fontWeight: "700", color: "var(--text-muted)", margin: "-4px 0" }}>+</div>

            {/* Vector 2: Attention Projection Output */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ color: "var(--text-muted)", fontWeight: "600" }}>Attention Out (x_attn_out):</span>
              <div style={{ display: "flex", gap: "3px" }}>
                {x_attn_out.map((val, idx) => (
                  <div key={`resin-ao-${idx}`} style={{ flex: 1, padding: "8px 0", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "6px", color: "var(--text-secondary)" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", fontSize: "1.2rem", fontWeight: "700", color: "var(--text-muted)", margin: "-4px 0" }}>=</div>

            {/* Vector 3: Sum */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ color: "var(--accent-color)", fontWeight: "700" }}>Residual Hidden State Vector:</span>
              <div style={{ display: "flex", gap: "3px" }}>
                {gamma.map((_, idx) => {
                  const sumVal = parseFloat((x_embed[idx] + x_attn_out[idx]).toFixed(3));
                  return (
                    <div key={`resin-sum-${idx}`} style={{ flex: 1, padding: "8px 0", textAlign: "center", backgroundColor: "var(--accent-glow)", border: "1px solid var(--accent-color)", color: "white", borderRadius: "6px", fontWeight: "700", boxShadow: "0 0 6px var(--accent-glow)" }}>
                      {sumVal.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        </div>

        {/* Right Panel: RMSNorm 2 Step-by-Step */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "Outfit", color: "var(--success-color)" }}>
            RMSNorm 2 Block
          </h3>
          
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
                  <div key={`resnorm2-n-${idx}`} style={{ flex: 1, padding: "6px 0", textAlign: "center", backgroundColor: "var(--query-glow)", border: "1px solid var(--query-color)", borderRadius: "4px", color: "white" }}>
                    {val.toFixed(2)}
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
                  <div key={`resnorm2-g-${idx}`} style={{ flex: 1, padding: "6px 0", textAlign: "center", backgroundColor: "var(--key-glow)", border: "1px solid var(--key-color)", borderRadius: "4px", color: "white" }}>
                    {val.toFixed(2)}
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
                  <div key={`resnorm2-out-${idx}`} style={{ flex: 1, padding: "8px 0", textAlign: "center", backgroundColor: "var(--success-glow)", border: "1px solid var(--success-color)", borderRadius: "6px", color: "white", fontWeight: "700", boxShadow: "0 0 6px var(--success-glow)" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>

      </div>

    </div>
  );
}
