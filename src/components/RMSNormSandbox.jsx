import React, { useState, useMemo } from "react";
import { Cpu, ToggleLeft, ToggleRight, Sparkles, HelpCircle, ArrowDown } from "lucide-react";

export default function RMSNormSandbox({ simData }) {
  const { rms, normalized, gamma, output } = simData.rmsnorm1;

  // Dropout state
  const [isTraining, setIsTraining] = useState(false);
  const [dropoutRate, setDropoutRate] = useState(0.25);
  const [dropoutSeed, setDropoutSeed] = useState(1);

  // Compute dropout mask and scaled values
  const dropoutData = useMemo(() => {
    const scale = 1 / (1 - dropoutRate);
    const mask = [];
    const values = [];
    
    // Seeded mask generation for consistency
    let seed = dropoutSeed;
    for (let i = 0; i < 8; i++) {
      const rand = (Math.sin(seed++) * 10000) - Math.floor(Math.sin(seed) * 10000);
      const dropped = Math.abs(rand) < dropoutRate;
      mask.push(dropped ? 0 : 1);
      values.push(dropped ? 0.0 : parseFloat((output[i] * scale).toFixed(3)));
    }
    
    return { mask, values, scale };
  }, [dropoutRate, output, dropoutSeed]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--accent-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Cpu className="w-6 h-6 text-teal-400 animate-pulse-slow" />
          Step 2: RMSNorm 1 & Dropout
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Mistral 7B normalizes activations prior to layers using <strong>RMSNorm (Root Mean Square Normalization)</strong>. 
          Unlike standard LayerNorm which calculates both mean and variance, RMSNorm assumes a zero-mean activation vector and scales solely by the root-mean-square. 
          This saves computational time. During training, a <strong>Dropout</strong> mask is optionally applied to regularize the network.
        </p>
      </div>

      {/* Main Layout Grid */}
      <div className="responsive-grid">
        
        {/* Left Panel: RMSNorm Math Flow Chart */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--query-color)" }}>
            RMSNorm Block Math Diagram
          </h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "12px", alignItems: "stretch" }}>
            
            {/* Step 1: Input Vector */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase" }}>
                1. Input Activation x [1 &times; 8]
              </span>
              <div style={{ display: "flex", gap: "4px" }}>
                {simData.embeddingLookup.outputVector.map((val, i) => (
                  <div key={`rmsin-x-${i}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", textAlign: "center", backgroundColor: "rgba(255,255,255,0.02)", border: "1px solid var(--border-color)", borderRadius: "6px", fontFamily: "monospace", fontSize: "0.68rem" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", margin: "-4px 0" }}>
              <ArrowDown className="w-4 h-4 text-text-muted" />
            </div>

            {/* Step 2: RMS Scalar math */}
            <div style={{
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-color)",
              borderRadius: "8px",
              padding: "12px 16px",
              fontSize: "0.8rem",
              display: "flex",
              flexDirection: "column",
              gap: "6px"
            }}>
              <div style={{ fontWeight: "700", color: "var(--text-primary)" }}>Calculate RMS Scaling Factor:</div>
              <div style={{ fontSize: "0.75rem", fontFamily: "monospace", color: "var(--text-secondary)" }}>
                RMS(x) = &radic;<span style={{ borderTop: "1px solid var(--text-secondary)" }}> (1/d) &Sigma; x<sub>i</sub><sup>2</sup> + &epsilon; </span>
              </div>
              <div style={{ fontSize: "0.75rem", fontFamily: "monospace", color: "var(--query-color)", overflowX: "auto", whiteSpace: "nowrap", paddingBottom: "4px" }}>
                RMS = &radic;<span style={{ borderTop: "1px solid var(--query-color)" }}> 1/8 ({simData.embeddingLookup.outputVector.map(v => (v * v).toFixed(2)).join(" + ")}) + 1e-5 </span> = <strong>{rms.toFixed(4)}</strong>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", margin: "-4px 0" }}>
              <ArrowDown className="w-4 h-4 text-text-muted" />
            </div>

            {/* Step 3: Normalized Vector x_hat */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase" }}>
                2. Normalized Vector x&#770; = x / RMS(x)
              </span>
              <div style={{ display: "flex", gap: "4px" }}>
                {normalized.map((val, i) => (
                  <div key={`rmsin-norm-${i}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.68rem", textAlign: "center", backgroundColor: "var(--query-glow)", border: "1px solid var(--query-color)", color: "var(--text-primary)", borderRadius: "6px", fontFamily: "monospace" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", margin: "-4px 0" }}>
              <span style={{ fontSize: "1.1rem", fontWeight: "700", color: "var(--text-muted)" }}>&times;</span>
            </div>

            {/* Step 4: Gamma Scaling vector */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase" }}>
                3. Gamma Weights Vector &gamma; [1 &times; 8]
              </span>
              <div style={{ display: "flex", gap: "4px" }}>
                {gamma.map((val, i) => (
                  <div key={`rmsin-g-${i}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.68rem", textAlign: "center", backgroundColor: "var(--key-glow)", border: "1px solid var(--key-color)", color: "var(--text-primary)", borderRadius: "6px", fontFamily: "monospace" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", margin: "-4px 0" }}>
              <span style={{ fontSize: "1.1rem", fontWeight: "700", color: "var(--text-muted)" }}>=</span>
            </div>

            {/* Step 5: Output */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.75rem", color: "var(--success-color)", fontWeight: "700", textTransform: "uppercase" }}>
                4. RMSNorm Output Vector y = x&#770; &odot; &gamma;
              </span>
              <div style={{ display: "flex", gap: "4px" }}>
                {output.map((val, i) => (
                  <div key={`rmsin-out-${i}`} style={{ flex: 1, minWidth: 0, padding: "6px 0", fontSize: "0.68rem", fontWeight: "700", textAlign: "center", backgroundColor: "var(--success-glow)", border: "1px solid var(--success-color)", color: "var(--text-primary)", borderRadius: "6px", fontFamily: "monospace", boxShadow: "0 0 6px var(--success-glow)" }}>
                    {val.toFixed(2)}
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>

        {/* Right Panel: Interactive Dropout regularizer */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ fontSize: "1.1rem", fontFamily: "inherit", color: "var(--attention-color)" }}>
              Dropout Simulator
            </h3>
            
            {/* Toggle training / evaluation */}
            <div 
              onClick={() => setIsTraining(!isTraining)}
              style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", userSelect: "none" }}
            >
              <span style={{ fontSize: "0.75rem", fontWeight: "700", color: isTraining ? "var(--attention-color)" : "var(--text-muted)" }}>
                {isTraining ? "TRAINING" : "INFERENCE"}
              </span>
              {isTraining ? (
                <ToggleRight className="w-6 h-6 text-attention-color" />
              ) : (
                <ToggleLeft className="w-6 h-6 text-text-muted" />
              )}
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "16px", fontSize: "0.85rem" }}>
            
            {isTraining ? (
              <>
                {/* Dropout slider */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                    <span style={{ color: "var(--text-secondary)" }}>Dropout Probability (p):</span>
                    <strong style={{ color: "var(--attention-color)" }}>{(dropoutRate * 100).toFixed(0)}%</strong>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.6"
                    step="0.05"
                    value={dropoutRate}
                    onChange={(e) => setDropoutRate(parseFloat(e.target.value))}
                    className="custom-slider"
                  />
                </div>

                {/* Reseed button */}
                <button
                  onClick={() => setDropoutSeed(prev => prev + 1)}
                  style={{
                    padding: "8px 12px",
                    borderRadius: "6px",
                    border: "1px solid var(--border-color)",
                    backgroundColor: "var(--bg-surface)",
                    color: "var(--text-primary)",
                    fontSize: "0.75rem",
                    cursor: "pointer",
                    fontWeight: "600",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "6px",
                    transition: "all 0.2s ease"
                  }}
                  onMouseEnter={(e) => e.target.style.borderColor = "var(--attention-color)"}
                  onMouseLeave={(e) => e.target.style.borderColor = "var(--border-color)"}
                >
                  <Sparkles className="w-3.5 h-3.5 text-attention-color" /> 
                  Generate Random Mask
                </button>

                {/* Active Dropout Grid */}
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", backgroundColor: "var(--bg-surface)", padding: "14px", borderRadius: "10px", border: "1px solid var(--border-color)" }}>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", fontFamily: "monospace" }}>
                    Scale Multiplier 1/(1-p) = <strong>{dropoutData.scale.toFixed(2)}x</strong>
                  </div>
                  
                  {/* Grid showing cells and status */}
                  <div style={{ display: "flex", gap: "4px" }}>
                    {dropoutData.values.map((val, idx) => {
                      const isDropped = dropoutData.mask[idx] === 0;
                      return (
                        <div 
                          key={`rmsdrop-cell-${idx}`}
                          style={{
                            flex: 1,
                            padding: "10px 0",
                            backgroundColor: isDropped ? "rgba(239, 68, 68, 0.1)" : "rgba(34, 211, 238, 0.08)",
                            border: `1px solid ${isDropped ? "rgb(239, 68, 68)" : "var(--query-color)"}`,
                            borderRadius: "6px",
                            textAlign: "center",
                            color: isDropped ? "rgb(239, 68, 68)" : "white",
                            fontWeight: "700",
                            textDecoration: isDropped ? "line-through" : "none",
                            fontFamily: "monospace",
                            fontSize: "0.75rem",
                            transition: "all 0.15s ease"
                          }}
                          title={isDropped ? "Zeroed out" : `Scaled: ${output[idx]} * ${dropoutData.scale.toFixed(2)}`}
                        >
                          {val.toFixed(2)}
                        </div>
                      );
                    })}
                  </div>
                </div>

                <p style={{ color: "var(--text-muted)", fontSize: "0.7rem", lineHeight: "1.3", margin: 0 }}>
                  💡 <strong>Inverted Dropout:</strong> To keep expected values identical between training and inference, surviving activations are scaled up by 1 / (1-p). This removes the need to scale weights down during inference.
                </p>
              </>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "12px", padding: "16px", backgroundColor: "var(--bg-surface)", border: "1px dashed var(--border-color)", borderRadius: "8px" }}>
                <div style={{ fontWeight: "700", color: "var(--success-color)", display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "var(--success-color)", boxShadow: "0 0 8px var(--success-color)" }}></span>
                  Inference Mode
                </div>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.75rem", lineHeight: "1.4", margin: 0 }}>
                  In evaluation/inference mode, dropout acts as a simple bypass. All activations pass through unaffected, guaranteeing deterministic text predictions.
                </p>
              </div>
            )}

          </div>
        </div>

      </div>

    </div>
  );
}
