import React, { useState } from "react";
import { Key, ArrowRight, Zap, Info } from "lucide-react";
import InteractiveMatrixMul from "./InteractiveMatrixMul";

export default function SwiGLUSandbox({ simData }) {
  const { input, W_gate, W_up, W_down, gateProj, upProj, siluGate, elementWiseMult, ffnOutput } = simData.swiglu;

  // Slider state for testing the SiLU activation curve
  const [testInput, setTestInput] = useState(0.8);

  const silu = (x) => x / (1 + Math.exp(-x));
  const testOutput = silu(testInput);

  // SVG dimensions for the SiLU plot
  const plotWidth = 260;
  const plotHeight = 140;
  
  // Coordinate mapper for [-4, 4] inputs and [-1, 4] outputs
  const getPlotCoords = (xVal, yVal) => {
    const x = 20 + ((xVal + 4) / 8) * (plotWidth - 40);
    const y = (plotHeight - 20) - ((yVal + 1) / 5) * (plotHeight - 40);
    return { x, y };
  };

  // Build the SVG path for the SiLU curve
  let curvePath = "";
  for (let x = -4.0; x <= 4.0; x += 0.25) {
    const y = silu(x);
    const p = getPlotCoords(x, y);
    if (curvePath === "") {
      curvePath = `M ${p.x} ${p.y}`;
    } else {
      curvePath += ` L ${p.x} ${p.y}`;
    }
  }

  const activePoint = getPlotCoords(testInput, testOutput);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--value-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Key className="w-6 h-6 text-emerald-400 animate-pulse-slow" />
          Step 9: SwiGLU MLP Feed-Forward Block
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Mistral 7B replaces traditional FFN activations (like GELU) with a **SwiGLU (Swish Gated Linear Unit)** block. 
          The input vector $x$ branches into two paths:
          <br />
          - **Gate Path**: Multiplied by {"$W_{gate}$"} and activated by **SiLU (Swish)**: SiLU(x) = x * sigmoid(x).
          - **Up Path**: Multiplied by {"$W_{up}$"} to form a gating state.
          <br />
          These branches are multiplied element-wise and projected back down to the hidden dimension using {"$W_{down}$"}.
        </p>
      </div>

      {/* Main Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: "24px", alignItems: "start" }}>
        
        {/* Left: Interactive Branching Flow and Projections */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Input State Display */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "10px", padding: "16px" }}>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase" }}>
              Input Hidden State Vector x [1 &times; 8]
            </span>
            <div style={{ display: "flex", gap: "4px" }}>
              {input.map((val, idx) => (
                <div key={`swi-input-${idx}`} style={{ flex: 1, padding: "8px 0", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "6px", fontFamily: "monospace", fontSize: "0.75rem" }}>
                  {val.toFixed(2)}
                </div>
              ))}
            </div>
          </div>

          {/* Gate Path Projection */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <h3 style={{ fontSize: "1.05rem", fontFamily: "Outfit", color: "var(--key-color)", display: "flex", alignItems: "center", gap: "6px" }}>
              <Zap className="w-4 h-4" /> Gate Path: X &times; W_gate &rarr; GateProj [1 &times; 8]
            </h3>
            
            <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
              <InteractiveMatrixMul
                leftVal={input}
                rightVal={W_gate}
                outputVal={gateProj}
                leftLabel="Input vector x"
                rightLabel="Gate Weights W_gate"
                outputLabel="Gate Projection Vector"
                leftColor="var(--query-color)"
                rightColor="var(--key-color)"
                outputColor="var(--accent-color)"
                leftRowNames={["x"]}
                rightColNames={Array.from({ length: 8 }, (_, i) => `g${i}`)}
              />
            </div>
          </div>

          {/* Up Path Projection */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <h3 style={{ fontSize: "1.05rem", fontFamily: "Outfit", color: "var(--value-color)", display: "flex", alignItems: "center", gap: "6px" }}>
              <Zap className="w-4 h-4" /> Up Path: X &times; W_up &rarr; UpProj [1 &times; 8]
            </h3>
            
            <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
              <InteractiveMatrixMul
                leftVal={input}
                rightVal={W_up}
                outputVal={upProj}
                leftLabel="Input vector x"
                rightLabel="Up Weights W_up"
                outputLabel="Up Projection Vector"
                leftColor="var(--query-color)"
                rightColor="var(--key-color)"
                outputColor="var(--value-color)"
                leftRowNames={["x"]}
                rightColNames={Array.from({ length: 8 }, (_, i) => `u${i}`)}
              />
            </div>
          </div>

        </div>

        {/* Right: Activation, Gating and Down Projection */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* SiLU Curve Test Sandbox */}
          <div className="glass-panel" style={{ padding: "18px", display: "flex", flexDirection: "column", gap: "12px" }}>
            <h3 style={{ fontSize: "1.05rem", fontFamily: "Outfit", color: "var(--success-color)" }}>
              SiLU (Swish) Activation Curve
            </h3>
            
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
              <svg width={plotWidth} height={plotHeight} style={{ backgroundColor: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-color)", overflow: "visible" }}>
                {/* Axes */}
                <line x1="20" y1="110" x2="240" y2="110" stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" />
                <line x1="130" y1="10" x2="130" y2="130" stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" />
                
                {/* Curve */}
                <path d={curvePath} fill="none" stroke="var(--success-color)" strokeWidth="2.5" />
                
                {/* Active point indicator */}
                <circle cx={activePoint.x} cy={activePoint.y} r="5" fill="var(--attention-color)" stroke="white" strokeWidth="1.5" style={{ filter: "drop-shadow(0 0 4px var(--attention-glow))" }} />
              </svg>
              
              <div style={{ width: "100%", display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                <span>Input z</span>
                <span>SiLU(z)</span>
              </div>
            </div>

            {/* Slider */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", marginBottom: "4px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Test Activation:</span>
                <span style={{ color: "var(--success-color)", fontFamily: "monospace" }}>SiLU({testInput.toFixed(2)}) = {testOutput.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="-4.0"
                max="4.0"
                step="0.1"
                value={testInput}
                onChange={(e) => setTestInput(parseFloat(e.target.value))}
                className="custom-slider"
              />
            </div>
          </div>

          {/* Merge & Down Projection */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <h3 style={{ fontSize: "1.05rem", fontFamily: "Outfit", color: "white" }}>
              Gating Merge & Down Projection
            </h3>

            <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.75rem", fontFamily: "monospace" }}>
              {/* activated gate vector */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ width: "110px", color: "var(--text-muted)" }}>SiLU(GateProj):</span>
                <div style={{ display: "flex", gap: "3px", flex: 1 }}>
                  {siluGate.map((val, idx) => (
                    <div key={`swi-sg-${idx}`} style={{ flex: 1, padding: "4px 0", textAlign: "center", backgroundColor: "hsla(152, 80%, 50%, 0.05)", border: "1px solid var(--success-color)", borderRadius: "4px", color: "white" }}>
                      {val.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ textAlign: "center", fontWeight: "700", color: "var(--text-muted)", fontSize: "1.0rem", margin: "-4px 0" }}>&otimes;</div>

              {/* upProj vector */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ width: "110px", color: "var(--text-muted)" }}>UpProj (xWu):</span>
                <div style={{ display: "flex", gap: "3px", flex: 1 }}>
                  {upProj.map((val, idx) => (
                    <div key={`swi-up-${idx}`} style={{ flex: 1, padding: "4px 0", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "4px", color: "white" }}>
                      {val.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ textAlign: "center", fontWeight: "700", color: "var(--text-muted)", fontSize: "1.0rem", margin: "-4px 0" }}>=</div>

              {/* gated product */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ width: "110px", color: "var(--attention-color)", fontWeight: "700" }}>Gated Product:</span>
                <div style={{ display: "flex", gap: "3px", flex: 1 }}>
                  {elementWiseMult.map((val, idx) => (
                    <div key={`swi-gmult-${idx}`} style={{ flex: 1, padding: "4px 0", textAlign: "center", backgroundColor: "var(--attention-glow)", border: "1px solid var(--attention-color)", borderRadius: "4px", color: "white", fontWeight: "700" }}>
                      {val.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ display: "flex", justifyContent: "center", margin: "4px 0" }}>
              <ArrowRight style={{ transform: "rotate(90deg)", color: "var(--text-muted)" }} className="w-5 h-5" />
            </div>

            {/* Down projection linear multiply */}
            <div style={{ padding: "8px", backgroundColor: "rgba(0,0,0,0.1)", borderRadius: "10px" }}>
              <InteractiveMatrixMul
                leftVal={elementWiseMult}
                rightVal={W_down}
                outputVal={ffnOutput}
                leftLabel="Gated Product Vector [1 x 8]"
                rightLabel="Down Weights W_down [8 x 8]"
                outputLabel="MLP Output Vector [1 x 8]"
                leftColor="var(--attention-color)"
                rightColor="var(--key-color)"
                outputColor="var(--success-color)"
                leftRowNames={["Gated"]}
                rightColNames={Array.from({ length: 8 }, (_, i) => `d${i}`)}
              />
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}
