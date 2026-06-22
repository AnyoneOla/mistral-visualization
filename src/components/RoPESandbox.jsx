import React, { useState } from "react";
import { RefreshCw, Zap, Compass, RotateCw } from "lucide-react";

export default function RoPESandbox({ simData }) {
  const [posM, setPosM] = useState(3); // Query position
  const [posN, setPosN] = useState(1); // Key position
  const [selectedPair, setSelectedPair] = useState(0); // Pair index (0 or 1)

  const thetaBase = 0.25; // Base angle step (radians)
  const angleScale = Math.pow(10, selectedPair); // decay factor for higher dimension pairs
  const theta = thetaBase / angleScale;

  const angleM = posM * theta;
  const angleN = posN * theta;
  const deltaAngle = angleM - angleN;

  // High contrast initial unrotated vectors for clear visual plotting
  const rawQ = [0.8, 0.2];
  const rawK = [0.7, -0.4];

  // Apply rotation matrix
  const rotQ = [
    parseFloat((rawQ[0] * Math.cos(angleM) - rawQ[1] * Math.sin(angleM)).toFixed(3)),
    parseFloat((rawQ[0] * Math.sin(angleM) + rawQ[1] * Math.cos(angleM)).toFixed(3))
  ];

  const rotK = [
    parseFloat((rawK[0] * Math.cos(angleN) - rawK[1] * Math.sin(angleN)).toFixed(3)),
    parseFloat((rawK[0] * Math.sin(angleN) + rawK[1] * Math.cos(angleN)).toFixed(3))
  ];

  // Dot products
  const dotProduct = parseFloat((rotQ[0] * rotK[0] + rotQ[1] * rotK[1]).toFixed(3));
  const rawDotProduct = parseFloat((rawQ[0] * rawK[0] + rawQ[1] * rawK[1]).toFixed(3));

  // Shift together to demonstrate invariance
  const shiftTogether = () => {
    if (posM < 9 && posN < 9) {
      setPosM(prev => prev + 1);
      setPosN(prev => prev + 1);
    } else {
      setPosM(posM - 5);
      setPosN(posN - 5);
    }
  };

  // SVG parameters
  const size = 280;
  const center = size / 2;
  const scale = 100;

  const getSvgCoords = (x, y) => {
    return {
      x: center + x * scale,
      y: center - y * scale
    };
  };

  const qCoords = getSvgCoords(rotQ[0], rotQ[1]);
  const kCoords = getSvgCoords(rotK[0], rotK[1]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--query-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "Outfit", display: "flex", alignItems: "center", gap: "10px" }}>
          <RefreshCw className="w-6 h-6 text-sky-400 animate-spin-slow" />
          Step 4: Rotary Position Embeddings (RoPE)
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Unlike standard transformers which add absolute position vectors, Mistral 7B uses **Rotary Position Embeddings (RoPE)**. 
          RoPE splits the Query and Key vectors into 2D coordinate slices and rotates each pair by an angle proportional to the token's position in the text.
          Because dot product captures vector alignment, rotating both vectors preserves their relative angle, letting self-attention depend naturally on **relative distance**.
        </p>
      </div>

      {/* Main Interactive Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px", alignItems: "start" }}>
        
        {/* Left: 2D Coordinate Rotation Circle */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "16px" }}>
          <h3 style={{ fontSize: "1.1rem", alignSelf: "flex-start", fontFamily: "Outfit", color: "white" }}>
            2D Coordinate Rotation Plane
          </h3>

          <div style={{ 
            position: "relative", 
            width: `${size}px`, 
            height: `${size}px`, 
            backgroundColor: "var(--bg-surface)",
            borderRadius: "50%",
            border: "1px solid var(--border-color)",
            boxShadow: "inset 0 0 20px rgba(0,0,0,0.6)"
          }}>
            <svg width={size} height={size}>
              {/* Reference grid rings */}
              <circle cx={center} cy={center} r={scale} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
              <circle cx={center} cy={center} r={scale * 0.7} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
              <circle cx={center} cy={center} r={scale * 0.4} fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
              
              {/* Coordinate Axes */}
              <line x1="10" y1={center} x2={size - 10} y2={center} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              <line x1={center} y1="10" x2={center} y2={size - 10} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
              
              {/* Origin */}
              <circle cx={center} cy={center} r="4.5" fill="var(--text-muted)" />

              {/* Draw rotation arc */}
              {(() => {
                const rArc = 32;
                const startAngle = -angleM;
                const endAngle = -angleN;
                const x1 = center + rArc * Math.cos(startAngle);
                const y1 = center + rArc * Math.sin(startAngle);
                const x2 = center + rArc * Math.cos(endAngle);
                const y2 = center + rArc * Math.sin(endAngle);
                const largeArc = Math.abs(deltaAngle) > Math.PI ? 1 : 0;
                const sweep = deltaAngle < 0 ? 1 : 0;

                return (
                  <path
                    d={`M ${x1} ${y1} A ${rArc} ${rArc} 0 ${largeArc} ${sweep} ${x2} ${y2}`}
                    fill="none"
                    stroke="var(--attention-color)"
                    strokeWidth="2.5"
                    strokeDasharray="3, 3"
                    opacity="0.8"
                  />
                );
              })()}

              {/* Unrotated Ghost original vectors */}
              {(() => {
                const uQ = getSvgCoords(rawQ[0], rawQ[1]);
                const uK = getSvgCoords(rawK[0], rawK[1]);
                return (
                  <g opacity="0.12">
                    <line x1={center} y1={center} x2={uQ.x} y2={uQ.y} stroke="var(--query-color)" strokeWidth="1.5" strokeDasharray="3, 2" />
                    <line x1={center} y1={center} x2={uK.x} y2={uK.y} stroke="var(--key-color)" strokeWidth="1.5" strokeDasharray="3, 2" />
                  </g>
                );
              })()}

              {/* Rotated Query Vector (Cyan) */}
              <line
                x1={center}
                y1={center}
                x2={qCoords.x}
                y2={qCoords.y}
                stroke="var(--query-color)"
                strokeWidth="3.5"
                style={{ filter: "drop-shadow(0 0 5px var(--query-glow))" }}
              />
              <circle cx={qCoords.x} cy={qCoords.y} r="5" fill="var(--query-color)" />

              {/* Rotated Key Vector (Orange) */}
              <line
                x1={center}
                y1={center}
                x2={kCoords.x}
                y2={kCoords.y}
                stroke="var(--key-color)"
                strokeWidth="3.5"
                style={{ filter: "drop-shadow(0 0 5px var(--key-glow))" }}
              />
              <circle cx={kCoords.x} cy={kCoords.y} r="5" fill="var(--key-color)" />
            </svg>
            
            {/* Axis labels */}
            <span style={{ position: "absolute", left: "10px", bottom: "10px", fontSize: "0.6rem", color: "var(--text-muted)", fontFamily: "monospace" }}>Real Part</span>
            <span style={{ position: "absolute", right: "10px", top: "10px", fontSize: "0.6rem", color: "var(--text-muted)", fontFamily: "monospace" }}>Imaginary Part</span>
          </div>
        </div>

        {/* Right: Controllers & Invariance proof */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          
          <div className="glass-panel" style={{ padding: "16px", display: "flex", flexDirection: "column", gap: "14px" }}>
            <h3 style={{ fontSize: "1rem", fontFamily: "Outfit", color: "var(--query-color)" }}>Rotation Configuration</h3>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "0.8rem" }}>
              
              {/* Pair index select */}
              <div>
                <span style={{ display: "block", color: "var(--text-secondary)", marginBottom: "6px" }}>Frequency Slice:</span>
                <div style={{ display: "flex", gap: "4px" }}>
                  {["Pair 0 (Fast freq)", "Pair 1 (Slow freq)"].map((lbl, idx) => (
                    <button
                      key={`pair-btn-${idx}`}
                      onClick={() => setSelectedPair(idx)}
                      style={{
                        flex: 1,
                        padding: "6px",
                        fontSize: "0.75rem",
                        borderRadius: "6px",
                        border: "1px solid var(--border-color)",
                        backgroundColor: selectedPair === idx ? "var(--accent-color)" : "var(--bg-surface)",
                        color: selectedPair === idx ? "white" : "var(--text-secondary)",
                        cursor: "pointer",
                        fontWeight: "600",
                        transition: "all 0.2s ease"
                      }}
                    >
                      {lbl}
                    </button>
                  ))}
                </div>
              </div>

              {/* Slider posM */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Query Token Position (m):</span>
                  <span style={{ fontWeight: "700", color: "var(--query-color)" }}>m = {posM}</span>
                </div>
                <input
                  type="range" min="0" max="10" step="1" value={posM}
                  onChange={(e) => setPosM(parseInt(e.target.value))}
                  className="custom-slider"
                />
              </div>

              {/* Slider posN */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                  <span style={{ color: "var(--text-secondary)" }}>Key Token Position (n):</span>
                  <span style={{ fontWeight: "700", color: "var(--key-color)" }}>n = {posN}</span>
                </div>
                <input
                  type="range" min="0" max="10" step="1" value={posN}
                  onChange={(e) => setPosN(parseInt(e.target.value))}
                  className="custom-slider"
                />
              </div>

            </div>
          </div>

          {/* Proof panel */}
          <div className="glass-panel" style={{ padding: "16px", borderColor: "var(--attention-glow)" }}>
            <h3 style={{ fontSize: "1rem", fontFamily: "Outfit", color: "var(--attention-color)", display: "flex", alignItems: "center", gap: "6px", marginBottom: "12px" }}>
              Relative Angle Invariance Proof
            </h3>

            <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.8rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-secondary)" }}>Relative Distance (d = m - n):</span>
                <strong style={{ color: "white" }}>{posM - posN}</strong>
              </div>

              <div style={{ fontFamily: "monospace", fontSize: "0.7rem", color: "var(--text-muted)", backgroundColor: "var(--bg-surface)", padding: "8px", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
                &theta;_q = {angleM.toFixed(3)} rad ({((angleM * 180)/Math.PI).toFixed(0)}&deg;)<br />
                &theta;_k = {angleN.toFixed(3)} rad ({((angleN * 180)/Math.PI).toFixed(0)}&deg;)<br />
                &Delta;&theta; = {deltaAngle.toFixed(3)} rad ({((deltaAngle * 180)/Math.PI).toFixed(0)}&deg;)
              </div>

              <div style={{ display: "flex", gap: "4px", backgroundColor: "var(--bg-surface)", padding: "10px", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.05)", fontFamily: "monospace", fontSize: "0.72rem", justifyContent: "space-between" }}>
                <span>Raw Dot: <strong>{rawDotProduct}</strong></span>
                <span style={{ color: "var(--attention-color)" }}>Rotated Dot: <strong>{dotProduct}</strong></span>
              </div>

              <button
                onClick={shiftTogether}
                style={{
                  padding: "8px 12px",
                  borderRadius: "6px",
                  border: "none",
                  backgroundColor: "var(--attention-glow)",
                  color: "var(--attention-color)",
                  fontWeight: "700",
                  fontSize: "0.75rem",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  transition: "all 0.2s ease"
                }}
              >
                <Compass className="w-3.5 h-3.5" /> Shift Both Positions (+1)
              </button>

              <p style={{ color: "var(--text-muted)", fontSize: "0.7rem", lineHeight: "1.3", margin: 0 }}>
                💡 Click the shift button! As both Query ($m$) and Key ($n$) positions increment, they rotate but their **relative angle** and **Rotated Dot Product** remain identical.
              </p>
            </div>
          </div>

          {/* Relative Distance Attention Decay Graph */}
          <div className="glass-panel" style={{ padding: "16px", borderColor: "rgba(34, 211, 238, 0.2)" }}>
            <h3 style={{ fontSize: "1rem", marginBottom: "10px", fontFamily: "Outfit", color: "var(--query-color)" }}>
              Destructive Interference Decay Curve
            </h3>
            
            <p style={{ color: "var(--text-muted)", fontSize: "0.7rem", lineHeight: "1.3", marginBottom: "8px" }}>
              As relative distance $d$ increases, high frequency rotation components cause the average dot product to decay, prioritizing local context.
            </p>

            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
              <svg width="240" height="110" style={{ backgroundColor: "var(--bg-surface)", borderRadius: "6px", border: "1px solid var(--border-color)", overflow: "visible" }}>
                {/* Horizontal reference */}
                <line x1="20" y1="55" x2="220" y2="55" stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" />
                
                {/* Ticks */}
                {Array.from({ length: 6 }).map((_, dVal) => {
                  const x = 20 + (dVal / 5) * 200;
                  return (
                    <g key={`rope-tick-${dVal}`}>
                      <line x1={x} y1="51" x2={x} y2="59" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
                      <text x={x} y="68" fill="var(--text-muted)" fontSize="7px" fontFamily="monospace" textAnchor="middle">{dVal * 2}</text>
                    </g>
                  );
                })}

                {/* Curve */}
                {(() => {
                  const pts = [];
                  for (let dVal = 0; dVal <= 10; dVal += 0.2) {
                    const ang = dVal * theta;
                    const rQ = [rawQ[0] * Math.cos(ang) - rawQ[1] * Math.sin(ang), rawQ[0] * Math.sin(ang) + rawQ[1] * Math.cos(ang)];
                    const dotVal = rQ[0] * rawK[0] + rQ[1] * rawK[1];
                    const x = 20 + (dVal / 10) * 200;
                    const y = 55 - dotVal * 35;
                    pts.push(`${x},${y}`);
                  }
                  
                  return (
                    <path
                      d={`M ${pts.join(" L ")}`}
                      fill="none"
                      stroke="var(--query-color)"
                      strokeWidth="2"
                    />
                  );
                })()}

                {/* Highlight active dot */}
                {(() => {
                  const currD = Math.abs(posM - posN);
                  const ang = currD * theta;
                  const rQ = [rawQ[0] * Math.cos(ang) - rawQ[1] * Math.sin(ang), rawQ[0] * Math.sin(ang) + rawQ[1] * Math.cos(ang)];
                  const dotVal = rQ[0] * rawK[0] + rQ[1] * rawK[1];
                  const x = 20 + (currD / 10) * 200;
                  const y = 55 - dotVal * 35;
                  
                  return (
                    <circle
                      cx={x}
                      cy={y}
                      r="4"
                      fill="var(--attention-color)"
                      stroke="white"
                      strokeWidth="1"
                    />
                  );
                })()}
              </svg>
              
              <div style={{ width: "100%", fontSize: "0.65rem", color: "var(--text-muted)", display: "flex", justifyContent: "space-between" }}>
                <span>Distance (d)</span>
                <span>Dot Product alignment</span>
              </div>
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
