import React, { useState, useEffect, useRef } from "react";
import { Key, ArrowRight, Zap, Info, Play, Pause, RotateCcw, ChevronRight, ChevronLeft, Sparkles, Activity } from "lucide-react";
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

  // --- Tab State for visualizers ---
  const [activeVizTab, setActiveVizTab] = useState("network"); // "network" or "pathway"

  // --- Animation States for MLP Flow ---
  const [animStep, setAnimStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef(null);

  const totalSteps = 6;
  const stepsInfo = [
    {
      title: "1. Input Hidden Vector (x)",
      desc: "Normalized vector enters the MLP Feed-Forward block after being computed by RMSNorm 2."
    },
    {
      title: "2. Branching Projections (Wg & Wu)",
      desc: "The vector splits into parallel branches: Gating Path (Wg) and Up Projection Path (Wu)."
    },
    {
      title: "3. Gate Activation (SiLU / Swish)",
      desc: "Gate Projection is passed through the non-linear SiLU activation, generating gate coefficients."
    },
    {
      title: "4. Merging Paths (Element-wise ⊗)",
      desc: "The activated Gate path modulates the Up path element-by-element (gating operation)."
    },
    {
      title: "5. Down Projection (W_down)",
      desc: "The gated product is projected back to the hidden dimension using the down projection matrix."
    },
    {
      title: "6. Complete MLP Output",
      desc: "The down-projected state is complete and will be added back to the residual stream."
    }
  ];

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setAnimStep((prev) => (prev + 1) % totalSteps);
      }, 2500);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying]);

  const handlePrevStep = () => {
    setIsPlaying(false);
    setAnimStep((prev) => (prev === 0 ? totalSteps - 1 : prev - 1));
  };

  const handleNextStep = () => {
    setIsPlaying(false);
    setAnimStep((prev) => (prev + 1) % totalSteps);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setAnimStep(0);
  };

  // --- Neural Network Pulse Animation State ---
  const [netPulseStage, setNetPulseStage] = useState(0); // 0: Idle, 1: Synapses 1, 2: Gate/Up Nodes, 3: SiLU, 4: Gated Nodes, 5: Synapses 2, 6: Output
  const [isNetPulseRunning, setIsNetPulseRunning] = useState(false);
  const netTimeoutRefs = useRef([]);

  const triggerNetPulse = () => {
    if (isNetPulseRunning) return;
    setIsNetPulseRunning(true);
    
    // Clear any existing timeouts
    netTimeoutRefs.current.forEach(clearTimeout);
    netTimeoutRefs.current = [];

    const stages = [
      { stage: 1, delay: 0 },
      { stage: 2, delay: 1000 },
      { stage: 3, delay: 1800 },
      { stage: 4, delay: 2600 },
      { stage: 5, delay: 3400 },
      { stage: 6, delay: 4400 },
      { stage: 0, delay: 5800 } // Reset to idle
    ];

    stages.forEach(({ stage, delay }) => {
      const t = setTimeout(() => {
        setNetPulseStage(stage);
        if (stage === 0) {
          setIsNetPulseRunning(false);
        }
      }, delay);
      netTimeoutRefs.current.push(t);
    });
  };

  useEffect(() => {
    return () => {
      netTimeoutRefs.current.forEach(clearTimeout);
    };
  }, []);

  // Y positions for 8 nodes in a column (shifted down for label spacing)
  const layerY = [65, 105, 145, 185, 225, 265, 305, 345];
  // Y positions for 12 intermediate nodes in a column
  const layer12Y = [55, 82, 109, 136, 163, 190, 217, 244, 271, 298, 325, 352];
  // Gate branch shifted up: Y indices 0 to 11 maps to compressed coordinates
  const gateY = [50, 63, 76, 89, 102, 115, 128, 141, 154, 167, 180, 193];
  // Up branch shifted down: Y indices 0 to 11 maps to compressed coordinates
  const upY = [222, 235, 248, 261, 274, 287, 300, 313, 326, 339, 352, 365];

  // Helper to color nodes by value
  const getNodeColor = (val, minVal = -1.5, maxVal = 1.5, baseHue = 190) => {
    const norm = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
    return `hsla(${baseHue}, 90%, ${40 + norm * 35}%, ${0.3 + norm * 0.7})`;
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Description Panel */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--value-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Key className="w-6 h-6 text-emerald-400 animate-pulse-slow" />
          Step 9: SwiGLU MLP Feed-Forward Block
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Mistral 7B replaces traditional FFN activations (like GELU) with a <strong>SwiGLU (Swish Gated Linear Unit)</strong> block. 
          The input vector x branches into two paths:
          <br />
          - <strong>Gate Path</strong>: Multiplied by W<sub>gate</sub> and activated by <strong>SiLU (Swish)</strong>: SiLU(x) = x * sigmoid(x).
          - <strong>Up Path</strong>: Multiplied by W<sub>up</sub> to form a gating state.
          <br />
          These branches are multiplied element-wise and projected back down to the hidden dimension using W<sub>down</sub>.
        </p>
      </div>

      {/* Interactive Visualizer Tabs */}
      <div className="glass-panel" style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "16px" }}>
        
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-color)", paddingBottom: "12px", flexWrap: "wrap", gap: "12px" }}>
          <div className="tab-container">
            <button 
              onClick={() => setActiveVizTab("network")}
              className={`tab-button ${activeVizTab === "network" ? "active" : ""}`}
              style={{ display: "flex", alignItems: "center", gap: "6px" }}
            >
              <Activity width={14} height={14} />
              <span>Neuron Network Graph</span>
            </button>
            <button 
              onClick={() => setActiveVizTab("pathway")}
              className={`tab-button ${activeVizTab === "pathway" ? "active" : ""}`}
              style={{ display: "flex", alignItems: "center", gap: "6px" }}
            >
              <Zap width={14} height={14} />
              <span>Pipeline Pathway Flow</span>
            </button>
          </div>

          {activeVizTab === "pathway" ? (
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <button onClick={handlePrevStep} className="ctrl-btn" style={{ padding: "8px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-surface)", cursor: "pointer", color: "var(--text-primary)", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
                <ChevronLeft size={16} />
              </button>
              <button onClick={() => setIsPlaying(!isPlaying)} className="ctrl-btn" style={{ padding: "8px 16px", gap: "8px", borderRadius: "8px", border: isPlaying ? "1px solid var(--attention-color)" : "1px solid var(--success-color)", backgroundColor: isPlaying ? "var(--attention-glow)" : "var(--success-glow)", cursor: "pointer", display: "flex", alignItems: "center", color: "var(--text-primary)", fontWeight: "600", transition: "all 0.2s" }}>
                {isPlaying ? <Pause size={16} color="var(--attention-color)" /> : <Play size={16} color="var(--success-color)" />}
                <span style={{ fontSize: "0.75rem" }}>{isPlaying ? "Pause" : "Play Flow"}</span>
              </button>
              <button onClick={handleNextStep} className="ctrl-btn" style={{ padding: "8px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-surface)", cursor: "pointer", color: "var(--text-primary)", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
                <ChevronRight size={16} />
              </button>
              <button onClick={handleReset} className="ctrl-btn" style={{ padding: "8px", borderRadius: "8px", border: "1px solid var(--border-color)", backgroundColor: "var(--bg-surface)", cursor: "pointer", color: "var(--text-secondary)", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
                <RotateCcw size={16} />
              </button>
            </div>
          ) : (
            <button 
              onClick={triggerNetPulse} 
              disabled={isNetPulseRunning}
              className="ctrl-btn" 
              style={{ 
                padding: "8px 16px", 
                borderRadius: "8px", 
                border: "1px solid var(--attention-color)", 
                backgroundColor: isNetPulseRunning ? "var(--bg-surface)" : "var(--attention-color)", 
                color: isNetPulseRunning ? "var(--text-muted)" : "#FFFFFF",
                cursor: isNetPulseRunning ? "not-allowed" : "pointer", 
                display: "flex", 
                alignItems: "center", 
                gap: "8px",
                fontWeight: "700",
                fontSize: "0.75rem",
                boxShadow: isNetPulseRunning ? "none" : "0 4px 12px var(--attention-glow)",
                transition: "all 0.2s"
              }}
            >
              <Sparkles className="w-4 h-4" style={{ color: isNetPulseRunning ? "var(--attention-color)" : "#FFFFFF" }} />
              <span>{isNetPulseRunning ? "Propagating Signal..." : "Trigger Feedforward Pulse"}</span>
            </button>
          )}
        </div>

        {/* Tab 1: NEURON NETWORK GRAPH */}
        {activeVizTab === "network" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            <div style={{ padding: "10px 14px", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "8px", fontSize: "0.75rem", color: "var(--text-secondary)", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                  <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "var(--success-color)", display: "inline-block" }}></span>
                  Positive Activation
                </span>
                <span style={{ color: "var(--border-color)" }}>|</span>
                <span style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                  <span style={{ width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "var(--query-color)", display: "inline-block" }}></span>
                  Negative Activation
                </span>
                <span style={{ color: "var(--border-color)" }}>|</span>
                <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                  <Info width={12} height={12} />
                  Hover over nodes to see vector index value
                </span>
              </span>
              <span style={{ fontWeight: "700", color: "var(--attention-color)" }}>
                {netPulseStage === 0 && "Ready. Click Pulse to send input x vector."}
                {netPulseStage === 1 && "Stage 1: Input x propagating through W_gate and W_up synapses..."}
                {netPulseStage === 2 && "Stage 2: Gating Path (xWg) and Up Path (xWu) projections loaded..."}
                {netPulseStage === 3 && "Stage 3: Gate Projection nodes applying SiLU Activation (Swish)..."}
                {netPulseStage === 4 && "Stage 4: Multiplying activated Gate & Up nodes element-wise (⊗)..."}
                {netPulseStage === 5 && "Stage 5: Gated product propagating down through W_down synapses..."}
                {netPulseStage === 6 && "Stage 6: Output computed and ready for residual stream addition."}
              </span>
            </div>

            {/* Neural Net SVG Canvas */}
            <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.03)", padding: "12px 0" }}>
              <svg width="760" height="410" style={{ overflow: "visible" }}>
                
                {/* DEFINITIONS FOR GRADIENTS AND GLOWS */}
                <defs>
                  <filter id="glow-neon" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                  </filter>
                  
                  {/* Gradients for synapses */}
                  <linearGradient id="grad-gate" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--query-color)" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="var(--accent-color)" stopOpacity="0.8" />
                  </linearGradient>
                  <linearGradient id="grad-up" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--value-color)" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="var(--value-color)" stopOpacity="0.8" />
                  </linearGradient>
                  <linearGradient id="grad-down" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--attention-color)" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="var(--success-color)" stopOpacity="0.8" />
                  </linearGradient>
                </defs>

                {/* --- SYNAPSE (CONNECTION) LINES --- */}
                {/* 1. Input layer to Gate Layer Projections (Wg) */}
                {layerY.map((y1, i) => 
                  gateY.map((y2, j) => {
                    const isActive = netPulseStage === 1;
                    return (
                      <line 
                        key={`syn-g-${i}-${j}`}
                        x1="50" y1={y1} x2="220" y2={y2}
                        stroke={isActive ? "url(#grad-gate)" : "var(--text-muted)"}
                        strokeWidth={isActive ? "1.5" : "0.8"}
                        opacity={isActive ? "1" : "0.15"}
                        strokeDasharray={isActive ? "5 5" : "none"}
                        style={{ 
                          animation: isActive ? "flowMarquee 0.8s linear infinite" : "none",
                          transition: "stroke-width 0.4s, opacity 0.4s"
                        }}
                      />
                    );
                  })
                )}

                {/* 2. Input layer to Up Layer Projections (Wu) */}
                {layerY.map((y1, i) => 
                  upY.map((y2, j) => {
                    const isActive = netPulseStage === 1;
                    return (
                      <line 
                        key={`syn-u-${i}-${j}`}
                        x1="50" y1={y1} x2="220" y2={y2}
                        stroke={isActive ? "url(#grad-up)" : "var(--text-muted)"}
                        strokeWidth={isActive ? "1.5" : "0.8"}
                        opacity={isActive ? "1" : "0.15"}
                        strokeDasharray={isActive ? "5 5" : "none"}
                        style={{ 
                          animation: isActive ? "flowMarquee 0.8s linear infinite" : "none",
                          transition: "stroke-width 0.4s, opacity 0.4s"
                        }}
                      />
                    );
                  })
                )}

                {/* 3. Gate to Gated (one-to-one mapping representing SiLU filter output) */}
                {gateY.map((y1, i) => {
                  const isActive = netPulseStage >= 3 && netPulseStage <= 4;
                  return (
                    <line 
                      key={`syn-silu-${i}`}
                      x1="220" y1={y1} x2="440" y2={layer12Y[i]}
                      stroke={isActive ? "var(--success-color)" : "var(--text-muted)"}
                      strokeWidth={isActive ? "2.5" : "1"}
                      opacity={isActive ? "1" : "0.3"}
                      strokeDasharray={isActive ? "4 4" : "none"}
                      style={{ 
                        animation: isActive ? "flowMarquee 0.6s linear infinite" : "none",
                        transition: "stroke-width 0.4s, opacity 0.4s"
                      }}
                    />
                  );
                })}

                {/* 4. Up to Gated (one-to-one mapping element-wise multiplication) */}
                {upY.map((y1, i) => {
                  const isActive = netPulseStage === 4;
                  return (
                    <line 
                      key={`syn-mult-${i}`}
                      x1="220" y1={y1} x2="440" y2={layer12Y[i]}
                      stroke={isActive ? "var(--value-color)" : "var(--text-muted)"}
                      strokeWidth={isActive ? "2.5" : "1"}
                      opacity={isActive ? "1" : "0.3"}
                      strokeDasharray={isActive ? "4 4" : "none"}
                      style={{ 
                        animation: isActive ? "flowMarquee 0.6s linear infinite" : "none",
                        transition: "stroke-width 0.4s, opacity 0.4s"
                      }}
                    />
                  );
                })}

                {/* 5. Gated layer to Output layer (W_down) */}
                {layer12Y.map((y1, i) => 
                  layerY.map((y2, j) => {
                    const isActive = netPulseStage === 5;
                    return (
                      <line 
                        key={`syn-d-${i}-${j}`}
                        x1="440" y1={y1} x2="650" y2={y2}
                        stroke={isActive ? "url(#grad-down)" : "var(--text-muted)"}
                        strokeWidth={isActive ? "1.5" : "0.8"}
                        opacity={isActive ? "1" : "0.15"}
                        strokeDasharray={isActive ? "5 5" : "none"}
                        style={{ 
                          animation: isActive ? "flowMarquee 0.8s linear infinite" : "none",
                          transition: "stroke-width 0.4s, opacity 0.4s"
                        }}
                      />
                    );
                  })
                )}

                {/* --- NEURON NODE GROUPS --- */}
                {/* 1. Input nodes x */}
                <g>
                  <text x="50" y="25" fill="var(--text-muted)" fontSize="0.6rem" textAnchor="middle" fontWeight="bold">Input x (d=8)</text>
                  {input.map((val, idx) => (
                    <g key={`net-in-${idx}`} transform={`translate(50, ${layerY[idx]})`} className="cursor-help">
                      <circle 
                        r="9" 
                        fill={getNodeColor(val, -1.5, 1.5, val >= 0 ? 152 : 355)}
                        stroke={netPulseStage >= 1 ? "var(--text-primary)" : "var(--text-secondary)"}
                        strokeWidth={netPulseStage >= 1 ? "1.5" : "0.8"}
                        opacity={netPulseStage >= 1 ? "1" : "0.5"}
                        style={{ transition: "all 0.3s" }}
                      />
                      <title>{`x[${idx}] = ${val.toFixed(3)}`}</title>
                    </g>
                  ))}
                </g>

                {/* 2. Gate nodes (x * W_gate) */}
                <g>
                  <text x="220" y="25" fill="var(--query-color)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Gate Proj (d=12)</text>
                  {gateProj.map((val, idx) => {
                    const isLit = netPulseStage >= 2;
                    return (
                      <g key={`net-gp-${idx}`} transform={`translate(220, ${gateY[idx]})`} className="cursor-help">
                        <circle 
                          r="5" 
                          fill={isLit ? getNodeColor(val, -2.5, 2.5, val >= 0 ? 152 : 355) : "var(--bg-card)"}
                          stroke={isLit ? "var(--query-color)" : "var(--text-secondary)"}
                          strokeWidth={isLit ? "1.5" : "0.8"}
                          opacity={isLit ? "1" : "0.5"}
                          style={{ transition: "all 0.3s" }}
                        />
                        <title>{`GateProj[${idx}] = ${val.toFixed(3)}`}</title>
                      </g>
                    );
                  })}
                </g>

                {/* 3. Up nodes (x * W_up) */}
                <g>
                  <text x="220" y="390" fill="var(--value-color)" fontSize="0.55rem" textAnchor="middle" fontWeight="bold">Up Proj (d=12)</text>
                  {upProj.map((val, idx) => {
                    const isLit = netPulseStage >= 2;
                    return (
                      <g key={`net-up-${idx}`} transform={`translate(220, ${upY[idx]})`} className="cursor-help">
                        <circle 
                          r="5" 
                          fill={isLit ? getNodeColor(val, -2.5, 2.5, val >= 0 ? 152 : 355) : "var(--bg-card)"}
                          stroke={isLit ? "var(--value-color)" : "var(--text-secondary)"}
                          strokeWidth={isLit ? "1.5" : "0.8"}
                          opacity={isLit ? "1" : "0.5"}
                          style={{ transition: "all 0.3s" }}
                        />
                        <title>{`UpProj[${idx}] = ${val.toFixed(3)}`}</title>
                      </g>
                    );
                  })}
                </g>

                {/* 4. SiLU Activation Indicator overlays */}
                {netPulseStage >= 3 && (
                  <g>
                    {siluGate.map((val, idx) => (
                      <circle 
                        key={`silu-g-${idx}`}
                        cx="220" cy={gateY[idx]} r="8"
                        fill="none"
                        stroke="var(--success-color)"
                        strokeWidth="1.5"
                        style={{ 
                          animation: "pulseGlow 1.2s infinite ease-in-out",
                          filter: "drop-shadow(0 0 3px var(--success-color))"
                        }}
                      />
                    ))}
                  </g>
                )}

                {/* 5. Gated Product nodes (SiLU(Gate) * Up) */}
                <g>
                  <text x="440" y="25" fill="var(--attention-color)" fontSize="0.6rem" textAnchor="middle" fontWeight="bold">Gated Product (d=12)</text>
                  {elementWiseMult.map((val, idx) => {
                    const isLit = netPulseStage >= 4;
                    return (
                      <g key={`net-gated-${idx}`} transform={`translate(440, ${layer12Y[idx]})`} className="cursor-help">
                        <circle 
                          r="9" 
                          fill={isLit ? getNodeColor(val, -2.5, 2.5, val >= 0 ? 152 : 355) : "var(--bg-card)"}
                          stroke={isLit ? "var(--attention-color)" : "var(--text-secondary)"}
                          strokeWidth={isLit ? "2" : "0.8"}
                          opacity={isLit ? "1" : "0.5"}
                          style={{ 
                            transition: "all 0.3s",
                            filter: isLit ? "drop-shadow(0 0 6px var(--attention-glow))" : "none"
                          }}
                        />
                        <title>{`GatedProduct[${idx}] = ${val.toFixed(3)}`}</title>
                      </g>
                    );
                  })}
                </g>

                {/* 6. Output nodes (Gated * W_down) */}
                <g>
                  <text x="650" y="25" fill="var(--success-color)" fontSize="0.6rem" textAnchor="middle" fontWeight="bold">MLP Output (d=8)</text>
                  {ffnOutput.map((val, idx) => {
                    const isLit = netPulseStage >= 6;
                    return (
                      <g key={`net-out-${idx}`} transform={`translate(650, ${layerY[idx]})`} className="cursor-help">
                        <circle 
                          r="9" 
                          fill={isLit ? getNodeColor(val, -1.5, 1.5, val >= 0 ? 152 : 355) : "var(--bg-card)"}
                          stroke={isLit ? "var(--success-color)" : "var(--text-secondary)"}
                          strokeWidth={isLit ? "2.5" : "0.8"}
                          opacity={isLit ? "1" : "0.5"}
                          style={{ 
                            transition: "all 0.4s",
                            filter: isLit ? "drop-shadow(0 0 8px var(--success-glow))" : "none"
                          }}
                        />
                        <title>{`ffnOutput[${idx}] = ${val.toFixed(3)}`}</title>
                      </g>
                    );
                  })}
                </g>

              </svg>
            </div>
          </div>
        )}

        {/* Tab 2: PIPELINE PATHWAY FLOW */}
        {activeVizTab === "pathway" && (
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {/* Pathway Stage Info Readout */}
            <div style={{ padding: "12px", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "8px", display: "flex", flexDirection: "column", gap: "4px" }}>
              <span style={{ fontSize: "0.85rem", fontWeight: "700", color: "var(--accent-color)" }}>
                {stepsInfo[animStep].title}
              </span>
              <p style={{ margin: 0, fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                {stepsInfo[animStep].desc}
              </p>
            </div>

            {/* SVG Pipeline Graph */}
            <div style={{ display: "flex", justifyContent: "center", backgroundColor: "var(--bg-surface)", padding: "15px 0", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.02)" }}>
              <svg width="740" height="180" style={{ overflow: "visible" }}>
                
                {/* Background Paths */}
                {/* Split from x to upper path and lower path */}
                <path d="M 50 90 L 120 90 L 120 45 L 200 45" fill="none" stroke={animStep >= 1 ? "var(--query-color)" : "var(--border-color)"} strokeWidth={animStep >= 1 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />
                <path d="M 120 90 L 120 135 L 200 135" fill="none" stroke={animStep >= 1 ? "var(--value-color)" : "var(--border-color)"} strokeWidth={animStep >= 1 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />
                
                {/* Gate path: W_gate to SiLU */}
                <line x1="200" y1="45" x2="310" y2="45" stroke={animStep >= 2 ? "var(--query-color)" : "var(--border-color)"} strokeWidth={animStep >= 2 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />
                
                {/* SiLU to merge */}
                <path d="M 310 45 L 420 45 L 420 90" fill="none" stroke={animStep >= 3 ? "var(--success-color)" : "var(--border-color)"} strokeWidth={animStep >= 3 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />
                
                {/* Up path: W_up to merge */}
                <path d="M 200 135 L 420 135 L 420 90" fill="none" stroke={animStep >= 2 ? "var(--value-color)" : "var(--border-color)"} strokeWidth={animStep >= 2 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />
                
                {/* Merge to W_down */}
                <line x1="420" y1="90" x2="520" y2="90" stroke={animStep >= 4 ? "var(--attention-color)" : "var(--border-color)"} strokeWidth={animStep >= 4 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />
                
                {/* W_down to Output */}
                <line x1="520" y1="90" x2="680" y2="90" stroke={animStep >= 5 ? "var(--success-color)" : "var(--border-color)"} strokeWidth={animStep >= 5 ? "3" : "2"} style={{ transition: "stroke 0.4s, stroke-width 0.4s" }} />

                {/* Nodes */}
                {/* 1. Input x */}
                <g transform="translate(50, 90)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(0)}>
                  <circle r="22" fill="var(--bg-surface)" stroke={animStep === 0 ? "var(--accent-color)" : "var(--border-color)"} strokeWidth="2.5" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--text-primary)" fontSize="0.75rem" fontWeight="bold">Input x</text>
                </g>

                {/* 2. W_gate Projection */}
                <g transform="translate(200, 45)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(1)}>
                  <rect x="-35" y="-20" width="70" height="40" rx="6" fill="var(--bg-surface)" stroke={animStep === 1 ? "var(--query-color)" : "var(--border-color)"} strokeWidth="2" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--text-secondary)" fontSize="0.7rem">x &times; W_gate</text>
                </g>

                {/* 3. W_up Projection */}
                <g transform="translate(200, 135)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(1)}>
                  <rect x="-35" y="-20" width="70" height="40" rx="6" fill="var(--bg-surface)" stroke={animStep === 1 ? "var(--value-color)" : "var(--border-color)"} strokeWidth="2" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--text-secondary)" fontSize="0.7rem">x &times; W_up</text>
                </g>

                {/* 4. SiLU Activation */}
                <g transform="translate(310, 45)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(2)}>
                  <circle r="20" fill="var(--bg-surface)" stroke={animStep === 2 ? "var(--success-color)" : "var(--border-color)"} strokeWidth="2" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--success-color)" fontSize="0.7rem" fontWeight="bold">SiLU</text>
                </g>

                {/* 5. Gating multiplication node (⊗) */}
                <g transform="translate(420, 90)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(3)}>
                  <circle r="18" fill="var(--bg-surface)" stroke={animStep === 3 ? "var(--attention-color)" : "var(--border-color)"} strokeWidth="2" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--attention-color)" fontSize="0.9rem" fontWeight="bold">&otimes;</text>
                </g>

                {/* 6. Down Projection */}
                <g transform="translate(520, 90)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(4)}>
                  <rect x="-35" y="-20" width="70" height="40" rx="6" fill="var(--bg-surface)" stroke={animStep === 4 ? "var(--accent-color)" : "var(--border-color)"} strokeWidth="2" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--text-secondary)" fontSize="0.7rem">&times; W_down</text>
                </g>

                {/* 7. Output Vector */}
                <g transform="translate(680, 90)" style={{ cursor: "pointer" }} onClick={() => setAnimStep(5)}>
                  <circle r="22" fill="var(--bg-surface)" stroke={animStep === 5 ? "var(--success-color)" : "var(--border-color)"} strokeWidth="2.5" style={{ transition: "stroke 0.3s" }} />
                  <text textAnchor="middle" dy="4" fill="var(--text-primary)" fontSize="0.72rem" fontWeight="bold">Output</text>
                </g>

                {/* --- Animated Flow Particles --- */}
                {/* Particle for Step 0 (Start) */}
                {animStep === 0 && (
                  <circle cx="50" cy="90" r="6" fill="var(--accent-color)" style={{ filter: "drop-shadow(0 0 5px var(--accent-color))" }}>
                    <animate attributeName="r" values="5;7;5" dur="1.5s" repeatCount="indefinite" />
                  </circle>
                )}

                {/* Particles for Step 1 (Branching out to W_gate and W_up) */}
                {animStep === 1 && (
                  <>
                    {/* Gate branch particle */}
                    <circle r="5" fill="var(--query-color)" style={{ filter: "drop-shadow(0 0 4px var(--query-color))" }}>
                      <animateMotion dur="2s" repeatCount="indefinite" path="M 50 90 L 120 90 L 120 45 L 200 45" />
                    </circle>
                    {/* Up branch particle */}
                    <circle r="5" fill="var(--value-color)" style={{ filter: "drop-shadow(0 0 4px var(--value-color))" }}>
                      <animateMotion dur="2s" repeatCount="indefinite" path="M 50 90 L 120 90 L 120 135 L 200 135" />
                    </circle>
                  </>
                )}

                {/* Particles for Step 2 (WG is computed, proceeding through SiLU) */}
                {animStep === 2 && (
                  <>
                    <circle r="5" fill="var(--query-color)" style={{ filter: "drop-shadow(0 0 4px var(--query-color))" }}>
                      <animateMotion dur="2s" repeatCount="indefinite" path="M 200 45 L 310 45" />
                    </circle>
                    <circle cx="200" cy="135" r="5" fill="var(--value-color)" opacity="0.6" />
                  </>
                )}

                {/* Particles for Step 3 (Gate is activated, merge paths together at multiply) */}
                {animStep === 3 && (
                  <>
                    {/* From SiLU to merge */}
                    <circle r="5" fill="var(--success-color)" style={{ filter: "drop-shadow(0 0 4px var(--success-color))" }}>
                      <animateMotion dur="2s" repeatCount="indefinite" path="M 310 45 L 420 45 L 420 90" />
                    </circle>
                    {/* From Wu to merge */}
                    <circle r="5" fill="var(--value-color)" style={{ filter: "drop-shadow(0 0 4px var(--value-color))" }}>
                      <animateMotion dur="2s" repeatCount="indefinite" path="M 200 135 L 420 135 L 420 90" />
                    </circle>
                  </>
                )}

                {/* Particle for Step 4 (Gated output projected down) */}
                {animStep === 4 && (
                  <circle r="5" fill="var(--attention-color)" style={{ filter: "drop-shadow(0 0 4px var(--attention-color))" }}>
                    <animateMotion dur="2s" repeatCount="indefinite" path="M 420 90 L 520 90" />
                  </circle>
                )}

                {/* Particle for Step 5 (Completing Down projection) */}
                {animStep === 5 && (
                  <circle r="5" fill="var(--success-color)" style={{ filter: "drop-shadow(0 0 4px var(--success-color))" }}>
                    <animateMotion dur="2s" repeatCount="indefinite" path="M 520 90 L 680 90" />
                  </circle>
                )}

              </svg>
            </div>
          </div>
        )}

      </div>

      {/* Main Grid */}
      <div className="responsive-grid">
        
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
            <h3 style={{ fontSize: "1.05rem", fontFamily: "inherit", color: "var(--key-color)", display: "flex", alignItems: "center", gap: "6px" }}>
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
                rightColNames={Array.from({ length: 12 }, (_, i) => `g${i}`)}
              />
            </div>
          </div>

          {/* Up Path Projection */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <h3 style={{ fontSize: "1.05rem", fontFamily: "inherit", color: "var(--value-color)", display: "flex", alignItems: "center", gap: "6px" }}>
              <Zap className="w-4 h-4" /> Up Path: X &times; W_up &rarr; UpProj [1 &times; 12]
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
                rightColNames={Array.from({ length: 12 }, (_, i) => `u${i}`)}
              />
            </div>
          </div>

        </div>

        {/* Right: Activation, Gating and Down Projection */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* SiLU Curve Test Sandbox */}
          <div className="glass-panel" style={{ padding: "18px", display: "flex", flexDirection: "column", gap: "12px" }}>
            <h3 style={{ fontSize: "1.05rem", fontFamily: "inherit", color: "var(--success-color)" }}>
              SiLU (Swish) Activation Curve
            </h3>
            
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
              <svg width={plotWidth} height={plotHeight} style={{ backgroundColor: "var(--bg-surface)", borderRadius: "8px", border: "1px solid var(--border-color)", overflow: "visible" }}>
                {/* Axes */}
                <line x1="20" y1="110" x2="240" y2="110" stroke="var(--border-color)" strokeWidth="1.5" />
                <line x1="130" y1="10" x2="130" y2="130" stroke="var(--border-color)" strokeWidth="1.5" />
                
                {/* Curve */}
                <path d={curvePath} fill="none" stroke="var(--success-color)" strokeWidth="2.5" />
                
                {/* Active point indicator */}
                <circle cx={activePoint.x} cy={activePoint.y} r="5" fill="var(--attention-color)" stroke="var(--bg-surface)" strokeWidth="1.5" style={{ filter: "drop-shadow(0 0 4px var(--attention-glow))" }} />
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
            <h3 style={{ fontSize: "1.05rem", fontFamily: "inherit", color: "var(--text-primary)" }}>
              Gating Merge & Down Projection
            </h3>

            <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.75rem", fontFamily: "monospace" }}>
              {/* activated gate vector */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ width: "110px", color: "var(--text-muted)" }}>SiLU(GateProj):</span>
                <div style={{ display: "flex", gap: "3px", flex: 1 }}>
                  {siluGate.map((val, idx) => (
                    <div key={`swi-sg-${idx}`} style={{ flex: 1, padding: "4px 0", textAlign: "center", backgroundColor: "hsla(152, 80%, 50%, 0.05)", border: "1px solid var(--success-color)", borderRadius: "4px", color: "var(--text-primary)" }}>
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
                    <div key={`swi-up-${idx}`} style={{ flex: 1, padding: "4px 0", textAlign: "center", backgroundColor: "var(--bg-surface)", border: "1px solid var(--border-color)", borderRadius: "4px", color: "var(--text-primary)" }}>
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
                    <div key={`swi-gmult-${idx}`} style={{ flex: 1, padding: "4px 0", textAlign: "center", backgroundColor: "var(--attention-glow)", border: "1px solid var(--attention-color)", borderRadius: "4px", color: "var(--text-primary)", fontWeight: "700" }}>
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
                leftLabel="Gated Product Vector [1 x 12]"
                rightLabel="Down Weights W_down [12 x 8]"
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
