import React from "react";
import { Disc, ArrowRight, Activity, Cpu } from "lucide-react";

export default function RollingCacheSandbox({ 
  simData, 
  presetIdx, 
  genCount, 
  setGenCount, 
  isPlaying, 
  setIsPlaying 
}) {
  const rollingCache = simData?.rollingCache || { windowSize: 4, slots: [], activeSlot: 0 };
  const windowSize = rollingCache.windowSize;
  const slots = rollingCache.slots || [];
  const activeSlot = rollingCache.activeSlot ?? 0;

  // Helper to draw circular ring sectors
  const getCircularPosition = (slotIdx, totalSlots, radius = 70) => {
    const angle = (slotIdx * 2 * Math.PI) / totalSlots - Math.PI / 2; // offset by -90deg to start top
    return {
      x: radius * Math.cos(angle),
      y: radius * Math.sin(angle)
    };
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
      
      {/* Overview Block */}
      <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "12px", borderLeft: "4px solid var(--value-color)" }}>
        <h2 style={{ fontSize: "1.4rem", fontFamily: "inherit", display: "flex", alignItems: "center", gap: "10px" }}>
          <Disc className="w-6 h-6 text-purple-400 animate-spin-slow" />
          Step 6: Rolling Buffer Key-Value Cache
        </h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: "1.5", margin: 0 }}>
          Because attention in Mistral 7B only spans a fixed window of size $W$, the model only needs to cache key-value states for the most recent $W$ tokens. Older tokens are never read again, so their cache entries can be evicted.
          Mistral stores the KV cache in a **Rolling Buffer (circular queue)**. The KV vectors for token $i$ are written directly to slot index $i$ mod $W$ in the cache. At step $i$, it overwrites the slot previously occupied by token $i - W$.
        </p>
      </div>

      {/* Main Interactive Grid */}
      <div className="responsive-grid">
        
        {/* Left Side: Circular Buffer Visualizer */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "16px", position: "relative" }}>
          <h3 style={{ fontSize: "1.1rem", alignSelf: "flex-start", fontFamily: "inherit", color: "var(--text-primary)" }}>
            Circular KV Cache Ring (Size W = {windowSize})
          </h3>
          
          <div style={{ 
            position: "relative", 
            width: "280px", 
            height: "280px", 
            backgroundColor: "var(--bg-surface)",
            borderRadius: "50%",
            border: "1px dashed var(--border-color)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "inset 0 0 25px rgba(0,0,0,0.7)"
          }}>
            
            <svg width="280" height="280" style={{ position: "absolute", top: 0, left: 0, overflow: "visible" }}>
              {/* Outer boundary circles */}
              <circle cx="140" cy="140" r="95" fill="none" stroke="var(--border-color)" strokeWidth="1.5" />
              <circle cx="140" cy="140" r="55" fill="none" stroke="var(--border-color)" strokeWidth="1" />

              {/* Sector dividers */}
              {Array.from({ length: windowSize }).map((_, idx) => {
                const innerPt = getCircularPosition(idx, windowSize, 55);
                const outerPt = getCircularPosition(idx, windowSize, 95);
                return (
                  <line
                    key={`ring-div-${idx}`}
                    x1={140 + innerPt.x}
                    y1={140 + innerPt.y}
                    x2={140 + outerPt.x}
                    y2={140 + outerPt.y}
                    stroke="var(--border-color)"
                    strokeWidth="1.5"
                  />
                );
              })}

              {/* Pointer path overlay */}
              {(() => {
                const angle = (activeSlot * 2 * Math.PI) / windowSize - Math.PI / 2;
                const arrowPt1 = getCircularPosition(activeSlot, windowSize, 40);
                const arrowPt2 = getCircularPosition(activeSlot, windowSize, 52);
                
                return (
                  <g style={{ transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)" }}>
                    <path
                      d={`M 140 140 
                          L ${140 + getCircularPosition(activeSlot, windowSize, 95).x} ${140 + getCircularPosition(activeSlot, windowSize, 95).y} 
                          A 95 95 0 0 1 ${140 + getCircularPosition(activeSlot + 1, windowSize, 95).x} ${140 + getCircularPosition(activeSlot + 1, windowSize, 95).y} 
                          Z`}
                      fill="var(--value-glow)"
                      opacity="0.35"
                    />
                    <line
                      x1={140 + arrowPt1.x}
                      y1={140 + arrowPt1.y}
                      x2={140 + arrowPt2.x}
                      y2={140 + arrowPt2.y}
                      stroke="var(--value-color)"
                      strokeWidth="2.5"
                    />
                    <circle cx={140 + arrowPt2.x} cy={140 + arrowPt2.y} r="3" fill="var(--value-color)" style={{ filter: "drop-shadow(0 0 3px var(--value-color))" }} />
                  </g>
                );
              })()}
            </svg>

            {/* Middle Circle Core */}
            <div style={{
              width: "100px",
              height: "100px",
              borderRadius: "50%",
              backgroundColor: "var(--bg-deep)",
              border: "1px solid var(--border-color)",
              zIndex: 10,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              textAlign: "center"
            }}>
              <Activity className="w-4 h-4 text-value-color animate-pulse-slow" style={{ marginBottom: "2px" }} />
              <div style={{ fontSize: "0.55rem", color: "var(--text-muted)", textTransform: "uppercase" }}>Slot Pointer</div>
              <div style={{ fontSize: "1.0rem", fontWeight: "700", fontFamily: "inherit", color: "var(--value-color)" }}>
                {simData.activeTokenIdx % windowSize}
              </div>
            </div>

            {/* Slot Labels */}
            {slots.map((slot, idx) => {
              const textPt = getCircularPosition(idx + 0.5, windowSize, 75);
              const isActive = activeSlot === idx;
              
              return (
                <div
                  key={`ring-lbl-${idx}`}
                  style={{
                    position: "absolute",
                    left: `${140 + textPt.x}px`,
                    top: `${140 + textPt.y}px`,
                    transform: "translate(-50%, -50%)",
                    zIndex: 15,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    transition: "all 0.3s ease"
                  }}
                >
                  <span style={{ 
                    fontSize: "0.55rem", 
                    color: isActive ? "var(--value-color)" : "var(--text-muted)",
                    fontWeight: isActive ? "700" : "500" 
                  }}>
                    Slot {idx}
                  </span>
                  
                  <span style={{ 
                    fontSize: "0.75rem", 
                    fontWeight: "700",
                    color: isActive ? "white" : "var(--text-secondary)",
                    maxWidth: "50px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap"
                  }}>
                    {slot.storedToken ? `"${slot.storedToken.token}"` : "empty"}
                  </span>
                </div>
              );
            })}

          </div>
        </div>

        {/* Right Side: Eviction Logger & Detail Table */}
        <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          
          {/* Eviction Tracker */}
          <div className="glass-panel" style={{ padding: "18px" }}>
            <h3 style={{ fontSize: "1rem", marginBottom: "12px", fontFamily: "inherit", color: "var(--value-color)" }}>
              Circular Write Eviction Logger
            </h3>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "0.85rem" }}>
              {slots[activeSlot]?.storedToken ? (
                <div>
                  <div style={{ color: "var(--text-secondary)", marginBottom: "6px" }}>
                    Writing active token at pos <strong>{simData.activeTokenIdx}</strong>:
                  </div>
                  
                  <div style={{ 
                    padding: "12px", 
                    backgroundColor: "hsla(272, 85%, 65%, 0.05)", 
                    border: "1px dashed var(--value-color)", 
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}>
                    {slots[activeSlot].evictedToken ? (
                      <>
                        <span style={{ color: "hsl(0, 75%, 65%)", fontWeight: "700", textDecoration: "line-through" }}>
                          "{slots[activeSlot].evictedToken.token}"
                        </span>
                        <ArrowRight className="w-4 h-4 text-text-muted" />
                        <span style={{ color: "var(--success-color)", fontWeight: "700" }}>
                          "{slots[activeSlot].storedToken.token}"
                        </span>
                        <span style={{ fontSize: "0.7rem", color: "hsl(0, 75%, 65%)", marginLeft: "auto", fontWeight: "700" }}>
                          EVICTED!
                        </span>
                      </>
                    ) : (
                      <>
                        <span style={{ color: "var(--text-muted)" }}>Empty</span>
                        <ArrowRight className="w-4 h-4 text-text-muted" />
                        <span style={{ color: "var(--success-color)", fontWeight: "700" }}>
                          "{slots[activeSlot].storedToken.token}"
                        </span>
                        <span style={{ fontSize: "0.7rem", color: "var(--success-color)", marginLeft: "auto", fontWeight: "700" }}>
                          FIRST WRITE
                        </span>
                      </>
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ color: "var(--text-muted)", fontStyle: "italic" }}>
                  Generate a token to see rolling write buffer updates.
                </div>
              )}

              <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", lineHeight: "1.3", margin: 0 }}>
                Since the window W is 4, token index <strong>i</strong> overwrites slot index <strong>i mod 4</strong>. 
                This prevents unbounded memory growth as prompt size scales.
              </p>
            </div>
          </div>

          {/* Cached Vectors Inspector */}
          <div className="glass-panel" style={{ padding: "18px" }}>
            <h3 style={{ fontSize: "1rem", marginBottom: "12px", fontFamily: "inherit", color: "var(--key-color)" }}>
              Active Slots KV cache
            </h3>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "160px", overflowY: "auto", paddingRight: "4px" }}>
              {slots.map((slot, idx) => {
                const isActive = activeSlot === idx;
                const kVals = simData.gqa.kVectors[slot.storedToken?.pos]?.[0] || [];
                const vVals = simData.gqa.vVectors[slot.storedToken?.pos]?.[0] || [];
                
                return (
                  <div 
                    key={`rolling-vector-row-${idx}`}
                    style={{
                      padding: "8px 12px",
                      backgroundColor: isActive ? "var(--value-glow)" : "var(--bg-surface)",
                      borderRadius: "8px",
                      border: `1px solid ${isActive ? "var(--value-color)" : "var(--border-color)"}`,
                      fontSize: "0.75rem",
                      display: "flex",
                      flexDirection: "column",
                      gap: "6px"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "700", color: isActive ? "var(--value-color)" : "white" }}>
                      <span>Slot {idx} {slot.storedToken ? `("${slot.storedToken.token}")` : "(empty)"}</span>
                      <span style={{ color: "var(--text-muted)", fontSize: "0.65rem" }}>pos: {slot.storedToken?.pos ?? "-"}</span>
                    </div>
                    
                    {slot.storedToken && (
                      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <span style={{ color: "var(--key-color)", fontSize: "0.65rem", width: "12px", fontWeight: "700" }}>K:</span>
                          <div style={{ display: "flex", gap: "2px", flex: 1 }}>
                            {kVals.slice(0, 4).map((v, ki) => (
                              <div key={`k-val-${idx}-${ki}`} style={{ flex: 1, padding: "2px 0", textAlign: "center", fontSize: "0.6rem", fontFamily: "monospace", backgroundColor: "rgba(251, 191, 36, 0.1)", border: "1px solid var(--key-color)", borderRadius: "3px", color: "var(--text-primary)" }}>
                                {v.toFixed(2)}
                              </div>
                            ))}
                          </div>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <span style={{ color: "var(--value-color)", fontSize: "0.65rem", width: "12px", fontWeight: "700" }}>V:</span>
                          <div style={{ display: "flex", gap: "2px", flex: 1 }}>
                            {vVals.slice(0, 4).map((v, vi) => (
                              <div key={`v-val-${idx}-${vi}`} style={{ flex: 1, padding: "2px 0", textAlign: "center", fontSize: "0.6rem", fontFamily: "monospace", backgroundColor: "rgba(167, 139, 250, 0.1)", border: "1px solid var(--value-color)", borderRadius: "3px", color: "var(--text-primary)" }}>
                                {v.toFixed(2)}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
