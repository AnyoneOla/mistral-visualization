import React, { useState } from "react";

/**
 * InteractiveMatrixMul - Displays a mathematical matrix-vector or matrix-matrix multiplication.
 * 
 * Props:
 * - leftVal: 1D array [cols] or 2D array [rows][cols]
 * - rightVal: 2D array [rows][cols]
 * - outputVal: 1D array [cols] or 2D array [rows][cols]
 * - leftLabel: Title for the left operand
 * - rightLabel: Title for the right operand
 * - outputLabel: Title for the output operand
 * - leftColor: css variable string for styling left highlight
 * - rightColor: css variable string for styling right highlight
 * - outputColor: css variable string for styling output highlight
 * - leftRowNames: array of string labels for left matrix rows (optional)
 * - rightColNames: array of string labels for right matrix columns (optional)
 */
export default function InteractiveMatrixMul({
  leftVal,
  rightVal,
  outputVal,
  leftLabel = "Vector A",
  rightLabel = "Weight Matrix B",
  outputLabel = "Result Vector C",
  leftColor = "var(--query-color)",
  rightColor = "var(--key-color)",
  outputColor = "var(--success-color)",
  leftRowNames = null,
  rightColNames = null
}) {
  const [hoveredCell, setHoveredCell] = useState(null); // { r: number, c: number }

  // Normalize leftVal to 2D array [rows][cols]
  const isLeft1D = !Array.isArray(leftVal[0]);
  const leftMatrix = isLeft1D ? [leftVal] : leftVal;
  const leftRows = leftMatrix.length;
  const leftCols = leftMatrix[0].length;

  // rightVal is always 2D
  const rightRows = rightVal.length;
  const rightCols = rightVal[0].length;

  // Normalize outputVal to 2D array [rows][cols]
  const isOutput1D = !Array.isArray(outputVal[0]);
  const outputMatrix = isOutput1D ? [outputVal] : outputVal;

  const handleCellHover = (r, c) => {
    setHoveredCell({ r, c });
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  // Generate math explanation for the hovered cell
  const getExplanation = () => {
    if (!hoveredCell) return null;
    const { r, c } = hoveredCell;
    const leftRow = leftMatrix[r];
    const rightCol = rightVal.map(row => row[c]);
    
    let sum = 0;
    const products = leftRow.map((lVal, k) => {
      const rVal = rightCol[k];
      const prod = lVal * rVal;
      sum += prod;
      return {
        lVal,
        rVal,
        prod: parseFloat(prod.toFixed(4))
      };
    });

    const sumStr = products.map(p => `(${p.lVal} × ${p.rVal})`).join(" + ");
    const finalSum = sum.toFixed(3);

    const leftRowLabelStr = leftRowNames && leftRowNames[r] ? `"${leftRowNames[r]}"` : `Row ${r}`;
    const rightColLabelStr = rightColNames && rightColNames[c] ? `Col ${c}` : `Col ${c}`;

    return {
      leftRowLabelStr,
      rightColLabelStr,
      sumStr,
      finalSum,
      products
    };
  };

  const explanation = getExplanation();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px", width: "100%", position: "relative", paddingBottom: "220px" }}>
      
      <div className="matrix-multiply-container">
        
        {/* LEFT MATRIX / VECTOR */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>
            {leftLabel} <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>[{leftRows}×{leftCols}]</span>
          </span>
          <div style={{ 
            display: "flex", 
            flexDirection: "column", 
            gap: "4px",
            border: "1px solid var(--border-color)",
            padding: "8px",
            borderRadius: "8px",
            backgroundColor: "var(--bg-surface)"
          }}>
            {leftMatrix.map((row, rIdx) => (
              <div key={`left-r-${rIdx}`} style={{ display: "flex", gap: "4px", alignItems: "center" }}>
                {leftRowNames && leftRowNames[rIdx] && (
                  <span style={{ fontSize: "0.65rem", color: "var(--text-muted)", width: "60px", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", marginRight: "4px", textAlign: "right" }}>
                    {leftRowNames[rIdx]}
                  </span>
                )}
                {row.map((val, cIdx) => {
                  const isHighlighted = hoveredCell && hoveredCell.r === rIdx;
                  return (
                    <div
                      key={`left-cell-${rIdx}-${cIdx}`}
                      style={{
                        width: "36px",
                        height: "26px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "0.7rem",
                        fontFamily: "monospace",
                        backgroundColor: isHighlighted ? `hsla(${leftColor === "var(--query-color)" ? 190 : 272}, 95%, 50%, 0.15)` : "rgba(255,255,255,0.02)",
                        border: `1px solid ${isHighlighted ? leftColor : "var(--border-color)"}`,
                        borderRadius: "4px",
                        color: isHighlighted ? "var(--text-primary)" : "var(--text-secondary)",
                        transition: "all 0.15s ease"
                      }}
                      title={`${leftLabel}[${rIdx}, ${cIdx}] = ${val}`}
                    >
                      {val.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* MULTIPLICATION SIGN */}
        <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "var(--text-muted)", padding: "0 4px" }}>&times;</div>

        {/* RIGHT WEIGHT MATRIX */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>
            {rightLabel} <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>[{rightRows}×{rightCols}]</span>
          </span>
          <div style={{ 
            display: "flex", 
            flexDirection: "column", 
            gap: "4px",
            border: "1px solid var(--border-color)",
            padding: "8px",
            borderRadius: "8px",
            backgroundColor: "var(--bg-surface)"
          }}>
            {rightVal.map((row, rIdx) => (
              <div key={`right-r-${rIdx}`} style={{ display: "flex", gap: "4px" }}>
                {row.map((val, cIdx) => {
                  const isHighlighted = hoveredCell && hoveredCell.c === cIdx;
                  return (
                    <div
                      key={`right-cell-${rIdx}-${cIdx}`}
                      style={{
                        width: "36px",
                        height: "24px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "0.65rem",
                        fontFamily: "monospace",
                        backgroundColor: isHighlighted ? `hsla(38, 95%, 55%, 0.12)` : "rgba(255,255,255,0.01)",
                        border: `1px solid ${isHighlighted ? rightColor : "var(--border-color)"}`,
                        borderRadius: "3px",
                        color: isHighlighted ? "var(--text-primary)" : "var(--text-muted)",
                        transition: "all 0.15s ease"
                      }}
                      title={`${rightLabel}[${rIdx}, ${cIdx}] = ${val}`}
                    >
                      {val.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            ))}
            {rightColNames && (
              <div style={{ display: "flex", gap: "4px", marginTop: "2px" }}>
                {rightColNames.map((name, idx) => (
                  <span key={`rc-n-${idx}`} style={{ width: "36px", fontSize: "0.55rem", color: "var(--text-muted)", textAlign: "center", display: "inline-block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={name}>
                    {name}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* EQUALS SIGN */}
        <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "var(--text-muted)", padding: "0 4px" }}>=</div>

        {/* OUTPUT MATRIX / VECTOR */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>
            {outputLabel} <span style={{ color: "var(--text-muted)", fontFamily: "monospace" }}>[{leftRows}×{rightCols}]</span>
          </span>
          <div style={{ 
            display: "flex", 
            flexDirection: "column", 
            gap: "4px",
            border: "1px solid var(--border-color)",
            padding: "8px",
            borderRadius: "8px",
            backgroundColor: "var(--bg-surface)"
          }}>
            {outputMatrix.map((row, rIdx) => (
              <div key={`out-r-${rIdx}`} style={{ display: "flex", gap: "4px" }}>
                {row.map((val, cIdx) => {
                  const isHovered = hoveredCell && hoveredCell.r === rIdx && hoveredCell.c === cIdx;
                  return (
                    <div
                      key={`out-cell-${rIdx}-${cIdx}`}
                      onMouseEnter={() => handleCellHover(rIdx, cIdx)}
                      onMouseLeave={handleMouseLeave}
                      style={{
                        width: "38px",
                        height: "26px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: "0.7rem",
                        fontFamily: "monospace",
                        fontWeight: "600",
                        backgroundColor: isHovered ? outputColor : "rgba(255,255,255,0.03)",
                        border: `1px solid ${isHovered ? outputColor : "var(--border-color)"}`,
                        borderRadius: "4px",
                        color: isHovered ? "var(--bg-deep)" : "var(--text-primary)",
                        cursor: "crosshair",
                        boxShadow: isHovered ? `0 0 10px ${outputColor}` : "none",
                        transition: "all 0.15s ease"
                      }}
                    >
                      {val.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* DETAILED EQUATION / MATH POPUP WRAPPER FOR HOVER STABILITY */}
      <div style={{ 
        position: "absolute",
        bottom: "0",
        left: "0",
        height: "210px", 
        display: "flex", 
        flexDirection: "column", 
        justifyContent: "center", 
        width: "100%",
        boxSizing: "border-box"
      }}>
        {hoveredCell && explanation ? (
          <div 
            className="math-popover animate-glow" 
            style={{ 
              alignSelf: "center", 
              width: "100%", 
              maxWidth: "700px", 
              border: `1px solid ${outputColor}`,
              boxShadow: `0 0 12px ${outputColor}22`,
              backgroundColor: "var(--bg-card)",
              padding: "14px 16px",
              borderRadius: "10px",
              fontSize: "0.8rem",
              boxSizing: "border-box"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", borderBottom: "1px solid var(--border-color)", paddingBottom: "6px" }}>
              <span style={{ fontWeight: "700", color: "var(--text-primary)" }}>Dot Product Calculation</span>
              <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                {explanation.leftRowLabelStr} &times; {explanation.rightColLabelStr}
              </span>
            </div>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              
              {/* Show equation */}
              <div style={{ overflowX: "auto", paddingBottom: "4px" }}>
                <code style={{ fontSize: "0.75rem", whiteSpace: "nowrap", color: "var(--text-secondary)" }}>
                  C[{hoveredCell.r},{hoveredCell.c}] = {explanation.sumStr}
                </code>
              </div>
              
              {/* Show value expansion */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 12px", fontSize: "0.7rem", color: "var(--text-muted)", backgroundColor: "var(--bg-surface)", padding: "6px 10px", borderRadius: "6px", border: "1px solid rgba(255,255,255,0.03)" }}>
                {explanation.products.map((p, idx) => (
                  <div key={`p-detail-${idx}`} style={{ display: "flex", gap: "4px" }}>
                    <span style={{ color: leftColor }}>{p.lVal.toFixed(2)}</span>
                    <span>&times;</span>
                    <span style={{ color: rightColor }}>{p.rVal.toFixed(2)}</span>
                    <span>=</span>
                    <span style={{ color: "var(--text-primary)", fontWeight: "500" }}>{p.prod.toFixed(3)}</span>
                  </div>
                ))}
              </div>

              {/* Sum Result */}
              <div style={{ fontSize: "0.85rem", fontWeight: "700", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "6px", marginTop: "2px" }}>
                <span>Sum Result:</span>
                <span style={{ color: outputColor, fontSize: "1.05rem", fontFamily: "monospace" }}>{explanation.finalSum}</span>
              </div>

            </div>
          </div>
        ) : (
          <div style={{ textAlign: "center", fontSize: "0.75rem", color: "var(--text-muted)", fontStyle: "italic", padding: "20px 0" }}>
            💡 Hover over any cell in the output matrix to inspect the row-column dot product calculations.
          </div>
        )}
      </div>

    </div>
  );
}
