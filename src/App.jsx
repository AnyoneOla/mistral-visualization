import React, { useState, useMemo, useEffect, useRef } from "react";
import { 
  Compass, 
  Network, 
  Cpu, 
  Disc, 
  RefreshCw, 
  Key, 
  BookOpen,
  Flame,
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  Sparkles,
  Sliders,
  Brain
} from "lucide-react";

import Overview from "./components/Overview";
import EmbeddingSandbox from "./components/EmbeddingSandbox";
import RMSNormSandbox from "./components/RMSNormSandbox";
import GQASandbox from "./components/GQASandbox";
import SWASandbox from "./components/SWASandbox";
import RollingCacheSandbox from "./components/RollingCacheSandbox";
import AttentionOutputSandbox from "./components/AttentionOutputSandbox";
import ResidualNormSandbox from "./components/ResidualNormSandbox";
import RoPESandbox from "./components/RoPESandbox";
import SwiGLUSandbox from "./components/SwiGLUSandbox";
import LogitsSandbox from "./components/LogitsSandbox";

import { simulateStep, PRESETS } from "./utils/mistralSim";

// SVG Flow connector separating module blocks
const FlowConnector = ({ isSuccess }) => (
  <div className="flow-connector">
    <svg width="40" height="50" style={{ overflow: "visible" }}>
      <line x1="20" y1="0" x2="20" y2="40" className={isSuccess ? "flow-line-success" : "flow-line"} strokeWidth="2" />
      <g className="flow-pulse-node">
        <circle cx="20" cy="0" r="4" fill={isSuccess ? "var(--success-color)" : "var(--query-color)"} style={{ filter: isSuccess ? "drop-shadow(0 0 4px var(--success-glow))" : "drop-shadow(0 0 4px var(--query-glow))" }} />
      </g>
      <path d="M 16 35 L 20 40 L 24 35" fill="none" stroke={isSuccess ? "var(--success-color)" : "var(--border-color)"} strokeWidth="2.5" />
    </svg>
  </div>
);

// Advanced Sampler helper function
const sampleNextToken = (logitsData, temp, k, p) => {
  const { vocab, rawLogits } = logitsData;
  
  // 1. Scale logits by Temperature
  const scaledLogits = rawLogits.map(l => l / Math.max(0.05, temp));
  
  // 2. Filter by Top-K
  const candidates = vocab.map((word, idx) => ({
    word,
    logit: scaledLogits[idx],
    index: idx
  }));
  candidates.sort((a, b) => b.logit - a.logit);
  const kActive = candidates.slice(0, k);
  
  // 3. Filter by Top-P (Nucleus Sampling)
  const maxLogit = Math.max(...kActive.map(c => c.logit));
  const exps = kActive.map(c => Math.exp(c.logit - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(e => e / sumExps);
  
  let cumulative = 0;
  const pActive = [];
  for (let i = 0; i < kActive.length; i++) {
    cumulative += probs[i];
    pActive.push({ word: kActive[i].word, prob: probs[i] });
    if (cumulative >= p) break;
  }
  
  // Re-normalize probabilities
  const activeSum = pActive.reduce((sum, item) => sum + item.prob, 0);
  const finalProbs = pActive.map(item => item.prob / activeSum);
  
  // 4. Weighted random choice sampling
  const r = Math.random();
  let cumSum = 0;
  for (let i = 0; i < pActive.length; i++) {
    cumSum += finalProbs[i];
    if (r <= cumSum) {
      return pActive[i].word;
    }
  }
  return pActive[pActive.length - 1].word;
};

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  
  // Simulation & Sampler States
  const [presetIdx, setPresetIdx] = useState(0);
  const [generatedTokens, setGeneratedTokens] = useState([]);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(8);
  const [topP, setTopP] = useState(1.0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Playground presets
  const windowSize = 4;
  const numQHeads = 8;
  const numKVHeads = 2;

  // Compute active simulation packet
  const simData = useMemo(() => {
    return simulateStep(presetIdx, generatedTokens, windowSize, numQHeads, numKVHeads);
  }, [presetIdx, generatedTokens, windowSize, numQHeads, numKVHeads]);

  // Autoplay Timer
  const timerRef = useRef(null);
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        if (simData.hasMore) {
          const nextWord = sampleNextToken(simData.logitsData, temperature, topK, topP);
          setGeneratedTokens(prev => [...prev, nextWord]);
        } else {
          setIsPlaying(false);
        }
      }, 1500);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, simData.hasMore, simData.logitsData, temperature, topK, topP]);

  const handleStep = () => {
    if (simData.hasMore) {
      const nextWord = sampleNextToken(simData.logitsData, temperature, topK, topP);
      setGeneratedTokens(prev => [...prev, nextWord]);
    }
  };

  const handleReset = () => {
    setIsPlaying(false);
    setGeneratedTokens([]);
  };

  const handlePresetChange = (e) => {
    setPresetIdx(parseInt(e.target.value));
    setIsPlaying(false);
    setGeneratedTokens([]);
  };

  // Scroll anchor behavior on sidebar click
  const handleScrollToSection = (sectionId) => {
    setActiveTab(sectionId);
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  // Synchronize sidebar active selection during user scroll
  useEffect(() => {
    const observerOptions = {
      root: null,
      rootMargin: "-25% 0px -55% 0px",
      threshold: 0
    };

    const observerCallback = (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setActiveTab(entry.target.id);
        }
      });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);
    const sections = ["overview", "embedding", "rmsnorm1", "gqa", "rope", "swa", "cache", "oproj", "resnorm2", "swiglu", "logits"];
    
    sections.forEach(id => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  // Intersection Observer for scroll animations
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
        }
      });
    }, { threshold: 0.15, rootMargin: "0px 0px -50px 0px" });

    const els = document.querySelectorAll('.fade-in-section');
    els.forEach(el => observer.observe(el));

    return () => observer.disconnect();
  }, [generatedTokens, presetIdx, isPlaying]);

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", backgroundColor: "var(--bg-deep)" }}>
      
      {/* Global Header & Controller Banner */}
      <header style={{
        backgroundColor: "var(--bg-surface)",
        borderBottom: "1px solid var(--border-color)",
        padding: "16px 40px",
        position: "sticky",
        top: 0,
        zIndex: 100,
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.5)"
      }}>
        <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: "16px" }}>
          
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{
              width: "32px",
              height: "32px",
              borderRadius: "8px",
              backgroundColor: "var(--accent-color)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 10px var(--accent-glow)"
            }}>
              <Flame className="w-4.5 h-4.5 text-white" />
            </div>
            <div>
              <h1 style={{ fontSize: "1.1rem", fontWeight: "800", fontFamily: "Outfit", color: "var(--text-primary)" }}>
                MISTRAL 7B EXPLAINER
              </h1>
              <div style={{ fontSize: "0.6rem", fontWeight: "600", color: "var(--text-muted)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                Autoregressive Layer Walkthrough
              </div>
            </div>
          </div>

          {/* Quick Sampler Controllers in Header */}
          <div style={{ display: "flex", alignItems: "center", gap: "16px", backgroundColor: "var(--bg-card)", padding: "6px 16px", borderRadius: "8px", border: "1px solid var(--border-color)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: "600" }}>
              <Sliders className="w-3.5 h-3.5 text-accent-color" />
              <span>Sampler:</span>
            </div>
            
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem" }}>
              <span style={{ color: "var(--text-muted)" }}>T:</span>
              <span style={{ fontWeight: "700", color: "var(--success-color)", width: "30px" }}>{temperature.toFixed(1)}</span>
              <input 
                type="range" min="0.1" max="2.0" step="0.1" value={temperature} 
                onChange={(e) => setTemperature(parseFloat(e.target.value))} 
                style={{ width: "70px", height: "4px", accentColor: "var(--success-color)", cursor: "pointer" }}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem" }}>
              <span style={{ color: "var(--text-muted)" }}>K:</span>
              <span style={{ fontWeight: "700", color: "var(--query-color)", width: "15px" }}>{topK}</span>
              <input 
                type="range" min="1" max="8" step="1" value={topK} 
                onChange={(e) => setTopK(parseInt(e.target.value))} 
                style={{ width: "60px", height: "4px", accentColor: "var(--query-color)", cursor: "pointer" }}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "0.75rem" }}>
              <span style={{ color: "var(--text-muted)" }}>P:</span>
              <span style={{ fontWeight: "700", color: "var(--attention-color)", width: "25px" }}>{topP.toFixed(1)}</span>
              <input 
                type="range" min="0.1" max="1.0" step="0.1" value={topP} 
                onChange={(e) => setTopP(parseFloat(e.target.value))} 
                style={{ width: "60px", height: "4px", accentColor: "var(--attention-color)", cursor: "pointer" }}
              />
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: "500" }}>Preset Prompt:</span>
            <select 
              value={presetIdx} 
              onChange={handlePresetChange}
              style={{
                padding: "8px 12px",
                backgroundColor: "var(--bg-card)",
                border: "1px solid var(--border-color)",
                borderRadius: "6px",
                color: "var(--text-primary)",
                fontFamily: "inherit",
                fontSize: "0.85rem",
                minWidth: "200px",
                outline: "none",
                cursor: "pointer"
              }}
            >
              {PRESETS.map((preset, idx) => (
                <option key={idx} value={idx}>{preset.title}</option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <button 
              onClick={() => setIsPlaying(!isPlaying)}
              style={{
                padding: "8px 16px",
                borderRadius: "6px",
                border: "none",
                backgroundColor: isPlaying ? "hsla(0, 80%, 50%, 0.15)" : "var(--accent-color)",
                color: isPlaying ? "hsl(0, 100%, 70%)" : "white",
                fontWeight: "600",
                fontSize: "0.8rem",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isPlaying ? "Pause" : "Auto Generate"}
            </button>
            
            <button 
              onClick={handleStep}
              disabled={!simData.hasMore}
              style={{
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid var(--border-color)",
                backgroundColor: "var(--bg-card)",
                color: simData.hasMore ? "var(--text-primary)" : "var(--text-muted)",
                fontSize: "0.8rem",
                cursor: simData.hasMore ? "pointer" : "not-allowed",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s ease"
              }}
              title="Generate next token (Step)"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>

            <button 
              onClick={handleReset}
              style={{
                padding: "8px 12px",
                borderRadius: "6px",
                border: "1px solid var(--border-color)",
                backgroundColor: "var(--bg-card)",
                color: "var(--text-secondary)",
                fontSize: "0.8rem",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s ease"
              }}
              title="Reset prompt"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Dynamic Token Sequence */}
        <div style={{
          backgroundColor: "hsl(224, 71%, 3%)",
          border: "1px solid var(--border-color)",
          borderRadius: "8px",
          padding: "10px 16px",
          display: "flex",
          flexWrap: "wrap",
          gap: "6px",
          alignItems: "center"
        }}>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontWeight: "600", textTransform: "uppercase", marginRight: "8px" }}>
            Generated Sequence:
          </span>
          {simData.promptTokens.map((t, idx) => (
            <span 
              key={`p-${idx}`} 
              className={`token-badge ${simData.activeTokenIdx === t.id ? "active" : ""}`}
              style={{ padding: "4px 8px", fontSize: "0.8rem" }}
            >
              {t.text}
            </span>
          ))}
          {simData.allTokens.slice(simData.promptTokens.length).map((t, idx) => (
            <span 
              key={`g-${idx}`} 
              className={`token-badge generated ${simData.activeTokenIdx === t.id ? "active" : ""}`}
              style={{ padding: "4px 8px", fontSize: "0.8rem" }}
            >
              {t.text}
            </span>
          ))}
          {isPlaying && simData.hasMore && (
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", animation: "pulse 1s infinite", display: "inline-flex", alignItems: "center", gap: "4px" }}>
              <Sparkles className="w-3 h-3 text-accent-color animate-spin-slow" /> predicting next...
            </span>
          )}
        </div>
      </header>

      {/* Main Grid Content */}
      <div style={{ display: "grid", gridTemplateColumns: "260px 1fr", flex: 1 }} className="app-container">
        
        {/* Sidebar anchors */}
        <aside style={{
          backgroundColor: "var(--bg-surface)",
          borderRight: "1px solid var(--border-color)",
          padding: "24px 16px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          height: "calc(100vh - 120px)",
          position: "sticky",
          top: "120px",
          overflowY: "auto"
        }} className="sidebar">
          
          <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div style={{ fontSize: "0.65rem", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", paddingLeft: "8px" }}>
              Visual Flow Map
            </div>
            
            <nav style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              
              <div 
                className={`sidebar-link ${activeTab === "overview" ? "active" : ""}`}
                onClick={() => handleScrollToSection("overview")}
              >
                <Compass className="w-4 h-4" />
                <span>Overview Pipeline</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "embedding" ? "active" : ""}`}
                onClick={() => handleScrollToSection("embedding")}
              >
                <Brain className="w-4 h-4" />
                <span>1. Embedding Lookup</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "rmsnorm1" ? "active" : ""}`}
                onClick={() => handleScrollToSection("rmsnorm1")}
              >
                <Cpu className="w-4 h-4" />
                <span>2. RMSNorm 1 & Dropout</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "gqa" ? "active" : ""}`}
                onClick={() => handleScrollToSection("gqa")}
              >
                <Network className="w-4 h-4" />
                <span>3. QKV & GQA Mapping</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "rope" ? "active" : ""}`}
                onClick={() => handleScrollToSection("rope")}
              >
                <RefreshCw className="w-4 h-4" />
                <span>4. Rotary Position (RoPE)</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "swa" ? "active" : ""}`}
                onClick={() => handleScrollToSection("swa")}
              >
                <Cpu className="w-4 h-4" />
                <span>5. Sliding Window (SWA)</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "cache" ? "active" : ""}`}
                onClick={() => handleScrollToSection("cache")}
              >
                <Disc className="w-4 h-4" />
                <span>6. Rolling KV Cache</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "oproj" ? "active" : ""}`}
                onClick={() => handleScrollToSection("oproj")}
              >
                <Network className="w-4 h-4" />
                <span>7. O-Proj Output Projection</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "resnorm2" ? "active" : ""}`}
                onClick={() => handleScrollToSection("resnorm2")}
              >
                <Cpu className="w-4 h-4" />
                <span>8. Residual 1 & RMSNorm 2</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "swiglu" ? "active" : ""}`}
                onClick={() => handleScrollToSection("swiglu")}
              >
                <Key className="w-4 h-4" />
                <span>9. SwiGLU MLP Block</span>
              </div>

              <div 
                className={`sidebar-link ${activeTab === "logits" ? "active" : ""}`}
                onClick={() => handleScrollToSection("logits")}
              >
                <Sparkles className="w-4 h-4" />
                <span>10. LM Head & Softmax</span>
              </div>

            </nav>
          </div>

          <div style={{
            padding: "10px",
            backgroundColor: "var(--bg-card)",
            borderRadius: "6px",
            border: "1px solid var(--border-color)",
            fontSize: "0.7rem",
            color: "var(--text-secondary)",
            display: "flex",
            flexDirection: "column",
            gap: "4px",
            marginTop: "20px"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "4px", fontWeight: "600", color: "var(--text-primary)" }}>
              <BookOpen className="w-3 h-3 text-accent-color" />
              <span>Student Guide</span>
            </div>
            <p style={{ lineHeight: "1.3", color: "var(--text-muted)", margin: 0 }}>
              Tracing a single Mistral forward pass sequentially. Edit weights, parameters, or settings and see changes ripple through immediately.
            </p>
          </div>

        </aside>

        {/* Scrollable Canvas Viewport */}
        <main style={{ padding: "40px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "10px" }}>
          <div style={{ maxWidth: "1000px", margin: "0 auto", display: "flex", flexDirection: "column", width: "100%" }}>
            
          {/* 1. Overview */}
          <div id="overview" className="section-card fade-in-section stagger-1">
            <Overview 
                presetIdx={presetIdx}
                simData={simData}
                setActiveTab={handleScrollToSection}
              />
          </div>
          
          <FlowConnector />

          {/* 2. Embedding */}
          <div id="embedding" className="section-card fade-in-section stagger-2">
            <EmbeddingSandbox simData={simData} />
          </div>
          
          <FlowConnector />

          {/* 3. RMSNorm 1 */}
          <div id="rmsnorm1" className="section-card fade-in-section stagger-3">
            <RMSNormSandbox simData={simData} />
          </div>
          
          <FlowConnector />

          {/* 4. Attention - Multi-Group QKV */}
          <div id="gqa" className="section-card fade-in-section">
            <GQASandbox simData={simData} />
          </div>

          <FlowConnector />

          {/* 5. RoPE */}
          <div id="rope" className="section-card fade-in-section">
            <RoPESandbox simData={simData} />
          </div>

          <FlowConnector />

          {/* 6. Sliding Window Attention */}
          <div id="swa" className="section-card fade-in-section">
            <SWASandbox simData={simData} />
          </div>

          <FlowConnector />

          {/* 7. Rolling Cache */}
          <div id="cache" className="section-card fade-in-section">
            <RollingCacheSandbox 
                simData={simData}
                presetIdx={presetIdx}
                genCount={generatedTokens.length}
                setGenCount={() => {}}
                isPlaying={isPlaying}
                setIsPlaying={setIsPlaying}
              />
          </div>

          <FlowConnector />

          {/* 8. Attention Output (O-Proj) */}
          <div id="oproj" className="section-card fade-in-section">
            <AttentionOutputSandbox simData={simData} />
          </div>

          <FlowConnector />

          {/* 9. Residual & RMSNorm 2 */}
          <div id="resnorm2" className="section-card fade-in-section">
            <ResidualNormSandbox simData={simData} />
          </div>

          <FlowConnector />

          {/* 10. SwiGLU FFN */}
          <div id="swiglu" className="section-card fade-in-section">
            <SwiGLUSandbox simData={simData} />
          </div>

          <FlowConnector isSuccess={true} />

          {/* 11. Logits Output (Predict next token) */}
          <div id="logits" className="section-card fade-in-section">
              <LogitsSandbox 
                simData={simData}
                temperature={temperature}
                setTemperature={setTemperature}
                topK={topK}
                setTopK={setTopK}
                topP={topP}
                setTopP={setTopP}
              />
            </div>

          </div>
        </main>

      </div>
      
    </div>
  );
}
