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
  ArrowRight,
  Sparkles,
  Sliders,
  Brain,
  Sun,
  Moon,
  ChevronDown
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
  
  // Theme state
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem("theme") || "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Favicon dynamic rendering effect
  useEffect(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext("2d");
    
    // Draw Mistral 3D logo facets
    ctx.fillStyle = "#FF531A";
    ctx.beginPath();
    ctx.moveTo(2.5, 27.5);
    ctx.lineTo(2.5, 6.25);
    ctx.lineTo(8.75, 2.5);
    ctx.lineTo(8.75, 23.75);
    ctx.fill();
    
    ctx.fillStyle = "#E14614";
    ctx.beginPath();
    ctx.moveTo(8.75, 23.75);
    ctx.lineTo(8.75, 2.5);
    ctx.lineTo(16.25, 13.75);
    ctx.lineTo(16.25, 28.75);
    ctx.fill();
    
    ctx.fillStyle = "#FF531A";
    ctx.beginPath();
    ctx.moveTo(16.25, 28.75);
    ctx.lineTo(16.25, 13.75);
    ctx.lineTo(23.75, 2.5);
    ctx.lineTo(23.75, 23.75);
    ctx.fill();
    
    ctx.fillStyle = "#E14614";
    ctx.beginPath();
    ctx.moveTo(23.75, 23.75);
    ctx.lineTo(23.75, 2.5);
    ctx.lineTo(30, 6.25);
    ctx.lineTo(30, 27.5);
    ctx.fill();
    
    const link = document.querySelector("link[rel*='icon']") || document.createElement("link");
    link.type = "image/x-icon";
    link.rel = "shortcut icon";
    link.href = canvas.toDataURL("image/x-icon");
    document.getElementsByTagName("head")[0].appendChild(link);
  }, []);

  // Simulation & Sampler States
  const [presetIdx, setPresetIdx] = useState(0);
  const [generatedTokens, setGeneratedTokens] = useState([]);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(8);
  const [topP, setTopP] = useState(1.0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Playground presets
  const [windowSize, setWindowSize] = useState(4);
  const [attentionType, setAttentionType] = useState("GQA"); // MHA, GQA, MQA
  const numQHeads = 8;
  const numKVHeads = attentionType === "MHA" ? 8 : attentionType === "MQA" ? 1 : 2;

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
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden", backgroundColor: "var(--bg-deep)" }}>
      
      {/* Global Header & Controller Banner */}
      <header style={{
        backgroundColor: "var(--bg-surface)",
        borderBottom: "2px solid var(--border-color)",
        padding: "12px 40px 10px 40px",
        position: "sticky",
        top: 0,
        zIndex: 100,
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.04)"
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          
          {/* Logo + Title block */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {/* Exact Mistral 3D block-style 'M' logo */}
            <svg viewBox="0 0 128 128" width="22" height="22" style={{ overflow: "visible" }}>
              <g>
                <polygon points="10,110 10,25 35,10 35,95" fill="#FF531A" />
                <polygon points="35,95 35,10 65,55 65,115" fill="#E14614" />
                <polygon points="65,115 65,55 95,10 95,95" fill="#FF531A" />
                <polygon points="95,95 95,10 120,25 120,110" fill="#E14614" />
              </g>
            </svg>
            <div>
              <h1 style={{ fontSize: "1rem", fontWeight: "700", fontFamily: "inherit", color: "var(--text-primary)", letterSpacing: "-0.01em", margin: 0 }}>
                Mistral 7B Explainer
              </h1>
            </div>
          </div>

          {/* Theme Mode Toggle (only) in top-right header */}
          <button 
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            style={{
              padding: "6px 12px",
              borderRadius: "8px",
              border: "1px solid var(--border-color)",
              backgroundColor: "var(--bg-surface)",
              color: "var(--text-secondary)",
              fontSize: "0.7rem",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.15s ease"
            }}
            title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            {theme === "dark" ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Dynamic Token Sequence - frozen in header */}
        <div style={{
          backgroundColor: "var(--bg-surface)",
          border: "1px solid var(--border-color)",
          borderRadius: "8px",
          padding: "6px 12px",
          display: "flex",
          flexWrap: "wrap",
          gap: "6px",
          alignItems: "center"
        }}>
          <span style={{ fontSize: "0.6rem", color: "var(--text-secondary)", fontWeight: "700", textTransform: "uppercase", marginRight: "6px", letterSpacing: "0.05em", fontFamily: "monospace" }}>
            GENERATED SEQUENCE:
          </span>
          {simData.promptTokens.map((t, idx) => (
            <span 
              key={`p-${idx}`} 
              className={`token-badge ${simData.activeTokenIdx === t.id ? "active" : ""}`}
              style={{ padding: "2px 5px", fontSize: "0.7rem" }}
            >
              {t.text}
            </span>
          ))}
          {simData.allTokens.slice(simData.promptTokens.length).map((t, idx) => (
            <span 
              key={`g-${idx}`} 
              className={`token-badge generated ${simData.activeTokenIdx === t.id ? "active" : ""}`}
              style={{ padding: "2px 5px", fontSize: "0.7rem" }}
            >
              {t.text}
            </span>
          ))}
          {isPlaying && simData.hasMore && (
            <span style={{ fontSize: "0.65rem", color: "var(--text-secondary)", animation: "pulse 1.2s infinite", display: "inline-flex", alignItems: "center", gap: "4px" }}>
              <Sparkles className="w-3 h-3 text-cyan-600 animate-pulse" /> predicting...
            </span>
          )}
        </div>
      </header>

      {/* Main Grid Content */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }} className="app-container">
        
        {/* Sidebar anchors */}
        <aside style={{
          backgroundColor: "var(--bg-surface)",
          borderRight: "2px solid var(--border-color)",
          padding: "20px 14px",
          display: "flex",
          flexDirection: "column",
          gap: "24px",
          width: "280px",
          flexShrink: 0,
          overflowY: "auto"
        }} className="sidebar">

          {/* CONTROL CONSOLE - Pinned in Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: "16px", borderBottom: "1px solid var(--border-color)", paddingBottom: "20px" }}>
            <div style={{ fontSize: "0.65rem", fontWeight: "700", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Control Console
            </div>

            {/* Presets dropdown */}
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <span style={{ fontSize: "0.7rem", color: "var(--text-secondary)", fontWeight: "600" }}>Preset Prompt:</span>
              <div style={{ position: "relative" }}>
                <select 
                  value={presetIdx} 
                  onChange={handlePresetChange}
                  style={{
                    width: "100%",
                    padding: "8px 30px 8px 10px",
                    backgroundColor: "var(--bg-card)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "8px",
                    color: "var(--text-primary)",
                    fontFamily: "inherit",
                    fontSize: "0.75rem",
                    outline: "none",
                    cursor: "pointer",
                    appearance: "none",
                    WebkitAppearance: "none",
                    MozAppearance: "none"
                  }}
                >
                  {PRESETS.map((preset, idx) => (
                    <option key={idx} value={idx}>{preset.title}</option>
                  ))}
                </select>
                <div style={{ position: "absolute", right: "10px", top: "50%", transform: "translateY(-50%)", pointerEvents: "none", color: "var(--text-secondary)", display: "flex", alignItems: "center" }}>
                  <ChevronDown className="w-4 h-4" />
                </div>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              <button 
                onClick={() => setIsPlaying(!isPlaying)}
                style={{
                  flex: "1 1 auto",
                  padding: "8px 14px",
                  borderRadius: "8px",
                  border: "none",
                  backgroundColor: isPlaying ? "var(--bg-surface)" : "var(--accent-color)",
                  color: isPlaying ? "var(--accent-color)" : "white",
                  fontWeight: "800",
                  fontSize: "0.75rem",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  cursor: "pointer",
                  transition: "all 0.15s ease"
                }}
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5 fill-current" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                <span>{isPlaying ? "Pause" : "Auto Gen"}</span>
              </button>
              
              <button 
                onClick={handleStep}
                disabled={!simData.hasMore}
                style={{
                  padding: "8px 12px",
                  borderRadius: "8px",
                  border: "1px solid var(--border-color)",
                  backgroundColor: "var(--bg-card)",
                  color: simData.hasMore ? "var(--text-primary)" : "var(--text-muted)",
                  cursor: simData.hasMore ? "pointer" : "not-allowed",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "all 0.15s ease"
                }}
                title="Generate next token"
              >
                <ArrowRight className="w-3.5 h-3.5" />
              </button>

              <button 
                onClick={handleReset}
                style={{
                  padding: "8px 12px",
                  borderRadius: "8px",
                  border: "1px solid var(--border-color)",
                  backgroundColor: "var(--bg-card)",
                  color: "var(--text-secondary)",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "all 0.15s ease"
                }}
                title="Reset generation"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Sliders */}
            <div style={{ display: "flex", flexDirection: "column", gap: "10px", backgroundColor: "var(--bg-surface)", padding: "10px", border: "1px solid var(--border-color)", borderRadius: "8px" }}>
              
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", fontWeight: "600", color: "var(--text-secondary)" }}>
                  <span>Temp (T):</span>
                  <span style={{ fontFamily: "monospace", color: "var(--accent-color)" }}>{temperature.toFixed(1)}</span>
                </div>
                <input 
                  type="range" min="0.1" max="2.0" step="0.1" value={temperature} 
                  onChange={(e) => setTemperature(parseFloat(e.target.value))} 
                  className="custom-slider"
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", fontWeight: "600", color: "var(--text-secondary)" }}>
                  <span>Top-K (K):</span>
                  <span style={{ fontFamily: "monospace", color: "var(--key-color)" }}>{topK}</span>
                </div>
                <input 
                  type="range" min="1" max="8" step="1" value={topK} 
                  onChange={(e) => setTopK(parseInt(e.target.value))} 
                  className="custom-slider"
                />
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", fontWeight: "600", color: "var(--text-secondary)" }}>
                  <span>Top-P (P):</span>
                  <span style={{ fontFamily: "monospace", color: "var(--success-color)" }}>{topP.toFixed(1)}</span>
                </div>
                <input 
                  type="range" min="0.1" max="1.0" step="0.1" value={topP} 
                  onChange={(e) => setTopP(parseFloat(e.target.value))} 
                  className="custom-slider"
                />
              </div>

            </div>
          </div>

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
        <main style={{ flex: 1, padding: "40px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "10px" }} className="content-pane">
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
            <GQASandbox 
              simData={simData} 
              attentionType={attentionType}
              setAttentionType={setAttentionType}
            />
          </div>

          <FlowConnector />

          {/* 5. RoPE */}
          <div id="rope" className="section-card fade-in-section">
            <RoPESandbox simData={simData} />
          </div>

          <FlowConnector />

          {/* 6. Sliding Window Attention */}
          <div id="swa" className="section-card fade-in-section">
            <SWASandbox 
              simData={simData} 
              windowSize={windowSize}
              setWindowSize={setWindowSize}
            />
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
