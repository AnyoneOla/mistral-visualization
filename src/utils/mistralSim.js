// mistralSim.js - High-fidelity mathematical simulation engine for Mistral 7B Explainer

// Seeded pseudo-random generator for realistic but deterministic values
function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function stringHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

// Generate a deterministic vector of length 'dim'
export function generateVector(tokenIdx, key, dim = 8, magnitude = 1.0) {
  const vec = [];
  let seed = tokenIdx * 17;
  for (let i = 0; i < key.length; i++) {
    seed += key.charCodeAt(i) * Math.pow(7, i % 3);
  }
  for (let i = 0; i < dim; i++) {
    const r = seededRandom(seed++) * 2 - 1; // -1 to 1
    vec.push(parseFloat((r * magnitude).toFixed(3)));
  }
  return vec;
}

// Generate a deterministic weight matrix of size Rows x Cols
export function generateMatrix(key, rows, cols, magnitude = 0.5) {
  const matrix = [];
  let seed = 0;
  for (let i = 0; i < key.length; i++) {
    seed += key.charCodeAt(i) * Math.pow(11, i % 4);
  }
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const val = seededRandom(seed++) * 2 - 1; // -1 to 1
      row.push(parseFloat((val * magnitude).toFixed(3)));
    }
    matrix.push(row);
  }
  return matrix;
}

// Vector-matrix multiplication: [1 x R] * [R x C] = [1 x C]
export function vectorMatrixMul(vec, matrix) {
  const R = vec.length;
  const C = matrix[0].length;
  const out = [];
  for (let c = 0; c < C; c++) {
    let sum = 0;
    for (let r = 0; r < R; r++) {
      sum += vec[r] * matrix[r][c];
    }
    out.push(parseFloat(sum.toFixed(3)));
  }
  return out;
}

import { PRESETS_DATA } from './presetsData.js';

export const PRESETS = PRESETS_DATA;

// Custom simulated BPE sub-word tokenizer
export function tokenizeText(text) {
  const words = text.split(/(\s+|-|Query|Attention|Mistral|RoPE|SwiGLU|RMSNorm|7B|KV|W=\d+)/g).filter(x => x !== "" && x !== undefined);
  return words.map((str, idx) => ({
    id: idx,
    text: str,
    isSpace: /^\s+$/.test(str)
  }));
}

export function rotate2D(x, y, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [
    parseFloat((x * cos - y * sin).toFixed(4)),
    parseFloat((x * sin + y * cos).toFixed(4))
  ];
}

const silu = (x) => x / (1 + Math.exp(-x));

// Custom embedding vocabulary mapping for the lookup visualizer
export const EMBEDDING_VOCAB = Array.from(new Set([
  "Mistral", "7B", "uses", "Grouped-Query", "Attention", "GQA", "to", "speed", 
  "up", "inference", "and", "reduce", "memory", "bandwidth", "during", "token", 
  "generation", "Sliding", "Window", "limits", "the", "range", "fixed", "window", 
  "size", "W", "allowing", "layers", "build", "larger", "context", "Rolling", 
  "Buffer", "Cache", "stores", "key-value", "keys", "in", "circular", "ring", 
  "buffer", "overwrites", "older", "caches", "Rotary", "Position", "Embeddings", 
  "RoPE", "rotate", "query", "key", "vectors", "2D", "planes", "angles", 
  "proportional", "absolute", "positions", "text", "SwiGLU", "activation", 
  "replaces", "traditional", "GELU", "feed-forward", "networks", "gated", 
  "mechanism", "improves", "learning", "capacity", "RMSNorm", "normalizes", 
  "activation", "without", "calculating", "mean", "overhead", "stable", 
  "Decoder-only", "transformers", "predicted", "appended", "inputs", "feed", 
  "back", "network", "loop", "decoder", "dot", "product", "decays", "relative", 
  "distance", "ensuring", "closer", "tokens", "prioritized", "gating", "valve", 
  "controller"
]));

// Pre-populate EMBEDDING_VOCAB with all possible words from presets to guarantee completeness
PRESETS_DATA.forEach(preset => {
  const pTokens = tokenizeText(preset.prompt);
  pTokens.forEach(t => {
    if (t.text.trim() && !EMBEDDING_VOCAB.includes(t.text.trim())) EMBEDDING_VOCAB.push(t.text.trim());
  });
  preset.completions.forEach(comp => {
    comp.words.forEach(w => {
      const clean = w.trim();
      if (clean && !EMBEDDING_VOCAB.includes(clean)) EMBEDDING_VOCAB.push(clean);
    });
  });
});

// Sort vocabulary alphabetically
EMBEDDING_VOCAB.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));

export function rmsNorm(vec, gamma, eps = 1e-5) {
  const d = vec.length;
  let sumSq = 0;
  for (let i = 0; i < d; i++) {
    sumSq += vec[i] * vec[i];
  }
  const rms = Math.sqrt(sumSq / d + eps);
  const normalized = vec.map(v => parseFloat((v / rms).toFixed(3)));
  const output = normalized.map((v, i) => parseFloat((v * gamma[i]).toFixed(3)));
  return { rms, normalized, output };
}

// Main function to run the Mistral simulation at a specific generation step
// Now accepts the dynamic array of generated token strings
export function simulateStep(presetIdx, generatedTokenStrings = [], windowSize = 4, numQHeads = 8, numKVHeads = 2) {
  const preset = PRESETS[presetIdx];
  const promptTokens = tokenizeText(preset.prompt);
  
  // Guard against generatedTokenStrings being a number
  const tokenStrings = Array.isArray(generatedTokenStrings) ? generatedTokenStrings : [];
  
  // Map generated strings to token format
  const genTokens = tokenStrings.map((str, idx) => ({
    id: promptTokens.length + idx,
    text: str,
    isSpace: /^\s+$/.test(str)
  }));
  
  const allTokens = [...promptTokens, ...genTokens];
  const seqLength = allTokens.length;
  const activeTokenIdx = seqLength - 1;
  const activeToken = allTokens[activeTokenIdx];

  // Dynamically add generated token to dictionary if missing (handles wild hallucinations)
  const cleanActiveText = activeToken.text.trim();
  if (cleanActiveText && !EMBEDDING_VOCAB.includes(cleanActiveText)) {
    EMBEDDING_VOCAB.push(cleanActiveText);
    EMBEDDING_VOCAB.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
  }

  // Base parameters
  const hiddenDim = 8;
  const headDim = 4;
  const qToKvRatio = numQHeads / numKVHeads;

  // Generate deterministic weight matrices
  const W_Q = Array.from({ length: numQHeads }, (_, h) => generateMatrix(`W_Q_head_${h}`, hiddenDim, headDim, 0.6));
  const W_K = Array.from({ length: numKVHeads }, (_, h) => generateMatrix(`W_K_head_${h}`, hiddenDim, headDim, 0.6));
  const W_V = Array.from({ length: numKVHeads }, (_, h) => generateMatrix(`W_V_head_${h}`, hiddenDim, headDim, 0.6));
  const W_O = generateMatrix("W_O_output", numQHeads * headDim, hiddenDim, 0.5);

  // 1. Embeddings Simulation & Lookup Matrix
  let embedLookupIdx = EMBEDDING_VOCAB.findIndex(w => w.toLowerCase() === cleanActiveText.toLowerCase());
  if (embedLookupIdx === -1) {
    embedLookupIdx = 0; // fallback if empty token
  }
  
  // We pass the entire vocabulary so the UI can render a stable scrollable table
  const lookupVocab = EMBEDDING_VOCAB;
  const lookupWeights = lookupVocab.map((word) => generateVector(stringHash(word), "embed", hiddenDim, 1.2));
  
  const activeEmbedding = lookupWeights[embedLookupIdx];
  const allEmbeddings = allTokens.map((tok) => {
    const cleanTok = tok.text.trim();
    return generateVector(stringHash(cleanTok || "."), "embed", hiddenDim, 1.2);
  });

  // 2. RMSNorm 1 & Dropout
  const gamma1 = generateVector(1, "gamma1", hiddenDim, 0.2).map(v => parseFloat((1.0 + v).toFixed(3)));
  const norm1Data = rmsNorm(activeEmbedding, gamma1);
  const x_norm1 = norm1Data.output;

  // 3. Rotary Position Embeddings (RoPE) Simulation
  const baseAngle = 0.25;
  const ropeCoordinates = allTokens.map((tok, pos) => {
    const rawQ0 = generateVector(tok.id, "rawQ_0", 4, 1.0);
    const rawK0 = generateVector(tok.id, "rawK_0", 4, 1.0);
    
    const rotQ0 = [];
    const rotK0 = [];
    for (let pair = 0; pair < 2; pair++) {
      const idx2 = pair * 2;
      const angle = pos * (baseAngle / Math.pow(10, pair));
      const [rqX, rqY] = rotate2D(rawQ0[idx2], rawQ0[idx2+1], angle);
      const [rkX, rkY] = rotate2D(rawK0[idx2], rawK0[idx2+1], angle);
      rotQ0.push(rqX, rqY);
      rotK0.push(rkX, rkY);
    }
    
    return {
      pos,
      token: tok.text,
      rawQ: rawQ0,
      rawK: rawK0,
      rotQ: rotQ0,
      rotK: rotK0,
      angle1: pos * baseAngle,
      angle2: pos * (baseAngle / 10)
    };
  });

  // 4. QKV Projections (Input to projections is the normalized state x_norm1)
  const qVectors = [];
  for (let q = 0; q < numQHeads; q++) {
    qVectors.push(vectorMatrixMul(x_norm1, W_Q[q]));
  }

  const kVectors = [];
  const vVectors = [];
  for (let t = 0; t < seqLength; t++) {
    const kHeads = [];
    const vHeads = [];
    // We normalize all tokens for attention computation consistency
    const tokEmbed = allEmbeddings[t];
    const tokNorm = rmsNorm(tokEmbed, gamma1).output;
    for (let kv = 0; kv < numKVHeads; kv++) {
      kHeads.push(vectorMatrixMul(tokNorm, W_K[kv]));
      vHeads.push(vectorMatrixMul(tokNorm, W_V[kv]));
    }
    kVectors.push(kHeads);
    vVectors.push(vHeads);
  }

  // 5. Apply Rotary Position Embeddings (RoPE) to Q and K Projections
  const rotatedQVectors = qVectors.map((qVec) => {
    const rotQ = [];
    for (let pair = 0; pair < headDim / 2; pair++) {
      const idx2 = pair * 2;
      const angle = activeTokenIdx * (baseAngle / Math.pow(10, pair));
      const [rx, ry] = rotate2D(qVec[idx2], qVec[idx2+1], angle);
      rotQ.push(rx, ry);
    }
    return rotQ;
  });

  const rotatedKVectors = [];
  for (let t = 0; t < seqLength; t++) {
    const kHeadsRotated = [];
    for (let kv = 0; kv < numKVHeads; kv++) {
      const kVec = kVectors[t][kv];
      const rotK = [];
      for (let pair = 0; pair < headDim / 2; pair++) {
        const idx2 = pair * 2;
        const angle = t * (baseAngle / Math.pow(10, pair));
        const [rx, ry] = rotate2D(kVec[idx2], kVec[idx2+1], angle);
        rotK.push(rx, ry);
      }
      kHeadsRotated.push(rotK);
    }
    rotatedKVectors.push(kHeadsRotated);
  }

  // 6. Grouped-Query Attention & Sliding Window Attention Masking
  const attentionWeights = [];
  const rawScores = [];
  
  for (let q = 0; q < numQHeads; q++) {
    const kvIdx = Math.floor(q / qToKvRatio);
    const qVec = rotatedQVectors[q];
    
    const headRaw = [];
    let maxScore = -Infinity;
    
    for (let t = 0; t < seqLength; t++) {
      const kVec = rotatedKVectors[t][kvIdx];
      
      let dot = 0;
      for (let d = 0; d < headDim; d++) {
        dot += qVec[d] * kVec[d];
      }
      
      const isMasked = t < activeTokenIdx - windowSize + 1;
      const score = isMasked ? -100.0 : dot / Math.sqrt(headDim);
      headRaw.push(score);
      if (score > maxScore) maxScore = score;
    }
    
    let sumExp = 0;
    const exps = [];
    for (let t = 0; t < seqLength; t++) {
      const score = headRaw[t];
      if (score <= -99.0) {
        exps.push(0);
      } else {
        const val = Math.exp(score - maxScore);
        exps.push(val);
        sumExp += val;
      }
    }
    
    const headSoftmax = [];
    for (let t = 0; t < seqLength; t++) {
      const w = sumExp > 0 ? exps[t] / sumExp : 0;
      headSoftmax.push(parseFloat(w.toFixed(4)));
    }
    
    rawScores.push(headRaw);
    attentionWeights.push(headSoftmax);
  }

  // 6. Multiplication with Values
  const attentionHeadOutputs = [];
  for (let q = 0; q < numQHeads; q++) {
    const kvIdx = Math.floor(q / qToKvRatio);
    const weights = attentionWeights[q];
    
    const headOut = Array(headDim).fill(0);
    for (let d = 0; d < headDim; d++) {
      for (let t = 0; t < seqLength; t++) {
        headOut[d] += weights[t] * vVectors[t][kvIdx][d];
      }
      headOut[d] = parseFloat(headOut[d].toFixed(3));
    }
    attentionHeadOutputs.push(headOut);
  }

  // 7. Output Projection (O-Proj)
  const concatenatedHeads = attentionHeadOutputs.flat(); // size 32
  const attentionOutputProj = vectorMatrixMul(concatenatedHeads, W_O); // size 8

  // 8. Residual 1 & RMSNorm 2
  const postAttentionHidden = attentionOutputProj.map((v, i) => parseFloat((v + activeEmbedding[i]).toFixed(3)));
  const gamma2 = generateVector(2, "gamma2", hiddenDim, 0.2).map(v => parseFloat((1.0 + v).toFixed(3)));
  const norm2Data = rmsNorm(postAttentionHidden, gamma2);
  const x_norm2 = norm2Data.output;

  // 9. MLP Block (SwiGLU FFN)
  const intermediateDim = 12; // Mistral 7B expands intermediate dimension (~3.5x hidden size)
  const W_gate = generateMatrix("W_gate_ffn", hiddenDim, intermediateDim, 0.7);
  const W_up = generateMatrix("W_up_ffn", hiddenDim, intermediateDim, 0.7);
  const W_down = generateMatrix("W_down_ffn", intermediateDim, hiddenDim, 0.7);

  const gateProj = vectorMatrixMul(x_norm2, W_gate);
  const upProj = vectorMatrixMul(x_norm2, W_up);
  const siluGate = gateProj.map(v => parseFloat(silu(v).toFixed(3)));
  const elementWiseMult = siluGate.map((v, i) => parseFloat((v * upProj[i]).toFixed(3)));
  const ffnOutput = vectorMatrixMul(elementWiseMult, W_down);

  // 10. Second Residual Stream & Final Hidden State
  const finalHiddenState = ffnOutput.map((v, i) => parseFloat((v + postAttentionHidden[i]).toFixed(3)));

  // 11. Rolling Buffer Cache States
  const cacheSlots = Array.from({ length: windowSize }, (_, idx) => {
    let storedToken = null;
    let evictedToken = null;
    
    for (let pos = activeTokenIdx; pos >= 0; pos--) {
      if (pos % windowSize === idx) {
        if (!storedToken) {
          storedToken = {
            pos,
            token: allTokens[pos].text,
            isNew: pos === activeTokenIdx
          };
        } else if (!evictedToken && pos < storedToken.pos) {
          evictedToken = {
            pos,
            token: allTokens[pos].text
          };
        }
      }
    }
    
    return {
      slotIndex: idx,
      storedToken,
      evictedToken
    };
  });

  // 12. Unembedding & Logits (Predicting Next Word)
  const genStringLength = tokenStrings.length;
  
  // Find all completions that match the generated token sequence so far
  const matchingCompletions = preset.completions.filter(comp => {
    if (comp.words.length <= genStringLength) return false;
    for (let i = 0; i < genStringLength; i++) {
      if (comp.words[i] !== tokenStrings[i]) return false;
    }
    return true;
  });

  // Extract valid next words and their weights
  const validNextWordsMap = new Map();
  if (matchingCompletions.length > 0) {
    matchingCompletions.forEach(comp => {
      const nextWord = comp.words[genStringLength];
      if (!validNextWordsMap.has(nextWord) || validNextWordsMap.get(nextWord) < comp.weight) {
        validNextWordsMap.set(nextWord, comp.weight);
      }
    });
  } else {
    // Fallback if off-path (should rarely happen unless highly random)
    validNextWordsMap.set(".", 1.0);
    validNextWordsMap.set(" The", 0.8);
    validNextWordsMap.set(" In", 0.5);
  }

  const validNextWords = Array.from(validNextWordsMap.keys());
  validNextWords.sort((a, b) => validNextWordsMap.get(b) - validNextWordsMap.get(a));
  const targetNextWord = validNextWords[0];
  
  const distractorPool = [
    " model", " network", " attention", " cache", " weights", " input", 
    " sequence", " parameter", " token", " compute", " vector", " sliding",
    " CPU", " GPU", " layers", " nodes", " system", " data", " matrix"
  ];
  
  const distractors = distractorPool
    .filter(w => !validNextWordsMap.has(w));
  // Deterministically select distractors based on sequence length to avoid jumping
  const selectedDistractors = [];
  let dIdx = tokenStrings.length % distractorPool.length;
  while(selectedDistractors.length < 8 - validNextWords.length) {
    if (!validNextWordsMap.has(distractors[dIdx % distractors.length])) {
      selectedDistractors.push(distractors[dIdx % distractors.length]);
    }
    dIdx++;
  }
  
  const vocabList = [...validNextWords, ...selectedDistractors].slice(0, 8); // ensure exactly 8
  
  const W_unembed = vocabList.map((word, idx) => {
    if (validNextWordsMap.has(word)) {
      const weight = validNextWordsMap.get(word); // between ~0.3 and 1.0
      // Align target next words with finalHiddenState, scaled by their path probability weight
      const base = [...finalHiddenState];
      return base.map((v, i) => parseFloat((v * (0.35 + 0.2 * weight) + seededRandom(i) * 0.1).toFixed(2)));
    } else {
      return generateVector(idx * 13, word, hiddenDim, 0.8);
    }
  });

  const logits = vocabList.map((_, idx) => {
    let dot = 0;
    for (let d = 0; d < hiddenDim; d++) {
      dot += finalHiddenState[d] * W_unembed[idx][d];
    }
    return parseFloat(dot.toFixed(3));
  });

  // Keep vocabulary visually stable (no shuffle)
  const shuffledVocab = vocabList;
  const shuffledLogits = logits;
  const correctShuffledIndex = shuffledVocab.indexOf(targetNextWord);

  return {
    prompt: preset.prompt,
    promptTokens,
    allTokens,
    activeTokenIdx,
    activeToken,
    hasMore: tokenStrings.length < 25, // limit client generation to 25 words max
    nextWord: targetNextWord,
    
    dimensions: {
      hiddenDim,
      headDim,
      numQHeads,
      numKVHeads
    },

    embeddingLookup: {
      vocab: lookupVocab,
      activeWord: activeToken.text,
      activeIndex: lookupVocab.findIndex(w => w.toLowerCase() === activeToken.text.trim().toLowerCase()) !== -1 ? lookupVocab.findIndex(w => w.toLowerCase() === activeToken.text.trim().toLowerCase()) : 0,
      embeddingWeights: lookupWeights,
      outputVector: activeEmbedding
    },

    rmsnorm1: {
      rms: norm1Data.rms,
      normalized: norm1Data.normalized,
      gamma: gamma1,
      output: norm1Data.output
    },

    embeddings: {
      tokens: allTokens.map(t => t.text),
      vectors: allEmbeddings,
      activeVector: activeEmbedding
    },
    
    rope: ropeCoordinates,
    
    gqa: {
      numQHeads,
      numKVHeads,
      W_Q,
      W_K,
      W_V,
      W_O,
      qVectors,
      kVectors,
      vVectors,
      concatenatedHeads,
      attentionOutputProj,
      postAttentionHidden,
      attentionWeights,
      rawScores
    },
    
    swa: {
      windowSize,
      attentionWeights
    },
    
    rollingCache: {
      windowSize,
      slots: cacheSlots,
      activeSlot: activeTokenIdx % windowSize
    },

    rmsnorm2: {
      rms: norm2Data.rms,
      normalized: norm2Data.normalized,
      gamma: gamma2,
      output: norm2Data.output
    },
    
    swiglu: {
      input: x_norm2,
      W_gate,
      W_up,
      W_down,
      gateProj,
      upProj,
      siluGate,
      elementWiseMult,
      downProj: ffnOutput,
      ffnOutput,
      finalHiddenState
    },

    logitsData: {
      vocab: shuffledVocab,
      W_unembed: W_unembed,
      rawLogits: shuffledLogits,
      correctWord: targetNextWord,
      correctIndex: correctShuffledIndex,
      validWords: validNextWords
    }
  };
}

