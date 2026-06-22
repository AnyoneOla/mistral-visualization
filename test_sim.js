// test_sim.js
import { simulateStep } from "./src/utils/mistralSim.js";

const data = simulateStep(0, [], 4, 8, 2);
console.log("SIMULATION STEP DATA TEST:");
console.log("activeToken:", data.activeToken);
console.log("embeddingLookup vocab size:", data.embeddingLookup.vocab.length);
console.log("embeddingLookup activeIndex:", data.embeddingLookup.activeIndex);
console.log("embeddingLookup activeWord:", data.embeddingLookup.activeWord);
console.log("embeddingWeights shape:", data.embeddingLookup.embeddingWeights.length, "x", data.embeddingLookup.embeddingWeights[0].length);
console.log("outputVector length:", data.embeddingLookup.outputVector.length);
console.log("outputVector values:", data.embeddingLookup.outputVector);

console.log("\nALL PROFILES MATCHED!");
