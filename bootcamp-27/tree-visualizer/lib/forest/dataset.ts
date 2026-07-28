import type { ClassLabel, Sample } from "./types";

/** Mulberry32 — tiny deterministic PRNG for reproducible demos */
export function createRng(seed: number) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number) {
  const u = 1 - rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp01(n: number) {
  return Math.min(1, Math.max(0, n));
}

/**
 * Toy "leaf" dataset — three overlapping clusters students can see.
 * Features are already normalized to [0, 1] for easy thresholds.
 */
export function generateDataset(seed = 42, perClass = 28): Sample[] {
  const rng = createRng(seed);
  const centers: { label: ClassLabel; cx: number; cy: number; sx: number; sy: number }[] = [
    { label: "oak", cx: 0.28, cy: 0.32, sx: 0.09, sy: 0.1 },
    { label: "maple", cx: 0.68, cy: 0.38, sx: 0.1, sy: 0.09 },
    { label: "pine", cx: 0.48, cy: 0.72, sx: 0.11, sy: 0.08 },
  ];

  const samples: Sample[] = [];
  let id = 0;
  for (const c of centers) {
    for (let i = 0; i < perClass; i++) {
      samples.push({
        id: id++,
        x: clamp01(c.cx + gaussian(rng) * c.sx),
        y: clamp01(c.cy + gaussian(rng) * c.sy),
        label: c.label,
      });
    }
  }
  return samples;
}

export const CLASSES: ClassLabel[] = ["oak", "maple", "pine"];
