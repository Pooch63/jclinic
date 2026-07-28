import { CLASSES } from "./dataset";
import { createRng } from "./dataset";
import type {
  ClassLabel,
  DecisionNode,
  LeafNode,
  Sample,
  TreeNode,
} from "./types";

function emptyVotes(): Record<ClassLabel, number> {
  return { oak: 0, maple: 0, pine: 0 };
}

function tally(samples: Sample[]): Record<ClassLabel, number> {
  const votes = emptyVotes();
  for (const s of samples) votes[s.label]++;
  return votes;
}

function majority(votes: Record<ClassLabel, number>): ClassLabel {
  let best: ClassLabel = "oak";
  let bestCount = -1;
  for (const c of CLASSES) {
    if (votes[c] > bestCount) {
      best = c;
      bestCount = votes[c];
    }
  }
  return best;
}

function gini(samples: Sample[]): number {
  if (samples.length === 0) return 0;
  const votes = tally(samples);
  let impurity = 1;
  const n = samples.length;
  for (const c of CLASSES) {
    const p = votes[c] / n;
    impurity -= p * p;
  }
  return impurity;
}

function isPure(samples: Sample[]): boolean {
  if (samples.length <= 1) return true;
  const first = samples[0].label;
  return samples.every((s) => s.label === first);
}

interface Split {
  feature: 0 | 1;
  threshold: number;
  left: Sample[];
  right: Sample[];
  gain: number;
}

function candidateThresholds(values: number[]): number[] {
  const sorted = [...new Set(values)].sort((a, b) => a - b);
  const out: number[] = [];
  for (let i = 0; i < sorted.length - 1; i++) {
    out.push((sorted[i] + sorted[i + 1]) / 2);
  }
  return out;
}

function findBestSplit(
  samples: Sample[],
  features: (0 | 1)[],
): Split | null {
  const parentGini = gini(samples);
  let best: Split | null = null;

  for (const feature of features) {
    const values = samples.map((s) => (feature === 0 ? s.x : s.y));
    for (const threshold of candidateThresholds(values)) {
      const left: Sample[] = [];
      const right: Sample[] = [];
      for (const s of samples) {
        const v = feature === 0 ? s.x : s.y;
        if (v <= threshold) left.push(s);
        else right.push(s);
      }
      if (left.length === 0 || right.length === 0) continue;
      const n = samples.length;
      const gain =
        parentGini -
        (left.length / n) * gini(left) -
        (right.length / n) * gini(right);
      if (!best || gain > best.gain) {
        best = { feature, threshold, left, right, gain };
      }
    }
  }
  return best && best.gain > 1e-9 ? best : null;
}

let nodeCounter = 0;
export function resetNodeIds() {
  nodeCounter = 0;
}

function nextId(prefix: string) {
  return `${prefix}-${nodeCounter++}`;
}

function makeLeaf(samples: Sample[], depth: number): LeafNode {
  const votes = tally(samples);
  return {
    id: nextId("leaf"),
    kind: "leaf",
    prediction: majority(votes),
    votes,
    sampleCount: samples.length,
    depth,
  };
}

function grow(
  samples: Sample[],
  depth: number,
  maxDepth: number,
  maxFeatures: number,
  rng: () => number,
): TreeNode {
  if (
    samples.length === 0 ||
    depth >= maxDepth ||
    isPure(samples)
  ) {
    return makeLeaf(samples, depth);
  }

  const allFeatures: (0 | 1)[] = [0, 1];
  const shuffled = [...allFeatures].sort(() => rng() - 0.5);
  const features = shuffled.slice(0, Math.min(maxFeatures, shuffled.length));

  const split = findBestSplit(samples, features);
  if (!split) return makeLeaf(samples, depth);

  const node: DecisionNode = {
    id: nextId("node"),
    kind: "decision",
    feature: split.feature,
    threshold: split.threshold,
    left: grow(split.left, depth + 1, maxDepth, maxFeatures, rng),
    right: grow(split.right, depth + 1, maxDepth, maxFeatures, rng),
    sampleCount: samples.length,
    depth,
  };
  return node;
}

export function buildTree(
  bag: Sample[],
  maxDepth: number,
  maxFeatures: number,
  seed: number,
): TreeNode {
  const rng = createRng(seed);
  return grow(bag, 0, maxDepth, maxFeatures, rng);
}

export { majority, tally, emptyVotes };
