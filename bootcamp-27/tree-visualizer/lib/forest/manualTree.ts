import { resetNodeIds } from "./tree";
import { majority, tally } from "./tree";
import type { ClassLabel, LeafNode, Sample, TreeNode } from "./types";
import type { Region } from "./trainViz";
import { nodeRegions } from "./trainViz";

function makeLeaf(samples: Sample[], depth: number): LeafNode {
  const votes = tally(samples);
  return {
    id: `leaf-${depth}-${samples.length}-${Math.random().toString(36).slice(2, 7)}`,
    kind: "leaf",
    prediction: majority(votes),
    votes,
    sampleCount: samples.length,
    depth,
  };
}

/** Single root leaf holding all training samples — your tree starts here. */
export function createManualTreeRoot(samples: Sample[]): TreeNode {
  resetNodeIds();
  return makeLeaf(samples, 0);
}

/** Samples that reached a node during training traversal. */
export function samplesAtNode(
  root: TreeNode,
  nodeId: string,
  allSamples: Sample[],
): Sample[] | null {
  function walk(node: TreeNode, nodeSamples: Sample[]): Sample[] | null {
    if (node.id === nodeId) return nodeSamples;
    if (node.kind === "leaf") return null;
    const left: Sample[] = [];
    const right: Sample[] = [];
    for (const s of nodeSamples) {
      const v = node.feature === 0 ? s.x : s.y;
      if (v <= node.threshold) left.push(s);
      else right.push(s);
    }
    return walk(node.left, left) ?? walk(node.right, right);
  }
  return walk(root, allSamples);
}

export function isPure(samples: Sample[]): boolean {
  if (samples.length <= 1) return true;
  const first = samples[0].label;
  return samples.every((s) => s.label === first);
}

export function partitionBySplit(
  samples: Sample[],
  feature: 0 | 1,
  threshold: number,
): { left: Sample[]; right: Sample[] } {
  const left: Sample[] = [];
  const right: Sample[] = [];
  for (const s of samples) {
    const v = feature === 0 ? s.x : s.y;
    if (v <= threshold) left.push(s);
    else right.push(s);
  }
  return { left, right };
}

export function isValidSplit(
  samples: Sample[],
  feature: 0 | 1,
  threshold: number,
): boolean {
  const { left, right } = partitionBySplit(samples, feature, threshold);
  return left.length > 0 && right.length > 0;
}

export function canSplitOnAxis(samples: Sample[], feature: 0 | 1): boolean {
  if (samples.length < 2) return false;
  const values = [
    ...new Set(samples.map((s) => (feature === 0 ? s.x : s.y))),
  ];
  return values.length > 1;
}

export function thresholdForClick(
  samples: Sample[],
  feature: 0 | 1,
  preferred: number,
  region?: Region,
): number | null {
  if (samples.length < 2) return null;
  if (!canSplitOnAxis(samples, feature)) return null;

  let pref = preferred;
  if (region) {
    if (feature === 0) {
      pref = Math.max(region.x0, Math.min(region.x1, preferred));
    } else {
      pref = Math.max(region.y0, Math.min(region.y1, preferred));
    }
  }

  if (isValidSplit(samples, feature, pref)) return pref;

  return nearestValidThreshold(samples, feature, pref);
}

export function nearestValidThreshold(
  samples: Sample[],
  feature: 0 | 1,
  preferred: number,
): number | null {
  const values = [
    ...new Set(samples.map((s) => (feature === 0 ? s.x : s.y))),
  ].sort((a, b) => a - b);
  const candidates: number[] = [];
  for (let i = 0; i < values.length - 1; i++) {
    candidates.push((values[i] + values[i + 1]) / 2);
  }
  const valid = candidates.filter((c) => isValidSplit(samples, feature, c));
  if (valid.length === 0) return null;

  let best = valid[0];
  let bestDist = Math.abs(best - preferred);
  for (const c of valid) {
    const dist = Math.abs(c - preferred);
    if (dist < bestDist) {
      best = c;
      bestDist = dist;
    }
  }
  return best;
}

export interface SplitResult {
  root: TreeNode;
  /** Id of the new decision node */
  decisionId: string;
  leftLeafId: string;
  rightLeafId: string;
}

/** Replace a leaf with a decision split; returns null if invalid. */
export function splitLeaf(
  root: TreeNode,
  nodeId: string,
  feature: 0 | 1,
  threshold: number,
  allSamples: Sample[],
): SplitResult | null {
  const nodeSamples = samplesAtNode(root, nodeId, allSamples);
  if (!nodeSamples || nodeSamples.length === 0) return null;

  const target = findNode(root, nodeId);
  if (!target || target.kind !== "leaf") return null;
  if (!isValidSplit(nodeSamples, feature, threshold)) return null;

  const { left, right } = partitionBySplit(nodeSamples, feature, threshold);
  const depth = target.depth;

  const leftLeaf = makeLeaf(left, depth + 1);
  const rightLeaf = makeLeaf(right, depth + 1);
  const decision = {
    id: `node-${depth}-${feature}-${threshold.toFixed(3)}`,
    kind: "decision" as const,
    feature,
    threshold,
    left: leftLeaf,
    right: rightLeaf,
    sampleCount: nodeSamples.length,
    depth,
  };

  function replace(node: TreeNode): TreeNode {
    if (node.id === nodeId) return decision;
    if (node.kind === "leaf") return node;
    return {
      ...node,
      left: replace(node.left),
      right: replace(node.right),
    };
  }

  return {
    root: replace(root),
    decisionId: decision.id,
    leftLeafId: leftLeaf.id,
    rightLeafId: rightLeaf.id,
  };
}

function findNode(node: TreeNode, id: string): TreeNode | null {
  if (node.id === id) return node;
  if (node.kind === "leaf") return null;
  return findNode(node.left, id) ?? findNode(node.right, id);
}

/** Leaf nodes that can still accept another split. */
export function splittableLeaves(
  root: TreeNode,
  allSamples: Sample[],
): LeafNode[] {
  const out: LeafNode[] = [];
  function visit(node: TreeNode) {
    if (node.kind === "leaf") {
      const samples = samplesAtNode(root, node.id, allSamples) ?? [];
      if (samples.length > 1) {
        out.push(node);
      }
      return;
    }
    visit(node.left);
    visit(node.right);
  }
  visit(root);
  return out;
}

/** Region in feature space for a node (for hit-testing clicks). */
export function regionForNode(root: TreeNode, nodeId: string): Region | null {
  return nodeRegions(root).get(nodeId) ?? null;
}

/** Find the deepest leaf whose region contains (x, y). */
export function leafAtPoint(
  root: TreeNode,
  x: number,
  y: number,
): LeafNode | null {
  const regions = nodeRegions(root);
  let best: LeafNode | null = null;
  let bestDepth = -1;

  function visit(node: TreeNode) {
    if (node.kind === "leaf") {
      const r = regions.get(node.id);
      if (
        r &&
        x >= r.x0 &&
        x <= r.x1 &&
        y >= r.y0 &&
        y <= r.y1 &&
        node.depth > bestDepth
      ) {
        best = node;
        bestDepth = node.depth;
      }
      return;
    }
    visit(node.left);
    visit(node.right);
  }
  visit(root);
  return best;
}

export function describeLeaf(samples: Sample[]): {
  prediction: ClassLabel;
  mixed: boolean;
} {
  const votes = tally(samples);
  return {
    prediction: majority(votes),
    mixed: !isPure(samples),
  };
}
