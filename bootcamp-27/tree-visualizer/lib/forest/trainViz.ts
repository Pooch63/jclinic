import type { ClassLabel, TreeNode } from "./types";
import { CLASS_COLORS } from "./types";

export interface Region {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export interface SplitViz {
  nodeId: string;
  feature: 0 | 1;
  threshold: number;
  region: Region;
}

export interface LeafRegionViz {
  nodeId: string;
  region: Region;
  color: string;
  prediction: ClassLabel;
}

/** Preorder node ids — same order TreeView uses for revealCount. */
export function growthOrder(node: TreeNode, order: string[] = []): string[] {
  order.push(node.id);
  if (node.kind === "decision") {
    growthOrder(node.left, order);
    growthOrder(node.right, order);
  }
  return order;
}

export function nodeRegions(root: TreeNode): Map<string, Region> {
  const map = new Map<string, Region>();

  function visit(node: TreeNode, region: Region) {
    map.set(node.id, region);
    if (node.kind !== "decision") return;

    const left: Region = { ...region };
    const right: Region = { ...region };
    if (node.feature === 0) {
      left.x1 = node.threshold;
      right.x0 = node.threshold;
    } else {
      left.y1 = node.threshold;
      right.y0 = node.threshold;
    }
    visit(node.left, left);
    visit(node.right, right);
  }

  visit(root, { x0: 0, x1: 1, y0: 0, y1: 1 });
  return map;
}

/** Split lines for decision nodes currently revealed. */
export function revealedSplits(
  root: TreeNode,
  revealCount: number,
): SplitViz[] {
  const order = growthOrder(root);
  const revealed = new Set(order.slice(0, Math.max(0, revealCount)));
  const regions = nodeRegions(root);
  const splits: SplitViz[] = [];

  function visit(node: TreeNode) {
    if (!revealed.has(node.id)) return;
    if (node.kind === "decision") {
      const region = regions.get(node.id);
      if (region) {
        splits.push({
          nodeId: node.id,
          feature: node.feature,
          threshold: node.threshold,
          region,
        });
      }
      visit(node.left);
      visit(node.right);
    }
  }

  visit(root);
  return splits;
}

/** Leaf regions for leaves currently revealed (for tinting the plot). */
export function revealedLeaves(
  root: TreeNode,
  revealCount: number,
): LeafRegionViz[] {
  const order = growthOrder(root);
  const revealed = new Set(order.slice(0, Math.max(0, revealCount)));
  const regions = nodeRegions(root);
  const leaves: LeafRegionViz[] = [];

  function visit(node: TreeNode) {
    if (!revealed.has(node.id)) return;
    if (node.kind === "leaf") {
      const region = regions.get(node.id);
      if (region) {
        leaves.push({
          nodeId: node.id,
          region,
          color: CLASS_COLORS[node.prediction],
          prediction: node.prediction,
        });
      }
      return;
    }
    visit(node.left);
    visit(node.right);
  }

  visit(root);
  return leaves;
}

/** Latest decision node among those revealed (for highlighting the active split). */
export function latestRevealedDecision(
  root: TreeNode,
  revealCount: number,
): SplitViz | null {
  const order = growthOrder(root);
  const regions = nodeRegions(root);
  for (let i = Math.min(revealCount, order.length) - 1; i >= 0; i--) {
    const id = order[i];
    const found = findNode(root, id);
    if (found?.kind === "decision") {
      const region = regions.get(id);
      if (!region) return null;
      return {
        nodeId: id,
        feature: found.feature,
        threshold: found.threshold,
        region,
      };
    }
  }
  return null;
}

function findNode(node: TreeNode, id: string): TreeNode | null {
  if (node.id === id) return node;
  if (node.kind === "leaf") return null;
  return findNode(node.left, id) ?? findNode(node.right, id);
}
