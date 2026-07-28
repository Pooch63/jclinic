import { CLASSES, createRng } from "./dataset";
import {
  buildTree,
  emptyVotes,
  majority,
  resetNodeIds,
  tally,
} from "./tree";
import type {
  ClassLabel,
  DecisionTree,
  ForestConfig,
  ForestPrediction,
  PathStep,
  RandomForest,
  Sample,
  TreeNode,
  TreePrediction,
} from "./types";

function bootstrapIndices(
  n: number,
  sampleRatio: number,
  rng: () => number,
): { bag: number[]; oob: number[] } {
  const bagSize = Math.max(1, Math.floor(n * sampleRatio));
  const bag: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < bagSize; i++) {
    const idx = Math.floor(rng() * n);
    bag.push(idx);
    seen.add(idx);
  }
  const oob: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!seen.has(i)) oob.push(i);
  }
  return { bag, oob };
}

export function trainForest(
  data: Sample[],
  config: ForestConfig,
): RandomForest {
  resetNodeIds();
  const rng = createRng(config.seed);
  const trees: DecisionTree[] = [];
  const maxFeatures = 1; // sqrt(2) ≈ 1 — classic RF randomness

  for (let t = 0; t < config.nTrees; t++) {
    const treeSeed = Math.floor(rng() * 1e9);
    const { bag, oob } = bootstrapIndices(data.length, config.sampleRatio, rng);
    const bagSamples = bag.map((i) => data[i]);
    const root = buildTree(bagSamples, config.maxDepth, maxFeatures, treeSeed);
    trees.push({
      id: t,
      root,
      bagIndices: bag,
      oobIndices: oob,
    });
  }

  return {
    trees,
    classes: CLASSES,
    maxDepth: config.maxDepth,
    maxFeatures,
  };
}

function walkTree(
  root: TreeNode,
  sample: Sample,
): { path: PathStep[]; prediction: ClassLabel; leafVotes: Record<ClassLabel, number> } {
  const path: PathStep[] = [];
  let node: TreeNode = root;

  while (node.kind === "decision") {
    const value = node.feature === 0 ? sample.x : sample.y;
    const wentLeft = value <= node.threshold;
    path.push({
      nodeId: node.id,
      direction: wentLeft ? "left" : "right",
      feature: node.feature,
      threshold: node.threshold,
      value,
      wentLeft,
    });
    node = wentLeft ? node.left : node.right;
  }

  path.push({ nodeId: node.id });
  return {
    path,
    prediction: node.prediction,
    leafVotes: { ...node.votes },
  };
}

export function predictTree(
  tree: DecisionTree,
  sample: Sample,
): TreePrediction {
  const result = walkTree(tree.root, sample);
  return {
    treeId: tree.id,
    prediction: result.prediction,
    path: result.path,
    leafVotes: result.leafVotes,
  };
}

export function predictForest(
  forest: RandomForest,
  sample: Sample,
): ForestPrediction {
  const treePredictions = forest.trees.map((t) => predictTree(t, sample));
  const votes = emptyVotes();
  for (const tp of treePredictions) {
    votes[tp.prediction]++;
  }
  return {
    sample,
    treePredictions,
    votes,
    prediction: majority(votes),
  };
}

export function featureImportance(forest: RandomForest): [number, number] {
  const counts = [0, 0];
  function visit(node: TreeNode) {
    if (node.kind === "leaf") return;
    counts[node.feature]++;
    visit(node.left);
    visit(node.right);
  }
  for (const t of forest.trees) visit(t.root);
  const total = counts[0] + counts[1] || 1;
  return [counts[0] / total, counts[1] / total];
}

export { tally, majority, emptyVotes };
