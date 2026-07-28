export type ClassLabel = "oak" | "maple" | "pine";

export interface Sample {
  id: number;
  x: number; // petal-like feature 0 (normalized 0–1)
  y: number; // feature 1 (normalized 0–1)
  label: ClassLabel;
}

export interface DecisionNode {
  id: string;
  kind: "decision";
  feature: 0 | 1;
  threshold: number;
  left: TreeNode;
  right: TreeNode;
  /** samples that reached this node during training (for viz) */
  sampleCount: number;
  depth: number;
}

export interface LeafNode {
  id: string;
  kind: "leaf";
  prediction: ClassLabel;
  votes: Record<ClassLabel, number>;
  sampleCount: number;
  depth: number;
}

export type TreeNode = DecisionNode | LeafNode;

export interface DecisionTree {
  id: number;
  root: TreeNode;
  /** indices into the original dataset used for this bag */
  bagIndices: number[];
  /** out-of-bag indices */
  oobIndices: number[];
}

export interface RandomForest {
  trees: DecisionTree[];
  classes: ClassLabel[];
  maxDepth: number;
  maxFeatures: number;
}

export interface PathStep {
  nodeId: string;
  direction?: "left" | "right";
  feature?: 0 | 1;
  threshold?: number;
  value?: number;
  wentLeft?: boolean;
}

export interface TreePrediction {
  treeId: number;
  prediction: ClassLabel;
  path: PathStep[];
  leafVotes: Record<ClassLabel, number>;
}

export interface ForestPrediction {
  sample: Sample;
  treePredictions: TreePrediction[];
  votes: Record<ClassLabel, number>;
  prediction: ClassLabel;
}

export type DemoStage =
  | "data"
  | "example-tree"
  | "train-yourself"
  | "grow"
  | "predict-step"
  | "predict"
  | "vote"
  | "explore";

export interface ForestConfig {
  nTrees: number;
  maxDepth: number;
  sampleRatio: number;
  seed: number;
}

export interface LaidOutNode {
  id: string;
  node: TreeNode;
  x: number;
  y: number;
  parentId: string | null;
}

export const CLASS_COLORS: Record<ClassLabel, string> = {
  oak: "#2d6a4f",
  maple: "#d97706",
  pine: "#0284c7",
};

export const CLASS_LABELS: Record<ClassLabel, string> = {
  oak: "Oak",
  maple: "Maple",
  pine: "Pine",
};

export const FEATURE_NAMES = ["Leaf width", "Leaf length"] as const;
