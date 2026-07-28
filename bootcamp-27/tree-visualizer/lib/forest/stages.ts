import type { DemoStage } from "./types";

export interface StageInfo {
  id: DemoStage;
  title: string;
  eyebrow: string;
  summary: string;
  bullets: string[];
  tip: string;
}

export const STAGES: StageInfo[] = [
  {
    id: "data",
    eyebrow: "Step 1",
    title: "Meet the data",
    summary: "Labeled leaves in feature space.",
    bullets: [],
    tip: "",
  },
  {
    id: "example-tree",
    eyebrow: "Step 2",
    title: "One tree learns",
    summary: "Watch splits carve the feature space into regions.",
    bullets: [],
    tip: "",
  },
  {
    id: "train-yourself",
    eyebrow: "Step 3",
    title: "Train yourself",
    summary: "Draw split lines and watch your tree take shape.",
    bullets: [],
    tip: "",
  },
  {
    id: "grow",
    eyebrow: "Step 4",
    title: "Grow a tree",
    summary: "One tree learns decision rules.",
    bullets: [],
    tip: "",
  },
  {
    id: "predict-step",
    eyebrow: "Step 5",
    title: "First split",
    summary: "A new leaf takes one step down the tree.",
    bullets: [],
    tip: "",
  },
  {
    id: "predict",
    eyebrow: "Step 6",
    title: "To a leaf",
    summary: "Follow the rest of the path.",
    bullets: [],
    tip: "",
  },
  {
    id: "vote",
    eyebrow: "Step 7",
    title: "Forest vote",
    summary: "Many trees, one answer.",
    bullets: [],
    tip: "",
  },
  {
    id: "explore",
    eyebrow: "Playground",
    title: "Explore on your own",
    summary:
      "Rebuild the forest with different settings. Watch bags, trees, and votes change.",
    bullets: [
      "n_trees: more voters usually means stabler predictions.",
      "max_depth: how many questions each tree may ask.",
      "sample_ratio: how unique each tree's training bag is.",
    ],
    tip: "Change one knob at a time, rebuild, and compare the vote panel.",
  },
];

export function getStage(id: DemoStage): StageInfo {
  return STAGES.find((s) => s.id === id) ?? STAGES[0];
}

export const STAGE_ORDER: DemoStage[] = STAGES.map((s) => s.id);
