import { predictForest } from "./forest";
import type { ClassLabel, DemoStage, RandomForest, Sample } from "./types";
import { CLASS_LABELS, FEATURE_NAMES } from "./types";

export interface TutorialStep {
  id: DemoStage;
  title: string;
  /** Static body; use getTutorialBody for dynamic steps */
  body: string;
}

export const TUTORIAL_DISMISSED_KEY = "forest-lab-tutorial-dismissed";

export function readTutorialDismissed(): boolean {
  try {
    return localStorage.getItem(TUTORIAL_DISMISSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function writeTutorialDismissed(): void {
  try {
    localStorage.setItem(TUTORIAL_DISMISSED_KEY, "1");
  } catch {
    // ignore quota / private mode
  }
}

/** Fixed grid of candidate query points in overlap regions. */
const QUERY_CANDIDATES: Array<{ x: number; y: number }> = [
  { x: 0.52, y: 0.62 },
  { x: 0.48, y: 0.58 },
  { x: 0.55, y: 0.55 },
  { x: 0.45, y: 0.65 },
  { x: 0.58, y: 0.68 },
  { x: 0.5, y: 0.7 },
  { x: 0.42, y: 0.52 },
  { x: 0.6, y: 0.6 },
];

export function pickTutorialQuery(forest: RandomForest): Sample {
  for (const { x, y } of QUERY_CANDIDATES) {
    const sample: Sample = { id: -1, x, y, label: "oak" };
    const pred = predictForest(forest, sample);
    const dissent = pred.treePredictions.filter(
      (t) => t.prediction !== pred.prediction,
    );
    if (dissent.length >= 1 && dissent.length < pred.treePredictions.length) {
      return sample;
    }
  }
  return { id: -1, x: 0.52, y: 0.62, label: "oak" };
}

export function findDissentingTreeId(
  forest: RandomForest,
  sample: Sample,
): number | null {
  const pred = predictForest(forest, sample);
  const dissent = pred.treePredictions.find(
    (t) => t.prediction !== pred.prediction,
  );
  return dissent?.treeId ?? null;
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "data",
    title: "The training data",
    body: "Each dot is a leaf measured by width and length. Color shows the species. Random forests learn patterns from data like this. They draw boundaries between the data so that when a new datapoint comes in, they can match it.",
  },
  {
    id: "example-tree",
    title: "How one tree learns",
    body: "Watch the tree ask questions on width and length. Each split line divides the space. Tinted regions show where the tree would guess each species. This is how a single tree learns from the data.",
  },
  {
    id: "train-yourself",
    title: "Try it yourself",
    body: "Click any region to place a split line. Press D to switch between width and length. When you're happy with it, save. Your tree will appear in the main display so you can compare it to the computer's.",
  },
  {
    id: "grow",
    title: "Now the computer grows one",
    body: "You just drew boundaries by hand. Next, watch the computer build a tree automatically — it picks splits on width and length until it can guess the species.",
  },
  {
    id: "predict-step",
    title: "A new leaf",
    body: "",
  },
  {
    id: "predict",
    title: "Walk through every tree",
    body: "Each tree asks its own questions until the sample lands in a leaf. Watch as we visit every tree one by one — same as Walk through trees in the playground.",
  },
  {
    id: "vote",
    title: "Many trees vote",
    body: "",
  },
  {
    id: "explore",
    title: "Your turn",
    body: "Click any leaf, walk through trees, and merge votes. Open the knobs to change settings and rebuild the forest.",
  },
];

export function getTutorialStep(stage: DemoStage): TutorialStep {
  return TUTORIAL_STEPS.find((s) => s.id === stage) ?? TUTORIAL_STEPS[0];
}

export function describeFirstSplit(sample: Sample, forest: RandomForest): string {
  const pred = predictForest(forest, sample);
  const step = pred.treePredictions[0]?.path[0];
  if (step?.feature === undefined) {
    return "This grey dot is a new leaf we have not seen before. Watch how the tree classifies it.";
  }
  const feat = FEATURE_NAMES[step.feature!].toLowerCase();
  const val = step.value!.toFixed(2);
  const thresh = step.threshold!.toFixed(2);
  const dir = step.wentLeft ? "left" : "right";
  return `This new leaf has ${feat} ${val}. The tree asks: is ${feat} ≤ ${thresh}? ${step.wentLeft ? "Yes" : "No"}, so we go ${dir}.`;
}

export function describeVoteStep(
  forest: RandomForest,
  sample: Sample,
): string {
  const pred = predictForest(forest, sample);
  const dissent = pred.treePredictions.filter(
    (t) => t.prediction !== pred.prediction,
  );
  if (dissent.length === 0) {
    return "Each tree casts one vote. The forest picks whichever species gets the most votes.";
  }
  const wrong = dissent[0];
  return `Tree ${wrong.treeId + 1} guessed ${CLASS_LABELS[wrong.prediction]}, but the boundary is tricky here. Sometimes individual trees get it wrong. So we grow many trees and let them vote. Majority wins.`;
}
