import type { DemoStage } from "./types";

export type TourTarget =
  | "brand"
  | "scatter"
  | "sidebar"
  | "tree"
  | "focus-tree"
  | "walk"
  | "votes"
  | "knobs"
  | "nav";

export interface TourStep {
  id: string;
  target: TourTarget;
  title: string;
  body: string;
  /** Stage to activate when this step is shown */
  stage: DemoStage;
  /** Prefer placing the coach card here relative to the target */
  placement?: "top" | "bottom" | "left" | "right";
}

/** localStorage key — set when the user skips or finishes the tour. */
export const TOUR_DISMISSED_KEY = "forest-lab-tour-dismissed";

export function readTourDismissed(): boolean {
  try {
    return localStorage.getItem(TOUR_DISMISSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function writeTourDismissed(): void {
  try {
    localStorage.setItem(TOUR_DISMISSED_KEY, "1");
  } catch {
    // ignore quota / private mode
  }
}

/** Kid-friendly, hand-holding tour that points at real UI. */
export const TOUR_STEPS: TourStep[] = [
  {
    id: "hello",
    target: "brand",
    stage: "data",
    placement: "bottom",
    title: "Welcome to Forest Lab!",
    body: "A Random Forest is a team of little decision trees that vote. You’re going to watch how they figure out what kind of leaf something is — step by step. Ready?",
  },
  {
    id: "meet-data",
    target: "scatter",
    stage: "data",
    placement: "right",
    title: "Meet the leaves",
    body: "Every colored dot is a leaf. Green is Oak, orange is Maple, blue is Pine. Click any leaf later — we’ll send it through the forest!",
  },
  {
    id: "sidebar",
    target: "sidebar",
    stage: "data",
    placement: "left",
    title: "Your story guide",
    body: "This panel always explains what’s happening. If you ever feel lost, read the title and the “Friendly tip” here.",
  },
  {
    id: "backpacks",
    target: "scatter",
    stage: "bootstrap",
    placement: "right",
    title: "Each tree packs a backpack",
    body: "Before a tree learns, it grabs a random bag of leaves (bright dots). Some leaves get left out on purpose. That way every tree sees a slightly different world!",
  },
  {
    id: "meet-tree",
    target: "tree",
    stage: "grow",
    placement: "left",
    title: "A tree asks questions",
    body: "Watch this tree grow. At each branch it asks something like “Is the leaf wide enough?” — yes goes one way, no the other — until it guesses the species.",
  },
  {
    id: "switch-trees",
    target: "focus-tree",
    stage: "grow",
    placement: "bottom",
    title: "Flip through the forest",
    body: "These tiny buttons at the bottom switch which tree you’re looking at. Try them — every tree grew from a different backpack!",
  },
  {
    id: "walk",
    target: "tree",
    stage: "predict",
    placement: "left",
    title: "Send a leaf on a walk",
    body: "We’ve picked a leaf for you. Tap ↗ on the tree panel to open the full forest, then hit “Walk through trees” to watch the leaf tumble down each tree and land on a guess.",
  },
  {
    id: "votes",
    target: "votes",
    stage: "vote",
    placement: "top",
    title: "Everyone votes!",
    body: "Each tree casts one vote. The forest’s final answer is whatever gets the most votes — like friends deciding together. Open the expanded view (↗) and press “Merge votes” to see it.",
  },
  {
    id: "knobs",
    target: "knobs",
    stage: "explore",
    placement: "top",
    title: "Now you drive",
    body: "Twist these knobs — more trees, deeper questions, bigger backpacks — then hit Rebuild. Play around. That’s how you really learn!",
  },
  {
    id: "jump",
    target: "nav",
    stage: "explore",
    placement: "left",
    title: "Jump anytime",
    body: "Use these numbers (or Back / Next up top) to revisit any step. You’re done with the tour — go explore!",
  },
];
