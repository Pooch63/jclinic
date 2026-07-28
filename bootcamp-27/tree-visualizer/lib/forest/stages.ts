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
    eyebrow: "Step 1 · Meet the data",
    title: "Three kinds of leaves",
    summary:
      "Each dot is a leaf measured by width and length. Color is the true species. Your job (and the forest’s) is to learn the boundaries.",
    bullets: [
      "Oak clusters left, Maple right, Pine higher up — with some overlap.",
      "Click any point to watch it travel through the trees later.",
      "Overlap is why a single rule (“if width > 0.5…”) isn’t enough.",
    ],
    tip: "Hover a point to peek at its features. That’s what every split looks at.",
  },
  {
    id: "bootstrap",
    eyebrow: "Step 2 · Bootstrap bags",
    title: "Each tree gets its own backpack",
    summary:
      "Before growing, we draw a random sample with replacement — a bootstrap bag. Some points appear twice; some are left out (out-of-bag).",
    bullets: [
      "Bags make trees disagree — that’s a feature, not a bug.",
      "Out-of-bag points are a free mini test set for that tree.",
      "Watch the highlighted points: they’re this tree’s training bag.",
    ],
    tip: "Try changing “Sample ratio” later — smaller bags → more diverse trees.",
  },
  {
    id: "grow",
    eyebrow: "Step 3 · Grow the trees",
    title: "Questions that carve the space",
    summary:
      "Each tree asks yes/no questions like “Is leaf width ≤ 0.42?” and splits until leaves are mostly one species — or we hit max depth.",
    bullets: [
      "At each split we only peek at a random feature — more diversity.",
      "Watch nodes bloom top-down as the tree evolves.",
      "Shallower trees = simpler rules; deeper trees = more detail (and risk).",
    ],
    tip: "Max depth is the most common knob students turn too high. Start small.",
  },
  {
    id: "predict",
    eyebrow: "Step 4 · Walk a sample",
    title: "Data flows down the tree",
    summary:
      "To classify a leaf, we send it down from the root. At every decision it takes left or right. The leaf it lands in is that tree’s vote.",
    bullets: [
      "Forward pass: sample → splits → leaf prediction. That’s the tree’s vote.",
      "No backprop here — inference only walks down, then the leaf casts its vote.",
      "Do this for every tree in the forest — they won’t all agree.",
    ],
    tip: "Pick a point near a cluster edge — that’s where trees disagree most.",
  },
  {
    id: "vote",
    eyebrow: "Step 5 · Merge the votes",
    title: "Many opinions, one answer",
    summary:
      "Each tree casts a vote. The forest’s prediction is the majority. Streams of color merge into a single final call.",
    bullets: [
      "Majority vote softens wild individual trees.",
      "Vote counts show confidence — a 5–0 forest is surer than 3–2.",
      "You’ve just watched bagging + aggregation — the heart of Random Forest.",
    ],
    tip: "If votes are close, try more trees — the majority usually stabilizes.",
  },
  {
    id: "explore",
    eyebrow: "Playground",
    title: "Twist the knobs yourself",
    summary:
      "Rebuild the forest with different settings. Watch bags, trees, and votes change. This is how you build intuition — not by memorizing formulas.",
    bullets: [
      "n_trees: more voters → usually stabler predictions.",
      "max_depth: how nosy each tree is allowed to be.",
      "sample_ratio: how unique each tree’s backpack is.",
    ],
    tip: "Change one knob at a time, rebuild, and compare the vote panel.",
  },
];

export function getStage(id: DemoStage): StageInfo {
  return STAGES.find((s) => s.id === id) ?? STAGES[0];
}

export const STAGE_ORDER: DemoStage[] = STAGES.map((s) => s.id);
