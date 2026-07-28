"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { EnsembleMerge } from "@/components/forest/EnsembleMerge";
import { ScatterPlot } from "@/components/forest/ScatterPlot";
import { StageSidebar } from "@/components/forest/StageSidebar";
import { TrainForestOverlay } from "@/components/forest/TrainForestOverlay";
import { TrainYourselfOverlay } from "@/components/forest/TrainYourselfOverlay";
import { TreeView } from "@/components/forest/TreeView";
import { TutorialCaption } from "@/components/forest/TutorialCaption";
import { Button } from "@/components/ui/Button";
import { Header } from "@/components/ui/Header";
import { Slider } from "@/components/ui/Slider";
import { Cluster, Stack } from "@/components/ui/Stack";
import { HelpHint } from "@/components/ui/Tooltip";
import { generateDataset } from "@/lib/forest/dataset";
import { predictForest, predictFromRoot, trainForest, USER_TREE_ID } from "@/lib/forest/forest";
import { countNodes } from "@/lib/forest/layout";
import { STAGE_ORDER } from "@/lib/forest/stages";
import {
  latestRevealedDecision,
  revealedLeaves,
  revealedSplits,
} from "@/lib/forest/trainViz";
import {
  describeFirstSplit,
  describeVoteStep,
  findDissentingTreeId,
  getTutorialStep,
  pickTutorialQuery,
  readTutorialDismissed,
  writeTutorialDismissed,
} from "@/lib/forest/tutorial";
import {
  CLASS_LABELS,
  FEATURE_NAMES,
  type ClassLabel,
  type DemoStage,
  type ForestConfig,
  type ForestPrediction,
  type Sample,
  type TreeNode,
} from "@/lib/forest/types";
import styles from "./ForestDemo.module.css";

const DEFAULT_CONFIG: ForestConfig = {
  nTrees: 5,
  maxDepth: 3,
  sampleRatio: 0.7,
  seed: 7,
};

function describeNode(node: TreeNode): { formula: string; detail: string } {
  if (node.kind === "leaf") {
    const votes = (Object.entries(node.votes) as [ClassLabel, number][])
      .filter(([, v]) => v > 0)
      .map(([k, v]) => `${CLASS_LABELS[k]} ${v}`)
      .join(", ");
    return {
      formula: `Predict ${CLASS_LABELS[node.prediction]}`,
      detail: `${votes || "no votes"} · ${node.sampleCount} training samples`,
    };
  }
  const feature = FEATURE_NAMES[node.feature];
  const threshold = node.threshold.toFixed(3);
  return {
    formula: `${feature} ≤ ${threshold}  →  left ·  ${feature} > ${threshold}  →  right`,
    detail: `${node.sampleCount} training samples reached this split`,
  };
}

export function ForestDemo() {
  const [stage, setStage] = useState<DemoStage>("data");
  const [config, setConfig] = useState<ForestConfig>(DEFAULT_CONFIG);
  const [draft, setDraft] = useState<ForestConfig>(DEFAULT_CONFIG);

  const data = useMemo(() => generateDataset(42, 26), []);
  const forest = useMemo(() => trainForest(data, config), [data, config]);
  const querySample = useMemo(() => pickTutorialQuery(forest), [forest]);
  const dissentTreeId = useMemo(
    () => findDissentingTreeId(forest, querySample),
    [forest, querySample],
  );

  const [focusTree, setFocusTree] = useState(0);
  const [revealNodes, setRevealNodes] = useState(0);
  const [selected, setSelected] = useState<Sample | null>(null);
  const [prediction, setPrediction] = useState<ForestPrediction | null>(null);
  const [pathIndex, setPathIndex] = useState(-1);
  const [flow, setFlow] = useState<"forward" | "none">("none");
  const [revealVotes, setRevealVotes] = useState(0);
  const [merging, setMerging] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [forestExpanded, setForestExpanded] = useState(false);
  const [expandedNode, setExpandedNode] = useState<{
    treeId: number;
    node: TreeNode;
  } | null>(null);
  const [trainOpen, setTrainOpen] = useState(false);
  const [trainYourselfOpen, setTrainYourselfOpen] = useState(false);
  const [userTree, setUserTree] = useState<TreeNode | null>(null);
  const [knobsOpen, setKnobsOpen] = useState(false);
  const [expandOrigin, setExpandOrigin] = useState<{
    top: number;
    left: number;
    width: number;
    height: number;
  } | null>(null);
  const timerRef = useRef<number | null>(null);
  const treePanelRef = useRef<HTMLDivElement | null>(null);
  const growStartedRef = useRef(false);
  const trainYourselfRootRef = useRef<TreeNode | null>(null);

  const inTutorial = stage !== "explore";
  const showTree = stage !== "data";
  const showVotes = stage === "vote" || stage === "explore";
  const showKnobs = stage === "explore";
  const showSidebar = stage === "explore";
  const showTreePicker =
    stage === "vote" ||
    stage === "explore" ||
    stage === "predict" ||
    (userTree != null && (stage === "grow" || stage === "predict-step"));
  const showQueryPoint =
    stage === "predict-step" || stage === "predict" || stage === "vote";

  const focusedForestTree =
    focusTree >= 0
      ? forest.trees[Math.min(focusTree, forest.trees.length - 1)]
      : null;
  const tree =
    focusTree === USER_TREE_ID && userTree
      ? {
          id: USER_TREE_ID,
          root: userTree,
          bagIndices: data.map((_, i) => i),
          oobIndices: [],
        }
      : (focusedForestTree ?? forest.trees[0]);
  const totalNodes = tree ? countNodes(tree.root) : 0;
  const viewingUserTree = focusTree === USER_TREE_ID && userTree != null;

  function clearTimer() {
    if (timerRef.current != null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  useEffect(() => () => clearTimer(), []);

  useEffect(() => {
    if (readTutorialDismissed()) {
      setStage("explore");
    } else {
      setStage("data");
    }
  }, []);

  useEffect(() => {
    if (stage === "train-yourself") {
      setTrainYourselfOpen(true);
    } else if (inTutorial) {
      setTrainYourselfOpen(false);
    }
  }, [stage, inTutorial]);

  useEffect(() => {
    growStartedRef.current = false;
    setFocusTree(0);
    setRevealNodes(totalNodes);
    setPathIndex(-1);
    setFlow("none");
    setRevealVotes(0);
    setMerging(false);
    setPlaying(false);
    clearTimer();

    if (stage === "predict-step") {
      const pred = predictForest(forest, querySample);
      setPrediction(pred);
      setPathIndex(0);
    } else if (stage === "predict") {
      runWalkThroughAnimation(querySample);
    } else if (stage === "vote") {
      const pred = predictForest(forest, querySample);
      setPrediction(pred);
      setFocusTree(dissentTreeId ?? 0);
      runTutorialVoteAnimation(pred);
    } else if (stage !== "explore") {
      setPrediction(null);
    }
  }, [stage, config]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (
      (stage !== "grow" && stage !== "example-tree") ||
      !inTutorial ||
      growStartedRef.current
    )
      return;
    growStartedRef.current = true;
    const t = window.setTimeout(() => replayGrowth(), 400);
    return () => window.clearTimeout(t);
  }, [stage, inTutorial, totalNodes]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setRevealNodes(totalNodes);
  }, [focusTree, totalNodes]);

  useEffect(() => {
    if (!forestExpanded) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setForestExpanded(false);
    }
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [forestExpanded]);

  useEffect(() => {
    if (!knobsOpen) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setKnobsOpen(false);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [knobsOpen]);

  const pathSample =
    stage === "explore" && selected
      ? selected
      : showQueryPoint
        ? querySample
        : null;

  const activePath = useMemo(() => {
    if (viewingUserTree && userTree && pathSample) {
      return predictFromRoot(userTree, pathSample).path;
    }
    if (focusTree >= 0 && prediction) {
      return prediction.treePredictions[focusTree]?.path ?? [];
    }
    return [];
  }, [viewingUserTree, userTree, pathSample, focusTree, prediction]);

  const tutorialInfo = getTutorialStep(stage);
  const tutorialBody = useMemo(() => {
    if (stage === "predict-step") {
      return describeFirstSplit(querySample, forest);
    }
    if (stage === "vote") {
      return describeVoteStep(forest, querySample);
    }
    return tutorialInfo.body;
  }, [stage, forest, querySample, tutorialInfo.body]);

  const exampleTreeRoot = forest.trees[0]?.root;
  const exampleSplits = useMemo(() => {
    if (stage !== "example-tree" || !exampleTreeRoot) return [];
    return revealedSplits(exampleTreeRoot, revealNodes);
  }, [stage, exampleTreeRoot, revealNodes]);

  const exampleLeaves = useMemo(() => {
    if (stage !== "example-tree" || !exampleTreeRoot) return [];
    return revealedLeaves(exampleTreeRoot, revealNodes);
  }, [stage, exampleTreeRoot, revealNodes]);

  const exampleActiveSplit = useMemo(() => {
    if (stage !== "example-tree" || !exampleTreeRoot) return null;
    return latestRevealedDecision(exampleTreeRoot, revealNodes);
  }, [stage, exampleTreeRoot, revealNodes]);

  const { exampleSplitLeftIds, exampleSplitRightIds } = useMemo(() => {
    if (stage !== "example-tree" || !exampleActiveSplit) {
      return {
        exampleSplitLeftIds: undefined as Set<number> | undefined,
        exampleSplitRightIds: undefined as Set<number> | undefined,
      };
    }
    const left = new Set<number>();
    const right = new Set<number>();
    const { feature, threshold, region } = exampleActiveSplit;
    for (const s of data) {
      const inRegion =
        s.x >= region.x0 &&
        s.x <= region.x1 &&
        s.y >= region.y0 &&
        s.y <= region.y1;
      if (!inRegion) continue;
      const v = feature === 0 ? s.x : s.y;
      if (v <= threshold) left.add(s.id);
      else right.add(s.id);
    }
    return { exampleSplitLeftIds: left, exampleSplitRightIds: right };
  }, [stage, exampleActiveSplit, data]);

  const liveNotes = useMemo(() => {
    if (stage !== "explore") return [];
    const notes: string[] = [];
    notes.push("Tweak knobs, then Rebuild forest to retrain.");
    notes.push(
      selected && prediction
        ? `Current call: ${CLASS_LABELS[prediction.prediction]} for sample #${selected.id}.`
        : "Select a point and predict anytime.",
    );
    return notes;
  }, [stage, selected, prediction]);

  function goStage(next: DemoStage) {
    setStage(next);
  }

  function nextStage() {
    const i = STAGE_ORDER.indexOf(stage);
    if (i < STAGE_ORDER.length - 1) setStage(STAGE_ORDER[i + 1]);
  }

  function prevStage() {
    const i = STAGE_ORDER.indexOf(stage);
    if (i > 0) setStage(STAGE_ORDER[i - 1]);
  }

  function finishTutorial() {
    writeTutorialDismissed();
    setStage("explore");
  }

  function restartTutorial() {
    growStartedRef.current = false;
    setSelected(null);
    setUserTree(null);
    setFocusTree(0);
    setStage("data");
  }

  function handleTutorialNext() {
    if (stage === "train-yourself") {
      const root = trainYourselfRootRef.current;
      if (root) {
        setUserTree(root);
        setFocusTree(USER_TREE_ID);
      }
      setTrainYourselfOpen(false);
      nextStage();
      return;
    }
    if (stage === "vote") {
      finishTutorial();
      return;
    }
    nextStage();
  }

  function handleSaveUserTree(root: TreeNode) {
    setUserTree(root);
    setFocusTree(USER_TREE_ID);
  }

  function handleSelect(sample: Sample) {
    if (inTutorial) return;
    setSelected(sample);
    setPrediction(null);
    setPathIndex(-1);
    setFlow("none");
    setRevealVotes(0);
    setMerging(false);
  }

  function runWalkThroughAnimation(sample: Sample) {
    const pred = predictForest(forest, sample);
    setPrediction(pred);
    setFocusTree(0);
    setPathIndex(0);
    setFlow("forward");
    setPlaying(true);
    clearTimer();

    let treeIdx = 0;
    let idx = 0;

    timerRef.current = window.setInterval(() => {
      const path = pred.treePredictions[treeIdx].path;
      if (idx < path.length - 1) {
        idx += 1;
        setPathIndex(idx);
        setFlow("forward");
        return;
      }
      if (treeIdx < pred.treePredictions.length - 1) {
        treeIdx += 1;
        setFocusTree(treeIdx);
        idx = 0;
        setPathIndex(0);
        setFlow("forward");
        return;
      }
      clearTimer();
      setPlaying(false);
      setFlow("none");
      setPathIndex(path.length - 1);
    }, 480);
  }

  function runTutorialVoteAnimation(pred: ForestPrediction) {
    setRevealVotes(0);
    setMerging(false);
    clearTimer();
    let count = 0;
    timerRef.current = window.setInterval(() => {
      count += 1;
      setRevealVotes(count);
      if (count >= pred.treePredictions.length) {
        clearTimer();
        setMerging(true);
      }
    }, 380);
  }

  function runPredictionAnimation() {
    if (!selected) return;
    runWalkThroughAnimation(selected);
  }

  function captureExpandOrigin() {
    const panel = treePanelRef.current;
    if (panel) {
      const rect = panel.getBoundingClientRect();
      setExpandOrigin({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      });
    } else {
      setExpandOrigin(null);
    }
  }

  function openForestExpand() {
    captureExpandOrigin();
    setTrainOpen(false);
    setForestExpanded(true);
  }

  function closeForestExpand() {
    setForestExpanded(false);
    setExpandedNode(null);
  }

  function openTrainFromScratch() {
    clearTimer();
    setPlaying(false);
    setForestExpanded(false);
    setPrediction(null);
    setPathIndex(-1);
    setRevealVotes(0);
    setMerging(false);
    setFlow("none");
    captureExpandOrigin();
    const next = { ...draft, seed: draft.seed + 1 };
    setDraft(next);
    setConfig(next);
    setTrainOpen(true);
  }

  function openTrainYourself() {
    captureExpandOrigin();
    setTrainYourselfOpen(true);
  }

  function closeTrainYourself() {
    if (stage === "train-yourself") return;
    setTrainYourselfOpen(false);
  }

  function closeTrainOverlay() {
    setTrainOpen(false);
    setRevealNodes(totalNodes);
  }

  const expandStyle = useMemo(() => {
    if (!expandOrigin || typeof window === "undefined") return undefined;
    const cx = expandOrigin.left + expandOrigin.width / 2;
    const cy = expandOrigin.top + expandOrigin.height / 2;
    return {
      "--origin-x": `${(cx / window.innerWidth) * 100}%`,
      "--origin-y": `${(cy / window.innerHeight) * 100}%`,
    } as CSSProperties;
  }, [expandOrigin]);

  function runVoteAnimation() {
    if (!selected) return;
    const pred = prediction ?? predictForest(forest, selected);
    setPrediction(pred);
    setFocusTree(0);
    setRevealVotes(0);
    setMerging(false);
    setPathIndex(0);
    setFlow("forward");
    setPlaying(true);
    clearTimer();

    let treeIdx = 0;
    let idx = 0;

    timerRef.current = window.setInterval(() => {
      const path = pred.treePredictions[treeIdx].path;
      if (idx < path.length - 1) {
        idx += 1;
        setPathIndex(idx);
        setFlow("forward");
        return;
      }
      setRevealVotes(treeIdx + 1);
      if (treeIdx < pred.treePredictions.length - 1) {
        treeIdx += 1;
        setFocusTree(treeIdx);
        idx = 0;
        setPathIndex(0);
        setFlow("forward");
        return;
      }
      clearTimer();
      setMerging(true);
      setPlaying(false);
      setFlow("none");
      setPathIndex(path.length - 1);
    }, 480);
  }

  function rebuild() {
    setConfig({ ...draft, seed: draft.seed + 1 });
    setPrediction(null);
    setPathIndex(-1);
    setRevealVotes(0);
    setMerging(false);
  }

  const showTreePath =
    stage === "predict-step" ||
    stage === "predict" ||
    stage === "explore" ||
    stage === "vote";

  const expandMinCol = Math.max(
    240,
    Math.floor(920 / Math.min(forest.trees.length, 4)),
  );

  function replayGrowth() {
    setRevealNodes(1);
    setPlaying(true);
    clearTimer();
    let n = 1;
    timerRef.current = window.setInterval(() => {
      n += 1;
      setRevealNodes(n);
      if (n >= totalNodes) {
        clearTimer();
        setPlaying(false);
      }
    }, 420);
  }

  const treePicker = showTreePicker ? (
    <div className={styles.treePicker}>
      {userTree ? (
        <button
          type="button"
          className={`${styles.treePickerBtn} ${styles.treePickerBtnUser} ${viewingUserTree ? styles.treePickerBtnActive : ""}`}
          disabled={playing || trainOpen || trainYourselfOpen}
          onClick={() => setFocusTree(USER_TREE_ID)}
          aria-label="Your tree"
          aria-pressed={viewingUserTree}
          title="Your tree"
        >
          You
        </button>
      ) : null}
      {forest.trees.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`${styles.treePickerBtn} ${focusTree === t.id ? styles.treePickerBtnActive : ""} ${dissentTreeId === t.id && stage === "vote" ? styles.treePickerBtnWrong : ""}`}
          disabled={playing || trainOpen || trainYourselfOpen}
          onClick={() => setFocusTree(t.id)}
          aria-label={`Focus tree ${t.id + 1}`}
          aria-pressed={focusTree === t.id}
        >
          {t.id + 1}
        </button>
      ))}
      {stage === "explore" ? (
        <button
          type="button"
          className={styles.treePickerExpand}
          disabled={trainOpen || trainYourselfOpen}
          onClick={openForestExpand}
          aria-label="Expand forest view"
          title="Expand forest"
        >
          ↗
        </button>
      ) : null}
    </div>
  ) : null;

  const actionToolbar = (
    <>
      <Cluster gap="sm">
        <span className={styles.toolLabel}>
          Focus tree
          <HelpHint label="Each tree was trained on a different bootstrap bag. Flip through them to see disagreement." />
        </span>
        {userTree ? (
          <Button
            size="sm"
            variant={viewingUserTree ? "accent" : "secondary"}
            onClick={() => setFocusTree(USER_TREE_ID)}
          >
            Your tree
          </Button>
        ) : null}
        {forest.trees.map((t) => (
          <Button
            key={t.id}
            size="sm"
            variant={focusTree === t.id ? "accent" : "secondary"}
            onClick={() => setFocusTree(t.id)}
          >
            {t.id + 1}
          </Button>
        ))}
      </Cluster>

      <Cluster gap="sm">
        <Button
          size="sm"
          variant="accent"
          disabled={playing || trainOpen || trainYourselfOpen}
          onClick={openTrainFromScratch}
        >
          Train trees from scratch
        </Button>
        <Button
          size="sm"
          variant="accent"
          disabled={!selected || playing || trainOpen || trainYourselfOpen}
          onClick={runPredictionAnimation}
        >
          Walk through trees
        </Button>
        <Button
          size="sm"
          variant="accent"
          disabled={!selected || playing || trainOpen || trainYourselfOpen}
          onClick={runVoteAnimation}
        >
          Merge votes
        </Button>
        <Button
          size="sm"
          variant="accent"
          disabled={playing || trainOpen || trainYourselfOpen}
          onClick={openTrainYourself}
        >
          Train yourself
        </Button>
      </Cluster>
    </>
  );

  const expandToolbar = actionToolbar;

  const panelsClass = [
    styles.panels,
    showVotes ? styles.panelsVote : "",
    stage === "data" ? styles.panelsSingle : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={styles.app}>
      <Header
        brand="Forest Lab"
        tagline="See how a random forest actually thinks"
        actions={
          stage === "explore" ? (
            <Button variant="ghost" size="sm" onClick={restartTutorial}>
              Replay tutorial
            </Button>
          ) : null
        }
      />

      <div className={`${styles.shell} ${!showSidebar ? styles.shellFull : ""}`}>
        <main className={styles.main}>
          <section className={styles.workspace}>
            {showKnobs ? (
              <div className={styles.knobsShell}>
                <button
                  type="button"
                  className={styles.knobsToggle}
                  onClick={() => setKnobsOpen((open) => !open)}
                  aria-expanded={knobsOpen}
                  aria-controls="forest-knobs-panel"
                >
                  Forest knobs
                  <span
                    className={`${styles.knobsToggleIcon} ${knobsOpen ? styles.knobsToggleIconOpen : ""}`}
                    aria-hidden
                  >
                    ▼
                  </span>
                </button>

                <div
                  id="forest-knobs-panel"
                  className={`${styles.knobsPanelWrap} ${knobsOpen ? styles.knobsPanelWrapOpen : ""}`}
                  aria-hidden={!knobsOpen}
                >
                  <div className={styles.knobsPanelInner}>
                    <Stack gap="md" className={styles.knobsPanel}>
                      <Slider
                        label="Number of trees"
                        hint="More trees usually means a stabler majority vote."
                        min={3}
                        max={9}
                        value={draft.nTrees}
                        onChange={(nTrees) => setDraft((d) => ({ ...d, nTrees }))}
                      />
                      <Slider
                        label="Max depth"
                        hint="How many questions a tree may ask."
                        min={1}
                        max={5}
                        value={draft.maxDepth}
                        onChange={(maxDepth) =>
                          setDraft((d) => ({ ...d, maxDepth }))
                        }
                      />
                      <Slider
                        label="Sample ratio"
                        hint="Fraction of data drawn into each bootstrap bag."
                        min={0.4}
                        max={1}
                        step={0.1}
                        value={draft.sampleRatio}
                        format={(v) => v.toFixed(1)}
                        onChange={(sampleRatio) =>
                          setDraft((d) => ({ ...d, sampleRatio }))
                        }
                      />
                      <Button variant="accent" onClick={rebuild}>
                        Rebuild forest
                      </Button>
                    </Stack>
                  </div>
                </div>
              </div>
            ) : null}

            <div className={panelsClass}>
              <div className={styles.panel}>
                <ScatterPlot
                  samples={data}
                  selectedId={inTutorial ? null : selected?.id ?? null}
                  pulseId={
                    stage === "explore" && selected ? selected.id : null
                  }
                  queryPoint={
                    showQueryPoint
                      ? { x: querySample.x, y: querySample.y }
                      : null
                  }
                  splits={stage === "example-tree" ? exampleSplits : undefined}
                  activeSplitId={
                    stage === "example-tree"
                      ? exampleActiveSplit?.nodeId ?? null
                      : undefined
                  }
                  leafRegions={
                    stage === "example-tree" ? exampleLeaves : undefined
                  }
                  splitLeftIds={exampleSplitLeftIds}
                  splitRightIds={exampleSplitRightIds}
                  onSelect={inTutorial ? undefined : handleSelect}
                  title="Leaf feature space"
                />
              </div>

              {showTree ? (
                <div
                  className={`${styles.panel} ${styles.treePanel}`}
                  ref={treePanelRef}
                >
                  <TreeView
                    root={tree.root}
                    title={viewingUserTree ? "Your tree" : `Tree ${focusTree + 1}`}
                    revealCount={
                      (stage === "grow" || stage === "example-tree") &&
                      !viewingUserTree
                        ? revealNodes
                        : undefined
                    }
                    activePath={showTreePath ? activePath : []}
                    pathIndex={showTreePath ? pathIndex : -1}
                    flow={showTreePath ? flow : "none"}
                  />
                  {treePicker}
                </div>
              ) : null}

              {showVotes ? (
                <div className={`${styles.panel} ${styles.panelWide}`}>
                  <EnsembleMerge
                    prediction={prediction}
                    revealVotes={revealVotes}
                    merging={merging}
                    hideTrueLabel={inTutorial && stage === "vote"}
                    highlightTreeId={
                      stage === "vote" ? dissentTreeId : null
                    }
                  />
                </div>
              ) : null}
            </div>

            {inTutorial ? (
              <TutorialCaption
                stage={stage}
                title={tutorialInfo.title}
                body={tutorialBody}
                docked={stage === "train-yourself"}
                onBack={prevStage}
                onNext={handleTutorialNext}
                onSkip={finishTutorial}
              />
            ) : null}

            {stage === "explore" ? (
              <div className={styles.toolbar}>{actionToolbar}</div>
            ) : null}

            {forestExpanded ? (
              <div
                className={styles.expandOverlay}
                style={expandStyle}
                role="dialog"
                aria-modal="true"
                aria-label="All trees in the forest"
              >
                <div className={styles.expandSurface}>
                  <div className={styles.expandHeader}>
                    <div>
                      <p className={styles.expandEyebrow}>Full forest</p>
                      <h2 className={styles.expandTitle}>
                        {forest.trees.length} trees side by side
                      </h2>
                    </div>
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={closeForestExpand}
                    >
                      Close
                    </Button>
                  </div>
                  <div className={styles.expandToolbar}>{expandToolbar}</div>
                  {expandedNode ? (() => {
                    const { formula, detail } = describeNode(expandedNode.node);
                    return (
                      <div className={styles.nodeDetail} aria-live="polite">
                        <p className={styles.nodeDetailEyebrow}>
                          {expandedNode.treeId === USER_TREE_ID
                            ? "Your tree"
                            : `Tree ${expandedNode.treeId + 1}`}{" "}
                          ·{" "}
                          {expandedNode.node.kind === "leaf"
                            ? "Leaf"
                            : "Split rule"}
                        </p>
                        <p className={styles.nodeDetailFormula}>{formula}</p>
                        <p className={styles.nodeDetailMeta}>{detail}</p>
                      </div>
                    );
                  })() : (
                    <p className={styles.nodeDetailHint}>
                      Click any node to see its split rule or leaf prediction.
                    </p>
                  )}
                  <div
                    className={styles.expandGrid}
                    style={{
                      gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${expandMinCol}px), 1fr))`,
                    }}
                  >
                    {userTree ? (
                      <div
                        key="user-tree"
                        className={`${styles.expandCard} ${viewingUserTree ? styles.expandCardActive : ""}`}
                      >
                        <TreeView
                          root={userTree}
                          title={`Your tree${
                            pathSample
                              ? ` · ${CLASS_LABELS[predictFromRoot(userTree, pathSample).prediction]}`
                              : ""
                          }`}
                          compact
                          selectedNodeId={
                            expandedNode?.treeId === USER_TREE_ID
                              ? expandedNode.node.id
                              : null
                          }
                          onNodeClick={(node) =>
                            setExpandedNode({ treeId: USER_TREE_ID, node })
                          }
                          activePath={
                            viewingUserTree && activePath.length > 0
                              ? activePath
                              : []
                          }
                          pathIndex={
                            viewingUserTree && activePath.length > 0
                              ? activePath.length - 1
                              : -1
                          }
                          flow="none"
                        />
                      </div>
                    ) : null}
                    {forest.trees.map((t) => {
                      const treePath =
                        prediction?.treePredictions[t.id]?.path ?? [];
                      const isAnimating =
                        t.id === focusTree && flow === "forward";
                      const canShowPath =
                        showTreePath && treePath.length > 0;
                      const walkedThrough =
                        canShowPath &&
                        (flow === "none"
                          ? true
                          : t.id < focusTree ||
                            (isAnimating && pathIndex >= 0));
                      const showLive = canShowPath && isAnimating;
                      const showDone =
                        walkedThrough && !showLive && treePath.length > 0;
                      return (
                        <div
                          key={t.id}
                          className={`${styles.expandCard} ${showLive ? styles.expandCardActive : ""}`}
                        >
                          <TreeView
                            root={t.root}
                            title={`Tree ${t.id + 1}${
                              prediction
                                ? ` · ${CLASS_LABELS[prediction.treePredictions[t.id].prediction]}`
                                : ""
                            }`}
                            compact
                            selectedNodeId={
                              expandedNode?.treeId === t.id
                                ? expandedNode.node.id
                                : null
                            }
                            onNodeClick={(node) =>
                              setExpandedNode({ treeId: t.id, node })
                            }
                            activePath={
                              showLive || showDone ? treePath : []
                            }
                            pathIndex={
                              showLive
                                ? pathIndex
                                : showDone
                                  ? treePath.length - 1
                                  : -1
                            }
                            flow={showLive ? flow : "none"}
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : null}

            <TrainYourselfOverlay
              open={trainYourselfOpen}
              data={data}
              originStyle={expandStyle}
              tutorialMode={stage === "train-yourself"}
              onClose={closeTrainYourself}
              onSave={handleSaveUserTree}
              onRootChange={(root) => {
                trainYourselfRootRef.current = root;
              }}
            />

            <TrainForestOverlay
              open={trainOpen}
              forest={forest}
              data={data}
              originStyle={expandStyle}
              onClose={closeTrainOverlay}
            />
          </section>
        </main>

        {showSidebar ? (
          <StageSidebar stage={stage} liveNotes={liveNotes} onJump={goStage} />
        ) : null}
      </div>
    </div>
  );
}
