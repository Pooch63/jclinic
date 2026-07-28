"use client";

import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { EnsembleMerge } from "@/components/forest/EnsembleMerge";
import { ScatterPlot } from "@/components/forest/ScatterPlot";
import { StageSidebar } from "@/components/forest/StageSidebar";
import { TrainForestOverlay } from "@/components/forest/TrainForestOverlay";
import { TreeView } from "@/components/forest/TreeView";
import { Button } from "@/components/ui/Button";
import { Header } from "@/components/ui/Header";
import { Slider } from "@/components/ui/Slider";
import { Cluster, Stack } from "@/components/ui/Stack";
import { HelpHint } from "@/components/ui/Tooltip";
import { WalkthroughPortal } from "@/components/walkthrough/WalkthroughPortal";
import { generateDataset } from "@/lib/forest/dataset";
import { predictForest, trainForest } from "@/lib/forest/forest";
import { countNodes } from "@/lib/forest/layout";
import { STAGE_ORDER } from "@/lib/forest/stages";
import {
  TOUR_STEPS,
  readTourDismissed,
  writeTourDismissed,
} from "@/lib/forest/tour";
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
  // Start closed to avoid SSR/hydration flash; open after reading localStorage.
  const [tourOpen, setTourOpen] = useState(false);
  const [tourStep, setTourStep] = useState(0);
  const [stage, setStage] = useState<DemoStage>("data");
  const [config, setConfig] = useState<ForestConfig>(DEFAULT_CONFIG);
  const [draft, setDraft] = useState<ForestConfig>(DEFAULT_CONFIG);

  const data = useMemo(() => generateDataset(42, 26), []);
  const forest = useMemo(() => trainForest(data, config), [data, config]);

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
  const [knobsOpen, setKnobsOpen] = useState(false);
  const [expandOrigin, setExpandOrigin] = useState<{
    top: number;
    left: number;
    width: number;
    height: number;
  } | null>(null);
  const timerRef = useRef<number | null>(null);
  const tourSeededRef = useRef(false);
  const treePanelRef = useRef<HTMLDivElement | null>(null);

  const tree = forest.trees[Math.min(focusTree, forest.trees.length - 1)];
  const totalNodes = tree ? countNodes(tree.root) : 0;

  const bagSet = useMemo(
    () => new Set(tree?.bagIndices.map((i) => data[i].id) ?? []),
    [tree, data],
  );
  const oobSet = useMemo(
    () => new Set(tree?.oobIndices.map((i) => data[i].id) ?? []),
    [tree, data],
  );

  function clearTimer() {
    if (timerRef.current != null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  useEffect(() => () => clearTimer(), []);

  useEffect(() => {
    if (!readTourDismissed()) setTourOpen(true);
  }, []);

  useEffect(() => {
    setFocusTree(0);
    setRevealNodes(totalNodes);
    setPathIndex(-1);
    setFlow("none");
    setRevealVotes(0);
    setMerging(false);
    setPlaying(false);
    clearTimer();
  }, [stage, config]); // eslint-disable-line react-hooks/exhaustive-deps

  // Switching focus trees shows the full tree — growth is handled by TrainForestOverlay.
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

  // Keep stage in sync with the active tour step, and seed a sample for predict/vote.
  useEffect(() => {
    if (!tourOpen) return;
    const step = TOUR_STEPS[tourStep];
    if (!step) return;
    setStage(step.stage);

    if (step.stage === "predict" || step.stage === "vote") {
      if (!tourSeededRef.current) {
        const pick =
          data.find((s) => s.id === 7) ??
          data[Math.floor(data.length / 2)] ??
          data[0];
        setSelected(pick);
        setPrediction(null);
        tourSeededRef.current = true;
      }
    }
  }, [tourOpen, tourStep, data]);

  const activePath = prediction?.treePredictions[focusTree]?.path ?? [];

  const liveNotes = useMemo(() => {
    const notes: string[] = [];
    switch (stage) {
      case "data":
        notes.push(`${data.length} labeled leaves across 3 species.`);
        notes.push(
          selected
            ? `Selected #${selected.id}: ${CLASS_LABELS[selected.label]} (${selected.x.toFixed(2)}, ${selected.y.toFixed(2)}).`
            : "Click a point when you’re curious — you’ll need one for prediction.",
        );
        break;
      case "bootstrap":
        notes.push(
          `Tree ${focusTree + 1} bag: ${tree.bagIndices.length} draws (${Math.round(config.sampleRatio * 100)}% of data).`,
        );
        notes.push(
          `Out-of-bag left aside: ${tree.oobIndices.length} points (dimmed).`,
        );
        break;
      case "grow":
        notes.push(
          playing
            ? `Growing tree ${focusTree + 1}… revealed ${Math.min(revealNodes, totalNodes)}/${totalNodes} nodes.`
            : `Tree ${focusTree + 1} fully grown (${totalNodes} nodes).`,
        );
        notes.push(
          "Each split picks a random feature, then the best threshold on that feature.",
        );
        break;
      case "predict":
        if (!selected) {
          notes.push("Select a sample on the scatter plot to begin.");
        } else if (!prediction) {
          notes.push("Press “Walk through trees” to animate the forward pass.");
        } else {
          const step = activePath[pathIndex];
          if (flow === "forward" && step?.feature !== undefined) {
            notes.push(
              `Forward: ${FEATURE_NAMES[step.feature]} = ${step.value?.toFixed(2)} ${step.wentLeft ? "≤" : ">"} ${step.threshold?.toFixed(2)} → go ${step.direction}.`,
            );
          } else if (pathIndex >= 0) {
            const pred = prediction.treePredictions[focusTree].prediction;
            notes.push(`Tree ${focusTree + 1} votes ${CLASS_LABELS[pred]}.`);
          }
          notes.push(
            `Sample #${selected.id} true label: ${CLASS_LABELS[selected.label]}.`,
          );
        }
        break;
      case "vote":
        if (!prediction) {
          notes.push("Select a leaf and press “Merge votes” (or walk trees first).");
        } else {
          notes.push(
            merging
              ? `Majority: ${CLASS_LABELS[prediction.prediction]} (${Object.entries(prediction.votes)
                  .map(([k, v]) => `${CLASS_LABELS[k as keyof typeof CLASS_LABELS]} ${v}`)
                  .join(", ")}).`
              : `Revealing votes… ${revealVotes}/${prediction.treePredictions.length}.`,
          );
          notes.push("Streams of color merge into one forest decision.");
        }
        break;
      case "explore":
        notes.push("Tweak knobs, then Rebuild forest to retrain.");
        notes.push(
          selected && prediction
            ? `Current call: ${CLASS_LABELS[prediction.prediction]} for sample #${selected.id}.`
            : "Select a point and predict anytime.",
        );
        break;
    }
    return notes;
  }, [
    stage,
    config,
    data.length,
    selected,
    focusTree,
    tree,
    playing,
    revealNodes,
    totalNodes,
    prediction,
    activePath,
    pathIndex,
    flow,
    merging,
    revealVotes,
  ]);

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

  function endTour() {
    writeTourDismissed();
    setTourOpen(false);
    setStage("explore");
  }

  function startTour() {
    tourSeededRef.current = false;
    setTourStep(0);
    setStage(TOUR_STEPS[0].stage);
    setTourOpen(true);
  }

  function handleTourNext() {
    if (tourStep >= TOUR_STEPS.length - 1) {
      endTour();
      return;
    }
    setTourStep((s) => s + 1);
  }

  function handleTourBack() {
    setTourStep((s) => Math.max(0, s - 1));
  }

  function handleSelect(sample: Sample) {
    setSelected(sample);
    setPrediction(null);
    setPathIndex(-1);
    setFlow("none");
    setRevealVotes(0);
    setMerging(false);
  }

  function runPredictionAnimation() {
    if (!selected) return;
    const pred = predictForest(forest, selected);
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
      // Leaf reached — that tree's vote is done; advance to the next tree.
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
    setStage("grow");
    setTrainOpen(true);
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
    setStage("vote");
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

  const highlightBootstrap = stage === "bootstrap";
  const showTreePath =
    stage === "predict" || stage === "explore" || stage === "vote";
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

  const treePicker = (
    <div className={styles.treePicker} data-tour="focus-tree">
      {forest.trees.map((t) => (
        <button
          key={t.id}
          type="button"
          className={`${styles.treePickerBtn} ${focusTree === t.id ? styles.treePickerBtnActive : ""}`}
          disabled={playing || trainOpen}
          onClick={() => setFocusTree(t.id)}
          aria-label={`Focus tree ${t.id + 1}`}
          aria-pressed={focusTree === t.id}
        >
          {t.id + 1}
        </button>
      ))}
      <button
        type="button"
        className={styles.treePickerExpand}
        disabled={trainOpen}
        onClick={openForestExpand}
        aria-label="Expand forest view"
        title="Expand forest"
      >
        ↗
      </button>
    </div>
  );

  const actionToolbar = (
    <>
      <div data-tour="focus-tree">
        <Cluster gap="sm">
          <span className={styles.toolLabel}>
            Focus tree
            <HelpHint label="Each tree was trained on a different bootstrap bag. Flip through them to see disagreement." />
          </span>
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
      </div>

      <Cluster gap="sm">
        <Button
          size="sm"
          variant="accent"
          disabled={playing || trainOpen}
          onClick={openTrainFromScratch}
        >
          Train trees from scratch
        </Button>
        {stage === "grow" ? (
          <Button
            size="sm"
            variant="secondary"
            disabled={trainOpen}
            onClick={replayGrowth}
          >
            Replay growth
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="accent"
          disabled={!selected || playing || trainOpen}
          onClick={runPredictionAnimation}
          data-tour="walk"
        >
          Walk through trees
        </Button>
        <Button
          size="sm"
          variant="accent"
          disabled={!selected || playing || trainOpen}
          onClick={runVoteAnimation}
        >
          Merge votes
        </Button>
      </Cluster>
    </>
  );

  return (
    <div className={styles.app}>
      <WalkthroughPortal
        open={tourOpen}
        step={tourStep}
        onSkip={endTour}
        onBack={handleTourBack}
        onNext={handleTourNext}
      />

      <Header
        brand="Forest Lab"
        tagline="See how a random forest actually thinks"
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={startTour}>
              Replay tour
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={prevStage}
              disabled={stage === STAGE_ORDER[0]}
            >
              Back
            </Button>
            <Button
              variant="accent"
              size="sm"
              onClick={nextStage}
              disabled={stage === "explore"}
            >
              Next step
            </Button>
          </>
        }
      />

      <div className={styles.shell}>
        <main className={styles.main}>
          <section className={styles.workspace}>
            <div className={styles.knobsShell} data-tour="knobs">
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
                      hint="More trees → stabler majority, slower training. Diminishing returns after a point."
                      min={3}
                      max={9}
                      value={draft.nTrees}
                      onChange={(nTrees) => setDraft((d) => ({ ...d, nTrees }))}
                    />
                    <Slider
                      label="Max depth"
                      hint="How many questions a tree may ask. Too deep → memorizes noise."
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

            <div className={`${styles.panels} ${styles.panelsVote}`}>
              <div className={styles.panel} data-tour="scatter">
                <ScatterPlot
                  samples={data}
                  highlightIds={highlightBootstrap ? bagSet : undefined}
                  dimIds={highlightBootstrap ? oobSet : undefined}
                  selectedId={selected?.id ?? null}
                  pulseId={
                    stage === "predict" || stage === "vote"
                      ? selected?.id ?? null
                      : null
                  }
                  onSelect={handleSelect}
                  title={
                    highlightBootstrap
                      ? `Bootstrap bag · Tree ${focusTree + 1}`
                      : "Leaf feature space"
                  }
                />
              </div>

              <div
                className={`${styles.panel} ${styles.treePanel}`}
                data-tour="tree"
                ref={treePanelRef}
              >
                <TreeView
                  root={tree.root}
                  title={`Tree ${focusTree + 1}`}
                  revealCount={stage === "grow" ? revealNodes : undefined}
                  activePath={showTreePath ? activePath : []}
                  pathIndex={showTreePath ? pathIndex : -1}
                  flow={showTreePath ? flow : "none"}
                />
                {treePicker}
              </div>

              <div
                className={`${styles.panel} ${styles.panelWide}`}
                data-tour="votes"
              >
                <EnsembleMerge
                  prediction={prediction}
                  revealVotes={revealVotes}
                  merging={merging}
                />
              </div>
            </div>

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
                  <div className={styles.expandToolbar}>{actionToolbar}</div>
                  {expandedNode ? (() => {
                    const { formula, detail } = describeNode(expandedNode.node);
                    return (
                      <div className={styles.nodeDetail} aria-live="polite">
                        <p className={styles.nodeDetailEyebrow}>
                          Tree {expandedNode.treeId + 1} ·{" "}
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
                          : t.id < focusTree || (isAnimating && pathIndex >= 0));
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

            <TrainForestOverlay
              open={trainOpen}
              forest={forest}
              data={data}
              originStyle={expandStyle}
              onClose={closeTrainOverlay}
            />

          </section>
        </main>

        <StageSidebar stage={stage} liveNotes={liveNotes} onJump={goStage} />
      </div>
    </div>
  );
}
