"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { ScatterPlot } from "@/components/forest/ScatterPlot";
import { TreeView } from "@/components/forest/TreeView";
import { Button } from "@/components/ui/Button";
import { countNodes } from "@/lib/forest/layout";
import {
  FEATURE_NAMES,
  type RandomForest,
  type Sample,
} from "@/lib/forest/types";
import {
  latestRevealedDecision,
  revealedLeaves,
  revealedSplits,
} from "@/lib/forest/trainViz";
import styles from "./TrainForestOverlay.module.css";

type TrainPhase = "bagging" | "growing" | "between" | "done";

interface TrainForestOverlayProps {
  open: boolean;
  forest: RandomForest;
  data: Sample[];
  originStyle?: CSSProperties;
  onClose: () => void;
}

const BAG_MS = 1200;
const GROW_MS = 400;
const BETWEEN_MS = 700;
const BAG_TICK_MS = 55;

export function TrainForestOverlay({
  open,
  forest,
  data,
  originStyle,
  onClose,
}: TrainForestOverlayProps) {
  const [treeIndex, setTreeIndex] = useState(0);
  const [phase, setPhase] = useState<TrainPhase>("bagging");
  const [revealCount, setRevealCount] = useState(0);
  /** How many unique bag point ids are lit during the bagging phase. */
  const [bagReveal, setBagReveal] = useState(0);
  /** Fully grown trees stay revealed in the grid. */
  const [doneReveals, setDoneReveals] = useState<number[]>([]);
  const timerRef = useRef<number | null>(null);
  const sessionRef = useRef(0);

  function clearTimer() {
    if (timerRef.current != null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }

  useEffect(() => () => clearTimer(), []);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  // Restart the cinematic whenever the overlay opens or the forest changes.
  useEffect(() => {
    if (!open) {
      clearTimer();
      return;
    }

    const session = ++sessionRef.current;
    setTreeIndex(0);
    setPhase("bagging");
    setRevealCount(0);
    setBagReveal(0);
    setDoneReveals(forest.trees.map(() => 0));

    function schedule(fn: () => void, ms: number) {
      clearTimer();
      timerRef.current = window.setTimeout(() => {
        if (sessionRef.current !== session) return;
        fn();
      }, ms);
    }

    function runBag(treeIdx: number) {
      const nextTree = forest.trees[treeIdx];
      if (!nextTree) {
        setPhase("done");
        return;
      }
      setTreeIndex(treeIdx);
      setPhase("bagging");
      setRevealCount(0);
      setBagReveal(0);

      const uniqueBagIds = [
        ...new Set(nextTree.bagIndices.map((i) => data[i]?.id).filter((id): id is number => id != null)),
      ];
      const totalBag = uniqueBagIds.length;
      const steps = Math.max(1, Math.min(totalBag, Math.floor(BAG_MS / BAG_TICK_MS)));
      const perTick = Math.max(1, Math.ceil(totalBag / steps));
      let shown = 0;

      clearTimer();
      const tickBag = () => {
        if (sessionRef.current !== session) return;
        shown = Math.min(totalBag, shown + perTick);
        setBagReveal(shown);
        if (shown >= totalBag) {
          schedule(() => runGrow(treeIdx), 280);
          return;
        }
        timerRef.current = window.setTimeout(tickBag, BAG_TICK_MS);
      };
      timerRef.current = window.setTimeout(tickBag, BAG_TICK_MS);
    }

    function runGrow(treeIdx: number) {
      const nextTree = forest.trees[treeIdx];
      if (!nextTree) {
        setPhase("done");
        return;
      }
      const total = countNodes(nextTree.root);
      setPhase("growing");
      setBagReveal(
        new Set(nextTree.bagIndices.map((i) => data[i]?.id).filter((id): id is number => id != null))
          .size,
      );
      setRevealCount(1);

      let n = 1;
      clearTimer();
      const tick = () => {
        if (sessionRef.current !== session) return;
        if (n >= total) {
          setRevealCount(total);
          setDoneReveals((prev) => {
            const next = [...prev];
            next[treeIdx] = total;
            return next;
          });
          if (treeIdx >= forest.trees.length - 1) {
            setPhase("done");
            return;
          }
          setPhase("between");
          schedule(() => runBag(treeIdx + 1), BETWEEN_MS);
          return;
        }
        n += 1;
        setRevealCount(n);
        timerRef.current = window.setTimeout(tick, GROW_MS);
      };
      timerRef.current = window.setTimeout(tick, GROW_MS);
    }

    runBag(0);
    return clearTimer;
    // `data` is stable for the demo lifetime (fixed seed dataset).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, forest]);

  const tree = forest.trees[treeIndex];
  const bagIdsOrdered = useMemo(() => {
    if (!tree) return [] as number[];
    const seen = new Set<number>();
    const ordered: number[] = [];
    for (const idx of tree.bagIndices) {
      const id = data[idx]?.id;
      if (id == null || seen.has(id)) continue;
      seen.add(id);
      ordered.push(id);
    }
    return ordered;
  }, [tree, data]);

  const bagSet = useMemo(() => {
    if (!tree) return new Set<number>();
    if (phase === "bagging") {
      return new Set(bagIdsOrdered.slice(0, bagReveal));
    }
    return new Set(bagIdsOrdered);
  }, [tree, phase, bagIdsOrdered, bagReveal]);

  const oobSet = useMemo(() => {
    if (!tree) return new Set<number>();
    const bag = new Set(bagIdsOrdered);
    return new Set(data.filter((s) => !bag.has(s.id)).map((s) => s.id));
  }, [tree, bagIdsOrdered, data]);

  const splits = useMemo(() => {
    if (!tree || phase === "bagging") return [];
    return revealedSplits(tree.root, revealCount);
  }, [tree, phase, revealCount]);

  const leaves = useMemo(() => {
    if (!tree || phase === "bagging") return [];
    return revealedLeaves(tree.root, revealCount);
  }, [tree, phase, revealCount]);

  const activeSplit = useMemo(() => {
    if (!tree || phase !== "growing") return null;
    return latestRevealedDecision(tree.root, revealCount);
  }, [tree, phase, revealCount]);

  const { splitLeftIds, splitRightIds } = useMemo(() => {
    if (!tree || !activeSplit || phase !== "growing") {
      return {
        splitLeftIds: undefined as Set<number> | undefined,
        splitRightIds: undefined as Set<number> | undefined,
      };
    }
    const left = new Set<number>();
    const right = new Set<number>();
    const { feature, threshold, region } = activeSplit;
    for (const idx of tree.bagIndices) {
      const s = data[idx];
      if (!s) continue;
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
    return { splitLeftIds: left, splitRightIds: right };
  }, [tree, activeSplit, phase, data]);

  const statusText = useMemo(() => {
    if (phase === "done") {
      return `All ${forest.trees.length} trees trained — each from its own bootstrap bag.`;
    }
    if (phase === "bagging") {
      return `Tree ${treeIndex + 1}: bagging samples… ${bagReveal}/${bagIdsOrdered.length} unique points drawn into the backpack.`;
    }
    if (phase === "between") {
      return `Tree ${treeIndex + 1} finished. Next bag…`;
    }
    if (activeSplit) {
      return `Tree ${treeIndex + 1}: splitting on ${FEATURE_NAMES[activeSplit.feature]} ≤ ${activeSplit.threshold.toFixed(2)} — points peel into left / right leaves.`;
    }
    const total = tree ? countNodes(tree.root) : 0;
    return `Tree ${treeIndex + 1}: growing… ${Math.min(revealCount, total)}/${total} nodes.`;
  }, [phase, forest.trees.length, treeIndex, tree, activeSplit, revealCount, bagReveal, bagIdsOrdered.length]);

  const expandMinCol = Math.max(
    200,
    Math.floor(720 / Math.min(forest.trees.length, 4)),
  );

  if (!open || !tree) return null;

  return (
    <div
      className={styles.overlay}
      style={originStyle}
      role="dialog"
      aria-modal="true"
      aria-label="Train trees from scratch"
    >
      <div className={styles.surface}>
        <div className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Training cinematic</p>
            <h2 className={styles.title}>Train trees from scratch</h2>
            <p className={styles.status}>{statusText}</p>
          </div>
          <div className={styles.headerActions}>
            {phase === "done" ? (
              <Button size="sm" variant="accent" onClick={onClose}>
                Done
              </Button>
            ) : (
              <Button size="sm" variant="secondary" onClick={onClose}>
                Skip
              </Button>
            )}
          </div>
        </div>

        <div className={styles.body}>
          <div className={styles.scatterPane}>
            <ScatterPlot
              samples={data}
              highlightIds={bagSet}
              dimIds={oobSet}
              splits={splits}
              activeSplitId={activeSplit?.nodeId ?? null}
              leafRegions={leaves}
              splitLeftIds={splitLeftIds}
              splitRightIds={splitRightIds}
              title={
                phase === "bagging"
                  ? `Bootstrap bag · Tree ${treeIndex + 1}`
                  : `Splits carving the bag · Tree ${treeIndex + 1}`
              }
            />
            <div className={styles.phasePills}>
              <span
                className={`${styles.pill} ${phase === "bagging" ? styles.pillActive : ""}`}
              >
                1 · Bag
              </span>
              <span
                className={`${styles.pill} ${phase === "growing" ? styles.pillActive : ""}`}
              >
                2 · Split & grow
              </span>
              <span
                className={`${styles.pill} ${phase === "done" ? styles.pillActive : ""}`}
              >
                3 · Forest ready
              </span>
            </div>
          </div>

          <div className={styles.treesPane}>
            <div
              className={styles.treeGrid}
              style={{
                gridTemplateColumns: `repeat(auto-fit, minmax(min(100%, ${expandMinCol}px), 1fr))`,
              }}
            >
              {forest.trees.map((t) => {
                const isCurrent = t.id === treeIndex;
                let cardReveal: number | undefined;
                if (phase === "done") {
                  cardReveal = undefined;
                } else if (t.id < treeIndex) {
                  cardReveal = doneReveals[t.id] || countNodes(t.root);
                } else if (isCurrent) {
                  cardReveal = phase === "bagging" ? 0 : revealCount;
                } else {
                  cardReveal = 0;
                }

                return (
                  <div
                    key={t.id}
                    className={`${styles.treeCard} ${isCurrent && phase !== "done" ? styles.treeCardActive : ""} ${phase === "done" ? styles.treeCardDone : ""}`}
                  >
                    <TreeView
                      root={t.root}
                      title={
                        isCurrent && phase === "bagging"
                          ? `Tree ${t.id + 1} · bagging…`
                          : isCurrent && phase === "growing"
                            ? `Tree ${t.id + 1} · growing`
                            : `Tree ${t.id + 1}`
                      }
                      compact
                      revealCount={cardReveal}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
