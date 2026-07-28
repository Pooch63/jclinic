"use client";

import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { InteractiveSplitPlot } from "@/components/forest/InteractiveSplitPlot";
import { TreeView } from "@/components/forest/TreeView";
import { Button } from "@/components/ui/Button";
import { Cluster } from "@/components/ui/Stack";
import {
  createManualTreeRoot,
  describeLeaf,
  samplesAtNode,
  splitLeaf,
  splittableLeaves,
} from "@/lib/forest/manualTree";
import { type Sample, type TreeNode } from "@/lib/forest/types";
import styles from "./TrainYourselfOverlay.module.css";

function findNodeById(node: TreeNode, id: string): TreeNode | null {
  if (node.id === id) return node;
  if (node.kind === "leaf") return null;
  return findNodeById(node.left, id) ?? findNodeById(node.right, id);
}

interface TrainYourselfOverlayProps {
  open: boolean;
  data: Sample[];
  originStyle?: CSSProperties;
  tutorialMode?: boolean;
  onClose: () => void;
  onSave?: (root: TreeNode) => void;
  onRootChange?: (root: TreeNode) => void;
}

export function TrainYourselfOverlay({
  open,
  data,
  originStyle,
  tutorialMode = false,
  onClose,
  onSave,
  onRootChange,
}: TrainYourselfOverlayProps) {
  const [history, setHistory] = useState<TreeNode[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [splitFeature, setSplitFeature] = useState<0 | 1>(0);
  const [axisHint, setAxisHint] = useState<string | null>(null);

  const root = history[history.length - 1] ?? null;

  const reset = useCallback(() => {
    const initial = createManualTreeRoot(data);
    setHistory([initial]);
    setSelectedNodeId(initial.id);
    setSplitFeature(0);
  }, [data]);

  useEffect(() => {
    if (open) reset();
  }, [open, reset]);

  useEffect(() => {
    if (root) onRootChange?.(root);
  }, [root, onRootChange]);

  const toggleSplitFeature = useCallback(() => {
    setSplitFeature((f) => (f === 0 ? 1 : 0));
  }, []);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLElement) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || e.target.isContentEditable) {
          return;
        }
      }
      if (e.key === "Escape") {
        if (!tutorialMode) onClose();
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      const key = e.key.toLowerCase();
      if (key === "d") {
        e.preventDefault();
        toggleSplitFeature();
      } else if (key === "w") {
        e.preventDefault();
        setSplitFeature(0);
      } else if (key === "l") {
        e.preventDefault();
        setSplitFeature(1);
      }
    }
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose, toggleSplitFeature, tutorialMode]);

  const splittable = useMemo(
    () => (root ? splittableLeaves(root, data) : []),
    [root, data],
  );

  const selectedSamples = useMemo(
    () =>
      root && selectedNodeId
        ? (samplesAtNode(root, selectedNodeId, data) ?? [])
        : [],
    [root, selectedNodeId, data],
  );

  const statusText = useMemo(() => {
    if (axisHint) return axisHint;
    if (!root) return "";
    if (splittable.length === 0) {
      return "Every region has only one point left — your tree is ready.";
    }
    if (!selectedNodeId) {
      return "Click a region to place a split line.";
    }
    const axis =
      splitFeature === 0 ? "vertical (width)" : "horizontal (length)";
    return `Click to place a ${axis} line · press D to switch axis.`;
  }, [root, splittable, selectedNodeId, splitFeature, axisHint]);

  function handleCommitSplit(nodeId: string, feature: 0 | 1, threshold: number) {
    if (!root) return;
    const result = splitLeaf(root, nodeId, feature, threshold, data);
    if (!result) return;

    setHistory((h) => [...h, result.root]);

    const freshSplittable = splittableLeaves(result.root, data);
    const preferLeft = freshSplittable.some((l) => l.id === result.leftLeafId);
    const preferRight = freshSplittable.some((l) => l.id === result.rightLeafId);
    setSelectedNodeId(
      preferLeft
        ? result.leftLeafId
        : preferRight
          ? result.rightLeafId
          : (freshSplittable[0]?.id ?? null),
    );
  }

  function undo() {
    if (history.length <= 1) return;
    const nextHistory = history.slice(0, -1);
    const prevRoot = nextHistory[nextHistory.length - 1];
    setHistory(nextHistory);
    const leaves = splittableLeaves(prevRoot, data);
    setSelectedNodeId(leaves[0]?.id ?? prevRoot.id);
  }

  function handleSelectNode(nodeId: string) {
    if (!root) return;
    const node = findNodeById(root, nodeId);
    if (!node || node.kind !== "leaf") return;
    setSelectedNodeId(nodeId);
  }

  function handleSave() {
    if (!root) return;
    onSave?.(root);
    if (!tutorialMode) onClose();
  }

  if (!open || !root) return null;

  return (
    <div
      className={`${styles.overlay} ${tutorialMode ? styles.overlayTutorial : ""}`}
      style={originStyle}
      role="dialog"
      aria-modal="true"
      aria-label="Train yourself"
    >
      <div className={styles.surface}>
        <div className={styles.header}>
          <div>
            <p className={styles.eyebrow}>Interactive lab</p>
            <h2 className={styles.title}>Train yourself</h2>
            <p className={styles.status}>{statusText}</p>
          </div>
          <div className={styles.headerActions}>
            <Button size="sm" variant="secondary" onClick={reset}>
              Reset
            </Button>
            {tutorialMode ? null : (
              <Button size="sm" variant="secondary" onClick={onClose}>
                Close
              </Button>
            )}
          </div>
        </div>

        <div className={styles.controls}>
          <Cluster gap="sm">
            <span className={styles.controlLabel}>Split on</span>
            <Button
              size="sm"
              variant={splitFeature === 0 ? "accent" : "secondary"}
              onClick={() => setSplitFeature(0)}
            >
              Leaf width
              <kbd className={styles.kbd}>W</kbd>
            </Button>
            <Button
              size="sm"
              variant={splitFeature === 1 ? "accent" : "secondary"}
              onClick={() => setSplitFeature(1)}
            >
              Leaf length
              <kbd className={styles.kbd}>L</kbd>
            </Button>
            <span className={styles.shortcutHint}>
              or press <kbd className={styles.kbdInline}>D</kbd> to toggle
            </span>
          </Cluster>
          <Cluster gap="sm">
            <Button
              size="sm"
              variant="secondary"
              disabled={history.length <= 1}
              onClick={undo}
            >
              Undo split
            </Button>
            <Button
              size="sm"
              variant="accent"
              disabled={history.length <= 1}
              onClick={handleSave}
            >
              Save tree
            </Button>
          </Cluster>
        </div>

        <div className={styles.body}>
          <div className={styles.scatterPane}>
            <InteractiveSplitPlot
              samples={data}
              root={root}
              selectedNodeId={selectedNodeId}
              splitFeature={splitFeature}
              onSelectNode={handleSelectNode}
              onCommitSplit={handleCommitSplit}
              onAxisHint={setAxisHint}
            />
          </div>

          <div className={styles.treePane}>
            <TreeView
              root={root}
              title="Your tree"
              selectedNodeId={selectedNodeId}
              onNodeClick={(node) => handleSelectNode(node.id)}
            />
            {selectedNodeId && selectedSamples.length > 0 ? (
              <div className={styles.selectionDetail}>
                <p className={styles.selectionEyebrow}>Selected region</p>
                <p className={styles.selectionMeta}>
                  {selectedSamples.length} samples ·{" "}
                  {describeLeaf(selectedSamples).mixed
                    ? "mixed classes"
                    : "pure class"}
                </p>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
