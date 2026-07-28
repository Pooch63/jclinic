"use client";

import { useCallback, useEffect, useId, useMemo, useState } from "react";
import {
  CLASS_COLORS,
  CLASS_LABELS,
  type ClassLabel,
  type Sample,
  type TreeNode,
} from "@/lib/forest/types";
import { nodeRegions } from "@/lib/forest/trainViz";
import { revealedLeaves, revealedSplits } from "@/lib/forest/trainViz";
import {
  canSplitOnAxis,
  leafAtPoint,
  partitionBySplit,
  samplesAtNode,
  thresholdForClick,
} from "@/lib/forest/manualTree";
import styles from "./ScatterPlot.module.css";
import plotStyles from "./InteractiveSplitPlot.module.css";

interface InteractiveSplitPlotProps {
  samples: Sample[];
  root: TreeNode;
  selectedNodeId: string | null;
  splitFeature: 0 | 1;
  onSelectNode: (nodeId: string) => void;
  onCommitSplit: (nodeId: string, feature: 0 | 1, threshold: number) => void;
  onAxisHint?: (hint: string | null) => void;
}

const W = 420;
const H = 320;
const PAD = 36;

function toX(v: number) {
  return PAD + v * (W - PAD * 2);
}

function toY(v: number) {
  return H - PAD - v * (H - PAD * 2);
}

export function InteractiveSplitPlot({
  samples,
  root,
  selectedNodeId,
  splitFeature,
  onSelectNode,
  onCommitSplit,
  onAxisHint,
}: InteractiveSplitPlotProps) {
  const uid = useId().replace(/:/g, "");
  const gridId = `split-grid-${uid}`;
  const [hover, setHover] = useState<{
    nodeId: string;
    threshold: number | null;
  } | null>(null);

  useEffect(() => {
    setHover(null);
    onAxisHint?.(null);
  }, [splitFeature, root, onAxisHint]);

  const regions = useMemo(() => nodeRegions(root), [root]);
  const splits = revealedSplits(root, Number.MAX_SAFE_INTEGER);
  const leaves = revealedLeaves(root, Number.MAX_SAFE_INTEGER);

  const selectedRegion = selectedNodeId
    ? regions.get(selectedNodeId)
    : null;

  const activeNodeId = hover?.nodeId ?? selectedNodeId;
  const activeRegion = activeNodeId ? regions.get(activeNodeId) : null;
  const previewThreshold = hover?.threshold ?? null;

  const activeSamples =
    activeNodeId != null
      ? (samplesAtNode(root, activeNodeId, samples) ?? [])
      : [];

  const previewSides =
    previewThreshold != null && activeRegion
      ? partitionBySplit(activeSamples, splitFeature, previewThreshold)
      : null;

  const previewLeftIds = previewSides
    ? new Set(previewSides.left.map((s) => s.id))
    : undefined;
  const previewRightIds = previewSides
    ? new Set(previewSides.right.map((s) => s.id))
    : undefined;

  const clientToData = useCallback((clientX: number, clientY: number) => {
    const svg = document.getElementById(`split-svg-${uid}`);
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const sx = ((clientX - rect.left) / rect.width) * W;
    const sy = ((clientY - rect.top) / rect.height) * H;
    const x = (sx - PAD) / (W - PAD * 2);
    const y = (H - PAD - sy) / (H - PAD * 2);
    return { x, y };
  }, [uid]);

  const resolveThreshold = useCallback(
    (nodeId: string, pt: { x: number; y: number }) => {
      const nodeSamples = samplesAtNode(root, nodeId, samples) ?? [];
      const region = regions.get(nodeId) ?? undefined;
      return thresholdForClick(
        nodeSamples,
        splitFeature,
        splitFeature === 0 ? pt.x : pt.y,
        region,
      );
    },
    [root, samples, splitFeature, regions],
  );

  const updateAxisHint = useCallback(
    (nodeId: string) => {
      if (!onAxisHint) return;
      const nodeSamples = samplesAtNode(root, nodeId, samples) ?? [];
      if (nodeSamples.length < 2) {
        onAxisHint(null);
        return;
      }
      if (canSplitOnAxis(nodeSamples, splitFeature)) {
        onAxisHint(null);
        return;
      }
      const other: 0 | 1 = splitFeature === 0 ? 1 : 0;
      if (canSplitOnAxis(nodeSamples, other)) {
        onAxisHint(
          splitFeature === 0
            ? "Points share the same width here — press D or L to split on length."
            : "Points share the same length here — press D or W to split on width.",
        );
        return;
      }
      onAxisHint(null);
    },
    [root, samples, splitFeature, onAxisHint],
  );

  const handleMouseMove = (e: React.MouseEvent) => {
    const pt = clientToData(e.clientX, e.clientY);
    if (!pt) return;

    const leaf = leafAtPoint(root, pt.x, pt.y);
    if (!leaf) {
      setHover(null);
      onAxisHint?.(null);
      return;
    }

    const threshold = resolveThreshold(leaf.id, pt);
    if (threshold == null) {
      setHover({ nodeId: leaf.id, threshold: null });
      updateAxisHint(leaf.id);
      return;
    }

    onAxisHint?.(null);
    setHover({ nodeId: leaf.id, threshold });
  };

  const handleMouseLeave = () => {
    setHover(null);
    onAxisHint?.(null);
  };

  const handleClick = (e: React.MouseEvent) => {
    const pt = clientToData(e.clientX, e.clientY);
    if (!pt) return;

    const leaf = leafAtPoint(root, pt.x, pt.y);
    if (!leaf) return;

    onSelectNode(leaf.id);

    const threshold = resolveThreshold(leaf.id, pt);
    if (threshold == null) {
      updateAxisHint(leaf.id);
      return;
    }

    onCommitSplit(leaf.id, splitFeature, threshold);
    setHover(null);
    onAxisHint?.(null);
  };

  const showPreview =
    previewThreshold != null && activeRegion;

  return (
    <div className={styles.root}>
      <div className={styles.caption}>
        <span>Draw a boundary</span>
        <span className={styles.legend}>
          {(Object.keys(CLASS_COLORS) as ClassLabel[]).map((c) => (
            <span key={c} className={styles.legendItem}>
              <i style={{ background: CLASS_COLORS[c] }} />
              {CLASS_LABELS[c]}
            </span>
          ))}
        </span>
      </div>
      <svg
        id={`split-svg-${uid}`}
        viewBox={`0 0 ${W} ${H}`}
        className={`${styles.svg} ${plotStyles.interactiveSvg}`}
        role="img"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        <defs>
          <pattern
            id={gridId}
            width="28"
            height="28"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 28 0 L 0 0 0 28"
              fill="none"
              stroke="rgba(45,106,79,0.08)"
              strokeWidth="1"
            />
          </pattern>
        </defs>
        <rect x="0" y="0" width={W} height={H} fill={`url(#${gridId})`} rx="16" />

        {leaves.map((leaf) => {
          const x = toX(leaf.region.x0);
          const y = toY(leaf.region.y1);
          const width = Math.max(0, toX(leaf.region.x1) - toX(leaf.region.x0));
          const height = Math.max(0, toY(leaf.region.y0) - toY(leaf.region.y1));
          const selected = leaf.nodeId === selectedNodeId;
          const hovered = leaf.nodeId === hover?.nodeId;
          return (
            <rect
              key={`leaf-${leaf.nodeId}`}
              x={x}
              y={y}
              width={width}
              height={height}
              fill={leaf.color}
              className={`${styles.leafTint} ${selected ? plotStyles.regionSelected : ""} ${hovered ? plotStyles.regionHovered : ""}`}
              style={{ pointerEvents: "none" }}
            />
          );
        })}

        {selectedRegion ? (
          <rect
            x={toX(selectedRegion.x0)}
            y={toY(selectedRegion.y1)}
            width={Math.max(0, toX(selectedRegion.x1) - toX(selectedRegion.x0))}
            height={Math.max(0, toY(selectedRegion.y0) - toY(selectedRegion.y1))}
            className={plotStyles.selectionOutline}
            pointerEvents="none"
          />
        ) : null}

        {splits.map((split) => {
          const { region, feature, threshold, nodeId } = split;
          const isVertical = feature === 0;
          const x1 = isVertical ? toX(threshold) : toX(region.x0);
          const y1 = isVertical ? toY(region.y1) : toY(threshold);
          const x2 = isVertical ? toX(threshold) : toX(region.x1);
          const y2 = isVertical ? toY(region.y0) : toY(threshold);
          return (
            <line
              key={`split-${nodeId}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              className={styles.splitLine}
            />
          );
        })}

        {showPreview && activeRegion ? (() => {
          const isVertical = splitFeature === 0;
          const x1 = isVertical
            ? toX(previewThreshold!)
            : toX(activeRegion.x0);
          const y1 = isVertical
            ? toY(activeRegion.y1)
            : toY(previewThreshold!);
          const x2 = isVertical
            ? toX(previewThreshold!)
            : toX(activeRegion.x1);
          const y2 = isVertical
            ? toY(activeRegion.y0)
            : toY(previewThreshold!);
          return (
            <line
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              className={`${styles.splitLine} ${styles.splitLineActive}`}
            />
          );
        })() : null}

        <text x={W / 2} y={H - 8} textAnchor="middle" className={styles.axis}>
          Leaf width →
        </text>
        <text
          x={14}
          y={H / 2}
          textAnchor="middle"
          className={styles.axis}
          transform={`rotate(-90 14 ${H / 2})`}
        >
          Leaf length →
        </text>

        {samples.map((s) => {
          const inActive = activeSamples.some((x) => x.id === s.id);
          const onLeft = previewLeftIds?.has(s.id);
          const onRight = previewRightIds?.has(s.id);
          const splitSide =
            hover && hover.threshold != null && inActive
              ? onLeft
                ? "left"
                : onRight
                  ? "right"
                  : null
              : null;

          return (
            <circle
              key={s.id}
              cx={toX(s.x)}
              cy={toY(s.y)}
              r={splitSide ? 7.5 : inActive ? 6.5 : 5.5}
              fill={CLASS_COLORS[s.label]}
              className={[
                styles.dot,
                !inActive && selectedNodeId ? styles.dim : "",
                splitSide === "left" ? styles.splitLeft : "",
                splitSide === "right" ? styles.splitRight : "",
              ]
                .filter(Boolean)
                .join(" ")}
              pointerEvents="none"
            >
              <title>
                {CLASS_LABELS[s.label]} · width {s.x.toFixed(2)} · length{" "}
                {s.y.toFixed(2)}
              </title>
            </circle>
          );
        })}
      </svg>
      <p className={plotStyles.hint}>
        Click in a region to place a{" "}
        {splitFeature === 0 ? "vertical (width)" : "horizontal (length)"} line.
        Press <kbd className={plotStyles.kbd}>D</kbd> to switch axis.
      </p>
    </div>
  );
}
