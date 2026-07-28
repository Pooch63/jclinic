"use client";

import { useId } from "react";
import {
  CLASS_COLORS,
  CLASS_LABELS,
  type ClassLabel,
  type Sample,
} from "@/lib/forest/types";
import type { LeafRegionViz, SplitViz } from "@/lib/forest/trainViz";
import styles from "./ScatterPlot.module.css";

interface ScatterPlotProps {
  samples: Sample[];
  highlightIds?: Set<number>;
  dimIds?: Set<number>;
  selectedId?: number | null;
  onSelect?: (sample: Sample) => void;
  pulseId?: number | null;
  title?: string;
  /** Decision splits to draw as threshold lines in feature space */
  splits?: SplitViz[];
  /** Id of the most recently revealed split (animated) */
  activeSplitId?: string | null;
  /** Leaf partitions tinted by predicted class */
  leafRegions?: LeafRegionViz[];
  /** Point ids that fall on the "left" of the active split */
  splitLeftIds?: Set<number>;
  /** Point ids that fall on the "right" of the active split */
  splitRightIds?: Set<number>;
  compact?: boolean;
}

export function ScatterPlot({
  samples,
  highlightIds,
  dimIds,
  selectedId,
  onSelect,
  pulseId,
  title = "Feature space",
  splits,
  activeSplitId,
  leafRegions,
  splitLeftIds,
  splitRightIds,
  compact = false,
}: ScatterPlotProps) {
  const uid = useId().replace(/:/g, "");
  const gridId = `scatter-grid-${uid}`;
  const w = compact ? 360 : 420;
  const h = compact ? 280 : 320;
  const pad = compact ? 28 : 36;

  const toX = (v: number) => pad + v * (w - pad * 2);
  const toY = (v: number) => h - pad - v * (h - pad * 2);

  const showSplitSides = Boolean(splitLeftIds || splitRightIds);

  return (
    <div className={styles.root}>
      <div className={styles.caption}>
        <span>{title}</span>
        <span className={styles.legend}>
          {(Object.keys(CLASS_COLORS) as ClassLabel[]).map((c) => (
            <span key={c} className={styles.legendItem}>
              <i style={{ background: CLASS_COLORS[c] }} />
              {CLASS_LABELS[c]}
            </span>
          ))}
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className={styles.svg} role="img">
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
        <rect x="0" y="0" width={w} height={h} fill={`url(#${gridId})`} rx="16" />

        {leafRegions?.map((leaf) => {
          const x = toX(leaf.region.x0);
          const y = toY(leaf.region.y1);
          const width = Math.max(0, toX(leaf.region.x1) - toX(leaf.region.x0));
          const height = Math.max(0, toY(leaf.region.y0) - toY(leaf.region.y1));
          return (
            <rect
              key={`leaf-${leaf.nodeId}`}
              x={x}
              y={y}
              width={width}
              height={height}
              fill={leaf.color}
              className={styles.leafTint}
            />
          );
        })}

        {splits?.map((split) => {
          const { region, feature, threshold, nodeId } = split;
          const active = nodeId === activeSplitId;
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
              className={`${styles.splitLine} ${active ? styles.splitLineActive : ""}`}
            />
          );
        })}

        <text x={w / 2} y={h - 8} textAnchor="middle" className={styles.axis}>
          Leaf width →
        </text>
        <text
          x={14}
          y={h / 2}
          textAnchor="middle"
          className={styles.axis}
          transform={`rotate(-90 14 ${h / 2})`}
        >
          Leaf length →
        </text>

        {samples.map((s) => {
          const highlighted = highlightIds?.has(s.id);
          const dimmed = dimIds?.has(s.id) || (highlightIds && !highlighted);
          const selected = selectedId === s.id;
          const pulsing = pulseId === s.id;
          const onLeft = splitLeftIds?.has(s.id);
          const onRight = splitRightIds?.has(s.id);
          const splitSide =
            showSplitSides && highlighted
              ? onLeft
                ? "left"
                : onRight
                  ? "right"
                  : null
              : null;

          return (
            <g key={s.id}>
              {pulsing ? (
                <circle
                  cx={toX(s.x)}
                  cy={toY(s.y)}
                  r={14}
                  className={styles.pulse}
                  fill={CLASS_COLORS[s.label]}
                />
              ) : null}
              <circle
                cx={toX(s.x)}
                cy={toY(s.y)}
                r={
                  selected
                    ? 8
                    : splitSide
                      ? 7.5
                      : highlighted
                        ? 7
                        : 5.5
                }
                fill={CLASS_COLORS[s.label]}
                className={[
                  styles.dot,
                  dimmed ? styles.dim : "",
                  selected ? styles.selected : "",
                  splitSide === "left" ? styles.splitLeft : "",
                  splitSide === "right" ? styles.splitRight : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                onClick={() => onSelect?.(s)}
                style={{ cursor: onSelect ? "pointer" : "default" }}
              >
                <title>
                  {CLASS_LABELS[s.label]} · width {s.x.toFixed(2)} · length{" "}
                  {s.y.toFixed(2)}
                </title>
              </circle>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
