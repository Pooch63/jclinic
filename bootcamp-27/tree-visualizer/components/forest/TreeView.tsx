"use client";

import { useMemo } from "react";
import { layoutTree } from "@/lib/forest/layout";
import {
  CLASS_COLORS,
  CLASS_LABELS,
  FEATURE_NAMES,
  type PathStep,
  type TreeNode,
} from "@/lib/forest/types";
import styles from "./TreeView.module.css";

interface TreeViewProps {
  root: TreeNode;
  title?: string;
  /** how many nodes to reveal (for grow animation); undefined = all */
  revealCount?: number;
  /** active path node ids during prediction */
  activePath?: PathStep[];
  /** current index along the path for animation */
  pathIndex?: number;
  /** flow direction for particle animation */
  flow?: "forward" | "none";
  compact?: boolean;
  /** highlight a node (e.g. user picked it in expand view) */
  selectedNodeId?: string | null;
  /** when set, nodes become clickable */
  onNodeClick?: (node: TreeNode) => void;
}

function collectIds(node: TreeNode, order: string[] = []): string[] {
  order.push(node.id);
  if (node.kind === "decision") {
    collectIds(node.left, order);
    collectIds(node.right, order);
  }
  return order;
}

export function TreeView({
  root,
  title = "Decision tree",
  revealCount,
  activePath = [],
  pathIndex = -1,
  flow = "none",
  compact = false,
  selectedNodeId = null,
  onNodeClick,
}: TreeViewProps) {
  const interactive = Boolean(onNodeClick);
  const width = compact ? 360 : 520;
  const height = compact ? 260 : 360;

  const growthOrder = useMemo(() => collectIds(root), [root]);
  const revealed = useMemo(() => {
    if (revealCount === undefined) return new Set(growthOrder);
    return new Set(growthOrder.slice(0, Math.max(0, revealCount)));
  }, [growthOrder, revealCount]);

  const layout = useMemo(
    () => layoutTree(root, width, height, compact ? 28 : 40),
    [root, width, height, compact],
  );

  const byId = useMemo(() => {
    const m = new Map(layout.map((n) => [n.id, n]));
    return m;
  }, [layout]);

  const activeIds = useMemo(() => {
    if (pathIndex < 0) return new Set<string>();
    return new Set(activePath.slice(0, pathIndex + 1).map((p) => p.nodeId));
  }, [activePath, pathIndex]);

  const currentId =
    pathIndex >= 0 && pathIndex < activePath.length
      ? activePath[pathIndex].nodeId
      : null;

  const particles = useMemo(() => {
    if (flow !== "forward" || pathIndex < 1) return [];
    const from = activePath[pathIndex - 1]?.nodeId;
    const to = activePath[pathIndex]?.nodeId;
    if (!from || !to) return [];
    const a = byId.get(from);
    const b = byId.get(to);
    if (!a || !b) return [];
    return [{ x1: a.x, y1: a.y, x2: b.x, y2: b.y, key: `${from}-${to}-${pathIndex}` }];
  }, [flow, pathIndex, activePath, byId]);

  return (
    <div className={styles.root}>
      <div className={styles.caption}>
        <span>{title}</span>
        {flow === "forward" ? (
          <span className={styles.flowBadge}>↓ Forward pass</span>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className={styles.svg}>
        {layout.map((n) => {
          if (!n.parentId || !revealed.has(n.id) || !revealed.has(n.parentId))
            return null;
          const parent = byId.get(n.parentId);
          if (!parent) return null;
          const onPath =
            activeIds.has(n.id) && activeIds.has(n.parentId);
          return (
            <line
              key={`e-${n.id}`}
              x1={parent.x}
              y1={parent.y}
              x2={n.x}
              y2={n.y}
              className={`${styles.edge} ${onPath ? styles.edgeActive : ""}`}
            />
          );
        })}

        {particles.map((p) => (
          <g key={p.key}>
            <circle r="5" className={styles.particle} fill="var(--color-sun)">
              <animateMotion
                dur="0.55s"
                repeatCount="1"
                path={`M ${p.x1} ${p.y1} L ${p.x2} ${p.y2}`}
                fill="freeze"
              />
            </circle>
          </g>
        ))}

        {layout.map((n) => {
          if (!revealed.has(n.id)) return null;
          const node = n.node;
          const isLeaf = node.kind === "leaf";
          const isCurrent = n.id === currentId;
          const isSelected = n.id === selectedNodeId;
          const onPath = activeIds.has(n.id);
          const label = isLeaf
            ? CLASS_LABELS[node.prediction]
            : `${FEATURE_NAMES[node.feature]} ≤ ${node.threshold.toFixed(2)}`;

          const fill = isLeaf
            ? CLASS_COLORS[node.prediction]
            : onPath
              ? "var(--color-moss)"
              : "var(--color-paper)";

          function handleNodeClick() {
            onNodeClick?.(node);
          }

          return (
            <g key={n.id} transform={`translate(${n.x}, ${n.y})`}>
              <g
                className={`${styles.node} ${isCurrent ? styles.current : ""} ${isSelected ? styles.selected : ""} ${interactive ? styles.interactive : ""}`}
                onClick={interactive ? handleNodeClick : undefined}
                onKeyDown={
                  interactive
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleNodeClick();
                        }
                      }
                    : undefined
                }
                role={interactive ? "button" : undefined}
                tabIndex={interactive ? 0 : undefined}
              >
                {interactive ? (
                  isLeaf ? (
                    <rect
                      x={-40}
                      y={-20}
                      width={80}
                      height={40}
                      fill="transparent"
                      className={styles.hitTarget}
                    />
                  ) : (
                    <circle
                      r={compact ? 22 : 26}
                      fill="transparent"
                      className={styles.hitTarget}
                    />
                  )
                ) : null}
                {isLeaf ? (
                  <rect
                    x={-36}
                    y={-16}
                    width={72}
                    height={32}
                    rx={10}
                    fill={fill}
                    className={`${styles.leaf} ${onPath ? styles.onPath : ""}`}
                  />
                ) : (
                  <circle
                    r={compact ? 16 : 20}
                    fill={fill}
                    className={`${styles.decision} ${onPath ? styles.onPath : ""}`}
                  />
                )}
                <text
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className={`${styles.label} ${isLeaf || onPath ? styles.labelLight : ""}`}
                >
                  {isLeaf
                    ? label
                    : compact
                      ? `f${node.feature}`
                      : label.length > 18
                        ? label.slice(0, 16) + "…"
                        : label}
                </text>
                {!isLeaf && !compact ? (
                  <text y={28} textAnchor="middle" className={styles.sub}>
                    n={node.sampleCount}
                  </text>
                ) : null}
              </g>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
