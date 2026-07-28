import type { CSSProperties } from "react";
import {
  CLASS_COLORS,
  CLASS_LABELS,
  type ClassLabel,
  type ForestPrediction,
} from "@/lib/forest/types";
import styles from "./EnsembleMerge.module.css";

interface EnsembleMergeProps {
  prediction: ForestPrediction | null;
  /** how many tree votes to reveal (0..n) */
  revealVotes: number;
  /** whether final merge animation is active */
  merging: boolean;
  /** hide the true label line (e.g. for unknown query points) */
  hideTrueLabel?: boolean;
  /** highlight a tree that voted differently from the majority */
  highlightTreeId?: number | null;
}

export function EnsembleMerge({
  prediction,
  revealVotes,
  merging,
  hideTrueLabel = false,
  highlightTreeId = null,
}: EnsembleMergeProps) {
  if (!prediction) {
    return (
      <div className={styles.empty}>
        Pick a sample and run a prediction to watch the trees vote.
      </div>
    );
  }

  const trees = prediction.treePredictions;
  const shown = trees.slice(0, revealVotes);
  const classes = Object.keys(CLASS_COLORS) as ClassLabel[];

  return (
    <div className={styles.root}>
      <div className={styles.caption}>
        <span>Tree votes → forest decision</span>
        <span className={styles.count}>
          {revealVotes}/{trees.length} trees
        </span>
      </div>

      <div className={styles.stage}>
        <div className={styles.voters}>
          {trees.map((tp, i) => {
            const visible = i < revealVotes;
            const isWrong =
              highlightTreeId != null &&
              tp.treeId === highlightTreeId &&
              tp.prediction !== prediction.prediction;
            return (
              <div
                key={tp.treeId}
                className={`${styles.voter} ${visible ? styles.visible : ""} ${isWrong ? styles.voterWrong : ""}`}
                style={
                  {
                    "--delay": `${i * 70}ms`,
                    "--vote-color": CLASS_COLORS[tp.prediction],
                  } as CSSProperties
                }
              >
                <span className={styles.voterLabel}>T{tp.treeId + 1}</span>
                <span className={styles.voterChip}>
                  {visible ? CLASS_LABELS[tp.prediction] : "…"}
                </span>
              </div>
            );
          })}
        </div>

        <div className={`${styles.streams} ${merging ? styles.merging : ""}`}>
          {shown.map((tp) => (
            <span
              key={`stream-${tp.treeId}`}
              className={styles.stream}
              style={{ background: CLASS_COLORS[tp.prediction] }}
            />
          ))}
        </div>

        <div
          className={`${styles.ballot} ${merging ? styles.ballotLive : ""}`}
        >
          <p className={styles.ballotTitle}>Forest ballot</p>
          <div className={styles.bars}>
            {classes.map((c) => {
              const v = prediction.votes[c];
              const max = Math.max(...classes.map((k) => prediction.votes[k]), 1);
              const width = merging ? (v / max) * 100 : 0;
              const revealedShare = shown.filter((t) => t.prediction === c).length;
              return (
                <div key={c} className={styles.barRow}>
                  <span className={styles.barLabel}>{CLASS_LABELS[c]}</span>
                  <div className={styles.barTrack}>
                    <div
                      className={styles.barFill}
                      style={{
                        width: `${merging ? width : (revealedShare / max) * 100}%`,
                        background: CLASS_COLORS[c],
                      }}
                    />
                  </div>
                  <span className={styles.barValue}>
                    {merging ? v : revealedShare}
                  </span>
                </div>
              );
            })}
          </div>

          <div
            className={`${styles.final} ${merging ? styles.finalShow : ""}`}
            style={
              {
                "--final-color": CLASS_COLORS[prediction.prediction],
              } as CSSProperties
            }
          >
            <span className={styles.finalEyebrow}>Majority vote</span>
            <span className={styles.finalLabel}>
              {CLASS_LABELS[prediction.prediction]}
            </span>
            {!hideTrueLabel ? (
              <span className={styles.finalNote}>
                True label: {CLASS_LABELS[prediction.sample.label]}
                {prediction.prediction === prediction.sample.label
                  ? " · correct"
                  : " · edge case!"}
              </span>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
