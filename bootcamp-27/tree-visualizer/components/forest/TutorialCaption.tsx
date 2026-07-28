"use client";

import { Button } from "@/components/ui/Button";
import { STAGE_ORDER } from "@/lib/forest/stages";
import type { DemoStage } from "@/lib/forest/types";
import styles from "./TutorialCaption.module.css";

interface TutorialCaptionProps {
  stage: DemoStage;
  title: string;
  body: string;
  docked?: boolean;
  onBack: () => void;
  onNext: () => void;
  onSkip: () => void;
}

export function TutorialCaption({
  stage,
  title,
  body,
  docked = false,
  onBack,
  onNext,
  onSkip,
}: TutorialCaptionProps) {
  const stepIndex = STAGE_ORDER.indexOf(stage);
  const lastTutorialStep = STAGE_ORDER.indexOf("explore") - 1;
  const isLast = stepIndex >= lastTutorialStep;

  return (
    <div
      className={`${styles.root} ${docked ? styles.overlayDock : ""}`}
      role="region"
      aria-label="Tutorial"
    >
      <div className={styles.header}>
        <p className={styles.kicker}>
          Step {stepIndex + 1} of {lastTutorialStep + 1}
        </p>
        <div className={styles.progress} aria-hidden>
          {STAGE_ORDER.slice(0, -1).map((id, i) => (
            <span
              key={id}
              className={`${styles.dot} ${i === stepIndex ? styles.dotOn : ""} ${i < stepIndex ? styles.dotDone : ""}`}
            />
          ))}
        </div>
      </div>

      <div className={styles.content}>
        <h2 className={styles.title}>{title}</h2>
        <p className={styles.body}>{body}</p>
      </div>

      <div className={styles.footer}>
        <Button variant="ghost" size="sm" onClick={onSkip}>
          Skip
        </Button>
        <div className={styles.nav}>
          {stepIndex > 0 ? (
            <Button variant="secondary" size="sm" onClick={onBack}>
              Back
            </Button>
          ) : null}
          <Button variant="accent" size="sm" onClick={onNext}>
            {isLast ? "Start exploring" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  );
}
