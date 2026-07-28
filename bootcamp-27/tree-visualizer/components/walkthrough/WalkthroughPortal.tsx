"use client";

import { useEffect, useLayoutEffect, useState } from "react";
import { Button } from "@/components/ui/Button";
import { TOUR_STEPS, type TourStep } from "@/lib/forest/tour";
import styles from "./WalkthroughPortal.module.css";

interface WalkthroughPortalProps {
  open: boolean;
  step: number;
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
}

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
  bottom: number;
  right: number;
}

const PAD = 10;

function readTargetRect(target: TourStep["target"]): Rect | null {
  const el = document.querySelector(`[data-tour="${target}"]`);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 2 && r.height < 2) return null;
  return {
    top: r.top,
    left: r.left,
    width: r.width,
    height: r.height,
    bottom: r.bottom,
    right: r.right,
  };
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

function placeCard(
  rect: Rect,
  placement: TourStep["placement"],
  cardW: number,
  cardH: number,
) {
  const gap = 14;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const margin = 12;

  const candidates: Array<{ top: number; left: number; placement: string }> = [];

  const order = [
    placement ?? "bottom",
    "bottom",
    "top",
    "right",
    "left",
  ].filter((v, i, a) => a.indexOf(v) === i);

  for (const p of order) {
    let top = 0;
    let left = 0;
    if (p === "bottom") {
      top = rect.bottom + gap;
      left = rect.left + rect.width / 2 - cardW / 2;
    } else if (p === "top") {
      top = rect.top - gap - cardH;
      left = rect.left + rect.width / 2 - cardW / 2;
    } else if (p === "right") {
      top = rect.top + rect.height / 2 - cardH / 2;
      left = rect.right + gap;
    } else {
      top = rect.top + rect.height / 2 - cardH / 2;
      left = rect.left - gap - cardW;
    }
    candidates.push({
      top: clamp(top, margin, vh - cardH - margin),
      left: clamp(left, margin, vw - cardW - margin),
      placement: p,
    });
  }

  return candidates[0];
}

export function WalkthroughPortal({
  open,
  step,
  onNext,
  onBack,
  onSkip,
}: WalkthroughPortalProps) {
  const current = TOUR_STEPS[step] ?? TOUR_STEPS[0];
  const last = step >= TOUR_STEPS.length - 1;
  const [rect, setRect] = useState<Rect | null>(null);
  const [cardPos, setCardPos] = useState({ top: 80, left: 24, placement: "bottom" });

  useLayoutEffect(() => {
    if (!open) return;

    let cancelled = false;
    let tries = 0;

    const measure = () => {
      const next = readTargetRect(current.target);
      if (!next) {
        if (tries < 20) {
          tries += 1;
          requestAnimationFrame(measure);
        }
        return;
      }
      if (cancelled) return;
      setRect(next);

      const el = document.querySelector(`[data-tour="${current.target}"]`);
      el?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });

      const cardW = Math.min(360, window.innerWidth - 24);
      const cardH = 220;
      setCardPos(placeCard(next, current.placement, cardW, cardH));
    };

    measure();

    const onResize = () => measure();
    window.addEventListener("resize", onResize);
    window.addEventListener("scroll", onResize, true);
    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      window.removeEventListener("scroll", onResize, true);
    };
  }, [open, step, current.target, current.placement]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onSkip();
      if (e.key === "ArrowRight" || e.key === "Enter") onNext();
      if (e.key === "ArrowLeft" && step > 0) onBack();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, step, onNext, onBack, onSkip]);

  if (!open) return null;

  const highlight = rect
    ? {
        top: Math.max(0, rect.top - PAD),
        left: Math.max(0, rect.left - PAD),
        width: rect.width + PAD * 2,
        height: rect.height + PAD * 2,
      }
    : null;

  return (
    <div className={styles.root} role="dialog" aria-modal="true" aria-labelledby="tour-title">
      <div
        className={`${styles.dim} ${highlight ? "" : styles.dimFallback}`}
        aria-hidden
      />

      {highlight ? (
        <div
          className={styles.spotlight}
          style={{
            top: highlight.top,
            left: highlight.left,
            width: highlight.width,
            height: highlight.height,
          }}
          aria-hidden
        />
      ) : null}

      <div
        className={styles.card}
        style={{ top: cardPos.top, left: cardPos.left }}
        data-placement={cardPos.placement}
      >
        <p className={styles.kicker}>
          Tour · {step + 1} of {TOUR_STEPS.length}
        </p>
        <h2 id="tour-title" className={styles.title}>
          {current.title}
        </h2>
        <p className={styles.body}>{current.body}</p>

        <div className={styles.dots} aria-hidden>
          {TOUR_STEPS.map((s, i) => (
            <span
              key={s.id}
              className={`${styles.dot} ${i === step ? styles.dotOn : ""} ${i < step ? styles.dotDone : ""}`}
            />
          ))}
        </div>

        <div className={styles.actions}>
          <Button variant="ghost" size="sm" onClick={onSkip}>
            Skip tour
          </Button>
          <div className={styles.navBtns}>
            {step > 0 ? (
              <Button variant="secondary" size="sm" onClick={onBack}>
                Back
              </Button>
            ) : null}
            <Button variant="accent" onClick={onNext}>
              {last ? "Start exploring" : "Next"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
