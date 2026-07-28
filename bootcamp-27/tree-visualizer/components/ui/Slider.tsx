"use client";

import styles from "./Slider.module.css";
import { HelpHint } from "./Tooltip";

interface SliderProps {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}

export function Slider({
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => String(v),
}: SliderProps) {
  return (
    <label className={styles.root}>
      <span className={styles.row}>
        <span className={styles.label}>
          {label}
          {hint ? <HelpHint label={hint} /> : null}
        </span>
        <span className={styles.value}>{format(value)}</span>
      </span>
      <input
        className={styles.input}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
