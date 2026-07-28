"use client";

import { useId, useState, type ReactNode } from "react";
import styles from "./Tooltip.module.css";

interface TooltipProps {
  content: string;
  children: ReactNode;
  side?: "top" | "bottom";
}

export function Tooltip({ content, children, side = "top" }: TooltipProps) {
  const id = useId();
  const [open, setOpen] = useState(false);

  return (
    <span
      className={styles.wrap}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <button
        type="button"
        className={styles.trigger}
        aria-describedby={open ? id : undefined}
      >
        {children}
      </button>
      {open ? (
        <span
          id={id}
          role="tooltip"
          className={`${styles.tip} ${styles[side]}`}
        >
          {content}
        </span>
      ) : null}
    </span>
  );
}

export function HelpHint({ label }: { label: string }) {
  return (
    <Tooltip content={label}>
      <span className={styles.hint} aria-label="More info">
        ?
      </span>
    </Tooltip>
  );
}
