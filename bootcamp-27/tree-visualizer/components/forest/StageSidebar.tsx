"use client";

import { getStage, STAGE_ORDER } from "@/lib/forest/stages";
import type { DemoStage } from "@/lib/forest/types";
import { Button } from "@/components/ui/Button";
import { Cluster, Stack } from "@/components/ui/Stack";
import styles from "./StageSidebar.module.css";

interface StageSidebarProps {
  stage: DemoStage;
  liveNotes: string[];
  onJump: (stage: DemoStage) => void;
}

export function StageSidebar({ stage, liveNotes, onJump }: StageSidebarProps) {
  const info = getStage(stage);

  return (
    <aside className={styles.sidebar} aria-label="What’s happening" data-tour="sidebar">
      <p className={styles.eyebrow}>{info.eyebrow}</p>
      <h2 className={styles.title}>{info.title}</h2>
      <p className={styles.summary}>{info.summary}</p>

      <Stack gap="sm">
        {info.bullets.length > 0 ? (
          <>
            <p className={styles.sectionLabel}>At this stage</p>
            <ul className={styles.bullets}>
              {info.bullets.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
          </>
        ) : null}
      </Stack>

      {info.tip ? (
        <div className={styles.tip}>
          <span className={styles.tipLabel}>Tip</span>
          <p>{info.tip}</p>
        </div>
      ) : null}

      {liveNotes.length > 0 ? (
        <div className={styles.live}>
          <p className={styles.sectionLabel}>Right now</p>
          <ul className={styles.liveList}>
            {liveNotes.map((n, i) => (
              <li key={`${i}-${n}`} className={styles.liveItem}>
                {n}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className={styles.nav} data-tour="nav">
        <p className={styles.sectionLabel}>Jump to</p>
        <Cluster gap="xs">
          {STAGE_ORDER.map((id, i) => (
            <Button
              key={id}
              size="sm"
              variant={id === stage ? "accent" : "ghost"}
              active={id === stage}
              onClick={() => onJump(id)}
              aria-current={id === stage ? "step" : undefined}
            >
              {i + 1}
            </Button>
          ))}
        </Cluster>
      </div>
    </aside>
  );
}
