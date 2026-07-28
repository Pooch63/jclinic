import type { ReactNode } from "react";
import styles from "./Header.module.css";

interface HeaderProps {
  brand: string;
  tagline?: string;
  actions?: ReactNode;
}

export function Header({ brand, tagline, actions }: HeaderProps) {
  return (
    <header className={styles.header} data-tour="brand">
      <div className={styles.brandBlock}>
        <span className={styles.mark} aria-hidden />
        <div>
          <p className={styles.brand}>{brand}</p>
          {tagline ? <p className={styles.tagline}>{tagline}</p> : null}
        </div>
      </div>
      {actions ? <div className={styles.actions}>{actions}</div> : null}
    </header>
  );
}
