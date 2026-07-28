import Image from "next/image";
import type { ReactNode } from "react";
import styles from "./Header.module.css";

interface HeaderProps {
  brand: string;
  tagline?: string;
  actions?: ReactNode;
  logoSrc?: string;
}

export function Header({
  brand,
  tagline,
  actions,
  logoSrc = "/jclinic.jpg",
}: HeaderProps) {
  return (
    <header className={styles.header} data-tour="brand">
      <div className={styles.brandBlock}>
        <Image
          src={logoSrc}
          alt=""
          width={34}
          height={34}
          className={styles.logo}
          priority
        />
        <div>
          <p className={styles.brand}>{brand}</p>
          {tagline ? <p className={styles.tagline}>{tagline}</p> : null}
        </div>
      </div>
      {actions ? <div className={styles.actions}>{actions}</div> : null}
    </header>
  );
}
