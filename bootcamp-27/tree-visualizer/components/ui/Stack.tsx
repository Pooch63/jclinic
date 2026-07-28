import type { ReactNode } from "react";
import styles from "./Stack.module.css";

interface StackProps {
  children: ReactNode;
  gap?: "xs" | "sm" | "md" | "lg" | "xl";
  className?: string;
}

export function Stack({ children, gap = "md", className = "" }: StackProps) {
  return (
    <div className={`${styles.stack} ${styles[gap]} ${className}`.trim()}>
      {children}
    </div>
  );
}

interface ClusterProps {
  children: ReactNode;
  gap?: "xs" | "sm" | "md";
  className?: string;
}

export function Cluster({
  children,
  gap = "sm",
  className = "",
}: ClusterProps) {
  return (
    <div className={`${styles.cluster} ${styles[`c-${gap}`]} ${className}`.trim()}>
      {children}
    </div>
  );
}
