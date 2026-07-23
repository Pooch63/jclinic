"""
sort.py — Sort graded students into skill tiers
Usage:
    python sort.py <grades.csv>
"""

import csv
import sys
from pathlib import Path


BEGINNER     = "Beginner"
INTERMEDIATE = "Intermediate"
ADVANCED     = "Advanced"

TIERS = [BEGINNER, INTERMEDIATE, ADVANCED]


def classify(row: dict) -> str:
    score   = int(row["score"])
    p1_pass = row["p1"] == "Solved"
    p2_pass = row["p2"] == "Solved"

    if score >= 4:
        return ADVANCED
    elif p1_pass and p2_pass:
        return INTERMEDIATE
    else:
        return BEGINNER


def main():
    if len(sys.argv) < 2:
        print("Usage: python sort.py <grades.csv>")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.is_file():
        print(f"File not found: {src}")
        sys.exit(1)

    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No results found in CSV.")
        sys.exit(0)

    for row in rows:
        row["tier"] = classify(row)

    # Console output 
    grouped = {tier: [] for tier in TIERS}
    for row in rows:
        grouped[row["tier"]].append(row)

    print()
    for tier in TIERS:
        for s in grouped[tier]:
            name = f"{s['first_name']} {s['last_name']}"
            print(f"{name}: {tier} ({s['score']}/{s['total']})")
    print()

    # CSV output 
    out_path = src.parent / f"{src.stem}_sorted.csv"
    fieldnames = ["tier", "last_name", "first_name", "score", "total", "file",
                  *[f"p{i+1}" for i in range(5)]]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tier in TIERS:
            writer.writerows(grouped[tier])

    print(f"  Sorted results saved to: {out_path}\n")


if __name__ == "__main__":
    main()