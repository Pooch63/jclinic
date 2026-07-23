import logging
import sys
import pandas as pd


def main(csv_path: str = "a.csv"):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read '{csv_path}': {e}")
        return

    for i, row in df.iterrows():
        sev = row.get("Visit1_Severity", None)
        typ = row.get("Visit1_Type", None)
        logging.info(f"Row {i}: Visit1_Severity={sev} | Visit1_Type={typ}")


if __name__ == "__main__":
    main()
