from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_real_methods import plot_comparison

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=str(WORKSPACE_ROOT / "outputs/semi_real_comparison/comparison_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs/semi_real_comparison"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = WORKSPACE_ROOT / input_csv
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Run run_real_methods.py first, or pass --input-csv explicitly."
        )
    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = WORKSPACE_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(rows, output_dir)
    print(f"[saved] plot -> {output_dir / 'real_method_comparison.svg'}", flush=True)


if __name__ == "__main__":
    main()
