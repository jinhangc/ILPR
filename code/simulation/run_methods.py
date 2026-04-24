from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_methods import parse_csv_numbers, run_trials

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--known-T", dest="known_T", default="")
    parser.add_argument("--unknown-T", dest="unknown_T", default="10000,20000,40000,60000,80000")
    parser.add_argument("--betas", default="2.0")
    parser.add_argument(
        "--output-csv",
        default=str(WORKSPACE_ROOT / "outputs/simulation_comparison/comparison_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    known_T = [int(x) for x in parse_csv_numbers(args.known_T, int)]
    unknown_T = [int(x) for x in parse_csv_numbers(args.unknown_T, int)]
    betas = [float(x) for x in parse_csv_numbers(args.betas, float)]

    rows = run_trials(known_T, betas, args.trials, known_utility=True) + run_trials(
        unknown_T, betas, args.trials, known_utility=False
    )

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = WORKSPACE_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "method", "T", "beta", "avg_regret", "std_regret", "n_trials"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] csv -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
