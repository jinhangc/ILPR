from __future__ import annotations

import argparse
import csv
from pathlib import Path

from compare_real_methods import DEFAULT_DATASET, parse_csv_numbers, run_trials

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--T", default="100,150,200,250,300,350,400,450,500,550,600,650,700")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--max-skus",
        type=int,
        default=None,
        help="Maximum number of usable products to include; default uses all usable products.",
    )
    parser.add_argument(
        "--max-purchase-rate",
        type=float,
        default=0.95,
        help="Drop products whose binary purchase rate exceeds this threshold.",
    )
    parser.add_argument(
        "--max-upper-bound-oracle-share",
        type=float,
        default=1.0,
        help="Drop products whose calibrated oracle sits at the upper price bound more often than this share.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes. Use 1 for sequential execution.")
    parser.add_argument(
        "--output-csv",
        default=str(WORKSPACE_ROOT / "outputs/semi_real_comparison/comparison_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(x) for x in parse_csv_numbers(args.T, int)]
    rows = run_trials(
        horizons,
        args.trials,
        dataset_path=args.dataset,
        max_skus=args.max_skus,
        workers=args.workers,
        max_purchase_rate=args.max_purchase_rate,
        max_upper_bound_oracle_share=args.max_upper_bound_oracle_share,
    )

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = WORKSPACE_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario",
                "sku",
                "method",
                "T",
                "avg_regret",
                "std_regret",
                "avg_regret_per_period",
                "avg_oracle_revenue",
                "avg_realized_revenue",
                "relative_regret",
                "n_trials",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] csv -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
