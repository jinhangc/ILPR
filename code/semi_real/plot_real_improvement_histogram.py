from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default=str(WORKSPACE_ROOT / "outputs/semi_real_comparison/comparison_summary.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "outputs/semi_real_comparison/real_method_improvement_histogram.svg"),
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional path for summary statistics CSV. Defaults to <output stem>_summary.csv beside the plot.",
    )
    parser.add_argument("--T", type=int, default=700)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--clip-quantiles",
        default="0.02,0.98",
        help="Lower and upper quantiles used to set the displayed x-axis range; outside values are folded into edge bins.",
    )
    return parser.parse_args()


def load_improvements(path: Path, horizon: int) -> dict[str, list[float]]:
    by_sku: dict[str, dict[str, dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["sku"] == "__aggregate__":
                continue
            if int(row["T"]) != horizon:
                continue
            by_sku.setdefault(row["sku"], {})[row["method"]] = row

    improvements = {"kernel_baseline": [], "dip": []}
    for sku, methods in by_sku.items():
        if "mymethod" not in methods:
            continue
        my_regret = float(methods["mymethod"]["avg_regret"])
        for baseline in ["kernel_baseline", "dip"]:
            if baseline not in methods:
                continue
            base_regret = float(methods[baseline]["avg_regret"])
            improvements[baseline].append((base_regret - my_regret) / max(base_regret, 1e-9))
    if not any(improvements.values()):
        raise ValueError(f"No product-level rows found for T={horizon} in {path}")
    return improvements


def make_histogram_svg(
    series: dict[str, list[float]],
    horizon: int,
    bins: int,
    q_low: float,
    q_high: float,
) -> str:
    width, height = 980, 620
    left, right, top, bottom = 90, 40, 140, 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    labels = {
        "kernel_baseline": "vs Kernel-based policy",
        "dip": "vs DIP",
    }
    colors = {
        "kernel_baseline": "#4C78A8",
        "dip": "#E45756",
    }
    active_methods = [method for method in ["kernel_baseline", "dip"] if series.get(method)]
    all_values = [v for method in active_methods for v in series[method]]
    sorted_vals = sorted(all_values)
    n = len(sorted_vals)

    def quantile(q: float) -> float:
        idx = min(max(int(round(q * (n - 1))), 0), n - 1)
        return sorted_vals[idx]

    raw_min = min(all_values)
    raw_max = max(all_values)
    vmin = quantile(q_low)
    vmax = quantile(q_high)
    if abs(vmax - vmin) < 1e-12:
        vmin = raw_min
        vmax = raw_max if raw_max > raw_min else raw_min + 1.0
    edges = [vmin + (vmax - vmin) * i / bins for i in range(bins + 1)]
    counts_by_method: dict[str, list[int]] = {}
    for method in active_methods:
        counts = [0 for _ in range(bins)]
        for v in series[method]:
            v_clip = min(max(v, vmin), vmax)
            idx = min(int((v_clip - vmin) / (vmax - vmin + 1e-12) * bins), bins - 1)
            counts[idx] += 1
        counts_by_method[method] = counts

    ymax = max(max(counts) for counts in counts_by_method.values()) if counts_by_method else 1

    def xmap(x: float) -> float:
        return left + (x - vmin) / (vmax - vmin + 1e-12) * plot_w

    def ymap(y: float) -> float:
        return top + (1.0 - y / max(ymax, 1)) * plot_h

    title = f"Histogram of Relative Regret Improvement Across Products (T={horizon})"
    xlabel = "(Baseline regret - ILPR regret) / Baseline regret"
    tick_fmt = lambda x: f"{100*x:.0f}%"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-size="28" font-family="Helvetica">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" stroke-width="1.5" />',
        f'<text x="{left + plot_w/2:.1f}" y="{height - 22}" text-anchor="middle" font-size="20" font-family="Helvetica">{xlabel}</text>',
        f'<text x="28" y="{top + plot_h/2:.1f}" text-anchor="middle" font-size="20" font-family="Helvetica" transform="rotate(-90, 28, {top + plot_h/2:.1f})">Number of products</text>',
    ]

    legend_x = width - 330
    legend_y = 86
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="260" height="58" fill="white" stroke="#444" />')
    for idx, method in enumerate(active_methods):
        y = legend_y + 18 + idx * 22
        svg.append(
            f'<rect x="{legend_x + 14}" y="{y - 9}" width="26" height="14" fill="{colors[method]}" fill-opacity="0.45" stroke="{colors[method]}" stroke-width="2" />'
        )
        svg.append(f'<text x="{legend_x + 50}" y="{y + 2}" font-size="16" font-family="Helvetica">{labels[method]}</text>')

    if vmin < 0 < vmax:
        x0 = xmap(0.0)
        svg.append(f'<line x1="{x0:.2f}" y1="{top}" x2="{x0:.2f}" y2="{top + plot_h}" stroke="#888" stroke-width="2" stroke-dasharray="7 5" />')

    for method in active_methods:
        for i, count in enumerate(counts_by_method[method]):
            x0 = xmap(edges[i])
            x1 = xmap(edges[i + 1])
            y0 = ymap(count)
            h = top + plot_h - y0
            w = max(x1 - x0 - 1.5, 1.0)
            svg.append(
                f'<rect x="{x0 + 0.75:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{colors[method]}" fill-opacity="0.38" stroke="{colors[method]}" stroke-width="1.5" />'
            )

    for yt in range(0, ymax + 1, max(1, math.ceil(ymax / 5))):
        y = ymap(float(yt))
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eee" stroke-width="1" />')
        svg.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="15" font-family="Helvetica">{yt}</text>')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xval = vmin + frac * (vmax - vmin)
        x = xmap(xval)
        svg.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 7}" stroke="#333" stroke-width="1.5" />')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="15" font-family="Helvetica">{tick_fmt(xval)}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def write_summary_csv(
    series: dict[str, list[float]],
    horizon: int,
    output: Path,
) -> None:
    labels = {
        "kernel_baseline": "Kernel-based policy",
        "dip": "DIP",
    }
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["T", "baseline", "n_products", "mean_relative_improvement", "median_relative_improvement"],
        )
        writer.writeheader()
        for method in ["kernel_baseline", "dip"]:
            vals = series.get(method, [])
            if not vals:
                continue
            vals_sorted = sorted(float(v) for v in vals)
            writer.writerow(
                {
                    "T": horizon,
                    "baseline": labels[method],
                    "n_products": len(vals_sorted),
                    "mean_relative_improvement": float(sum(vals_sorted) / len(vals_sorted)),
                    "median_relative_improvement": float(vals_sorted[len(vals_sorted) // 2]),
                }
            )


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

    output = Path(args.output)
    if not output.is_absolute():
        output = WORKSPACE_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output = Path(args.summary_output) if args.summary_output else output.with_name(f"{output.stem}_summary.csv")
    if not summary_output.is_absolute():
        summary_output = WORKSPACE_ROOT / summary_output
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    improvements = load_improvements(input_csv, args.T)
    q_parts = [float(part.strip()) for part in args.clip_quantiles.split(",") if part.strip()]
    if len(q_parts) != 2:
        raise ValueError("--clip-quantiles must contain two comma-separated floats")
    q_low, q_high = q_parts
    svg = make_histogram_svg(improvements, args.T, args.bins, q_low, q_high)
    output.write_text(svg, encoding="utf-8")
    write_summary_csv(improvements, args.T, summary_output)
    print(f"[saved] histogram -> {output}", flush=True)
    print(f"[saved] summary -> {summary_output}", flush=True)


if __name__ == "__main__":
    main()
