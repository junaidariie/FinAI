import argparse
import csv
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_utilization": "Context utilization",
}
DEFAULT_CSV = Path(__file__).resolve().parent / "results.csv"


def plot_metrics(csv_path: Path, output_path: Path) -> Path:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        raise ValueError(f"No evaluation results found in {csv_path}.")

    averages: Dict[str, float] = {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in METRIC_LABELS
    }
    labels = [METRIC_LABELS[name] for name in averages]
    values = list(averages.values())

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color="#4472C4")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"Average RAG Evaluation Metrics ({len(rows)} question(s))")
    axis.tick_params(axis="x", labelrotation=25)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average RAG evaluation metrics.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "metrics.png",
    )
    args = parser.parse_args()
    print(f"Saved metric plot to {plot_metrics(args.csv, args.output)}")


if __name__ == "__main__":
    main()
