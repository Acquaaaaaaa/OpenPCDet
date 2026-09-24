"""Compare two Booth A4 profile runs for exact integer repeatability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


INTEGER_COLUMNS = {
    "N_total",
    "N_A4",
    "N_exception",
    "N_zero_fp32_exact",
    "N_zero_quantized",
    "N_zero_from_rounding",
    "N_nonzero_quantized",
    "N_A4_nonzero_quantized",
    "N_clipped_low",
    "N_clipped_high",
    "N_q_minus128",
    "q_min",
    "q_max",
    "call_count",
}


def _csv_integer_projection(
    path: Path,
    key_columns: tuple[str, ...],
    integer_columns=INTEGER_COLUMNS,
) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        tuple(row[column] for column in key_columns): {
            column: int(row[column])
            for column in integer_columns
            if column in row and row[column] != ""
        }
        for row in rows
    }


def _compare_npz(left: Path, right: Path) -> bool:
    with np.load(left) as left_data, np.load(right) as right_data:
        if set(left_data.files) != set(right_data.files):
            return False
        return all(np.array_equal(left_data[key], right_data[key]) for key in left_data.files)


def compare_runs(left: Path, right: Path) -> dict:
    summary_columns = INTEGER_COLUMNS - {"N_nonzero_quantized", "N_A4_nonzero_quantized"}
    summary_equal = _csv_integer_projection(
        left / "activation_summary_long.csv", ("activation_id", "scale_method"), summary_columns
    ) == _csv_integer_projection(
        right / "activation_summary_long.csv", ("activation_id", "scale_method"), summary_columns
    )
    frame_equal = _csv_integer_projection(
        left / "per_frame_counts.csv", ("frame_id", "activation_id", "scale_method")
    ) == _csv_integer_projection(
        right / "per_frame_counts.csv", ("frame_id", "activation_id", "scale_method")
    )
    int8_hist_equal = _compare_npz(
        left / "int8_histograms.npz", right / "int8_histograms.npz"
    )
    digit_hist_equal = _compare_npz(
        left / "booth_digit_histograms.npz", right / "booth_digit_histograms.npz"
    )
    result = {
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "summary_integer_counts_equal": summary_equal,
        "per_frame_integer_counts_equal": frame_equal,
        "int8_histograms_equal": int8_hist_equal,
        "booth_digit_histograms_equal": digit_hist_equal,
    }
    result["all_integer_results_exactly_equal"] = all(
        (summary_equal, frame_equal, int8_hist_equal, digit_hist_equal)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result))
    if not result["all_integer_results_exactly_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
