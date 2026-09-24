"""Generate readable layer-wide tables and a concise Booth A4 study report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil


METHODS = ("minmax", "p99_99", "p99_9")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row, key):
    return float(row[key]) if row.get(key) not in (None, "") else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.md", "activation_summary_long.csv", "layer_summary_wide.csv", "consumer_activation_map.csv"):
        if (args.output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite existing report artifact: {filename}")
    summary = _read_csv(args.profile_dir / "activation_summary_long.csv")
    registry = json.loads((args.profile_dir / "activation_registry.json").read_text(encoding="utf-8"))
    by_key = {(row["activation_id"], row["scale_method"]): row for row in summary}

    wide_rows = []
    for consumer in registry["consumers"]:
        activation_id = consumer["activation_id"]
        row = {
            "layer_order": consumer["layer_order"],
            "consumer_layer": consumer["consumer_layer"],
            "module_type": consumer["module_type"],
            "weight_shape": json.dumps(consumer["weight_shape"]),
            "activation_id": activation_id,
            "shared_activation": consumer["shared_activation"],
        }
        for method in METHODS:
            source = by_key.get((activation_id, method))
            if source:
                row[f"RA4_{method}"] = source["RA4_tensor_micro"]
                row[f"Rexc_{method}"] = source["Rexc_tensor_micro"]
                row[f"Rclip_{method}"] = source["Rclip"]
        minmax = by_key.get((activation_id, "minmax"))
        if minmax:
            row["Rzero_quantized_minmax"] = minmax["Rzero_quantized"]
            row["RA4_nonzero_q_minmax"] = minmax["RA4_nonzero_q"]
        wide_rows.append(row)
    with (args.output_dir / "layer_summary_wide.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=wide_rows[0].keys())
        writer.writeheader()
        writer.writerows(wide_rows)
    shutil.copy2(args.profile_dir / "activation_summary_long.csv", args.output_dir / "activation_summary_long.csv")
    shutil.copy2(args.profile_dir / "layer_consumers.csv", args.output_dir / "consumer_activation_map.csv")

    minmax_rows = sorted(
        (row for row in summary if row["scale_method"] == "minmax"),
        key=lambda row: int(row["activation_order"]),
    )
    sensitivity = sorted(
        minmax_rows,
        key=lambda row: _float(row, "threshold") / _float(by_key[(row["activation_id"], "p99_99")], "threshold"),
        reverse=True,
    )[:5]
    nonzero_best = sorted(minmax_rows, key=lambda row: _float(row, "RA4_nonzero_q"), reverse=True)[:5]
    zero_dominated = sorted(minmax_rows, key=lambda row: _float(row, "Rzero_quantized"), reverse=True)[:5]
    clipping_rows = sorted(
        (row for row in summary if row["scale_method"] != "minmax"),
        key=lambda row: _float(row, "Rclip"),
        reverse=True,
    )[:5]
    macro_differences = sorted(
        minmax_rows,
        key=lambda row: abs(_float(row, "RA4_tensor_micro") - _float(row, "macro_mean")),
        reverse=True,
    )[:5]
    shared_edges = [edge for edge in registry["activation_edges"] if edge["shared_consumer_count"] > 1]

    lines = [
        "# PointPillars FP32-reference INT8 Booth A4 统计报告",
        "",
        "> 结论边界：结果来自 FP32 reference activation 的冻结 scale 独立离线量化；不代表传播式 A8W8 网络的真实运行分布。",
        "",
        "## Min-max 主结果（unique logical activation edge）",
        "",
        "| activation_id | RA4 | Rexc | Rzero(q) | RA4(nonzero q) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in minmax_rows:
        lines.append(
            f"| `{row['activation_id']}` | {_float(row, 'RA4_tensor_micro'):.6f} | "
            f"{_float(row, 'Rexc_tensor_micro'):.6f} | {_float(row, 'Rzero_quantized'):.6f} | "
            f"{_float(row, 'RA4_nonzero_q'):.6f} |"
        )
    lines.extend(["", "## Scale 敏感性", ""])
    for row in sensitivity:
        p99 = by_key[(row["activation_id"], "p99_99")]
        ratio = _float(row, "threshold") / _float(p99, "threshold")
        lines.append(
            f"- `{row['activation_id']}`：min-max/P99.99 threshold={ratio:.2f}，"
            f"RA4 { _float(row, 'RA4_tensor_micro'):.6f} → {_float(p99, 'RA4_tensor_micro'):.6f}，"
            f"P99.99 clipping={_float(p99, 'Rclip'):.6g}。"
        )
    lines.extend(["", "## 排除量化零后的候选层", ""])
    for row in nonzero_best:
        lines.append(
            f"- `{row['activation_id']}`：RA4_nonzero_q={_float(row, 'RA4_nonzero_q'):.6f}，"
            f"RA4={_float(row, 'RA4_tensor_micro'):.6f}。"
        )
    lines.extend(["", "## 零值来源诊断（min-max）", ""])
    for row in zero_dominated:
        lines.append(
            f"- `{row['activation_id']}`：Rzero(q)={_float(row, 'Rzero_quantized'):.6f}，"
            f"Rzero(FP32 exact)={_float(row, 'Rzero_fp32'):.6f}，"
            f"Rzero(rounding)={_float(row, 'Rzero_from_rounding'):.6f}。"
        )
    lines.extend(["", "## Clipping 较高的 percentile 结果", ""])
    for row in clipping_rows:
        lines.append(
            f"- `{row['activation_id']}` / `{row['scale_method']}`："
            f"Rclip={_float(row, 'Rclip'):.6g}，RA4={_float(row, 'RA4_tensor_micro'):.6f}。"
        )
    lines.extend(["", "## Micro 与 per-frame macro", ""])
    for row in macro_differences:
        difference = _float(row, "macro_mean") - _float(row, "RA4_tensor_micro")
        lines.append(
            f"- `{row['activation_id']}`：micro={_float(row, 'RA4_tensor_micro'):.6f}，"
            f"macro_mean={_float(row, 'macro_mean'):.6f}，差值={difference:+.6g}，"
            f"P5/P50/P95={_float(row, 'macro_p5'):.6f}/"
            f"{_float(row, 'macro_p50'):.6f}/{_float(row, 'macro_p95'):.6f}。"
        )
    lines.extend(["", "## 共享 activation", ""])
    for edge in shared_edges:
        lines.append(
            f"- `{edge['activation_id']}` → {', '.join(f'`{name}`' for name in edge['consumer_layers'])}；"
            "tensor-level 计数只进行一次。"
        )
    lines.extend([
        "",
        "## 解释限制",
        "",
        "- 主指标是 logical tensor element 的 micro-average，卷积边界 padding 不计入。",
        "- 结构零、FP32 精确零和舍入零会提高 RA4，因此必须同时查看 `Rzero_quantized` 与 `RA4_nonzero_q`。",
        "- Percentile scale 可能提高 clipping；不能仅凭 RA4 选择量化方案。",
        "- 当前未实现精确 MAC-weighted RA4，也未验证 QDQ 传播后的 KITTI AP。",
        "- 多个 consumer 共享同一 activation 时不得把相同 tensor 结果重复相加。",
        "- 报告不生成含义模糊的全网络 RA4；表格主键是 unique logical activation edge。",
        "",
    ])
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"activation_rows": len(summary), "consumer_rows": len(wide_rows)}))


if __name__ == "__main__":
    main()
