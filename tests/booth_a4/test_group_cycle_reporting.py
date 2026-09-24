import csv
import json

import torch

from tools.booth_a4.group_counter import GroupCounter
from tools.booth_a4.merge_group_cycle_shards import _merge_aggregate_records
from tools.booth_a4.operand_mapper import OperandGroupBatch
from tools.booth_a4.report_group_cycle import (
    build_network_rows,
    build_report_rows,
    generate_reports,
    load_aggregate_records,
    write_aggregate_records,
)


def _record(
    layer_order, consumer_layer, scale_method="minmax", *, retain_sensitivity=True
):
    digits = torch.zeros((2, 8, 4), dtype=torch.int8)
    digits[0, 0, 0] = 1
    digits[0, 0, 2] = 1
    digits[1, 1, 1] = -1
    logical = torch.ones((2, 8), dtype=torch.bool)
    zeros = torch.zeros_like(logical)
    counter = GroupCounter(
        8, window_sizes=(8, 32, 128), retain_sensitivity=retain_sensitivity
    )
    counter.update(OperandGroupBatch(digits, logical, zeros, zeros, zeros))
    counts = counter.finalize()
    return {
        "layer_order": layer_order,
        "consumer_layer": consumer_layer,
        "activation_id": f"act_{consumer_layer}",
        "module_type": "Conv2d",
        "weight_shape": json.dumps([16, 8, 1, 1]),
        "scale_method": scale_method,
        "mapping_mode": "consumer_aware_reference_cim",
        "padding_policy": "physical_rows_included",
        "pfn_token_policy": "fixed_slots_included",
        "N_main": 8,
        "C_out": 16,
        "counts": counts.to_dict(),
    }


def test_report_round_trip_and_network_uses_ratio_of_sums(tmp_path):
    records = [_record(0, "layer0"), _record(1, "layer1")]
    aggregate = tmp_path / "aggregate.json"
    write_aggregate_records(aggregate, records)
    restored = load_aggregate_records(aggregate)
    group_rows, cycle_rows, sensitivity_rows = build_report_rows(
        restored,
        main_output_parallelism=8,
        correction_output_parallelism=8,
    )
    assert len(group_rows) == 2
    assert len(cycle_rows) == 2
    assert sensitivity_rows
    assert group_rows[0]["P_active_d2"] == 0.5
    network = build_network_rows(cycle_rows)
    assert len(network) == 1
    assert network[0]["C_A_tile_network"] == sum(row["C_A_tile"] for row in cycle_rows)
    assert network[0]["speedup_C_vector_vs_B_tile_network"] == (
        sum(row["C_B_tile"] for row in cycle_rows)
        / sum(row["C_C_vector_tile"] for row in cycle_rows)
    )


def test_generate_reports_writes_all_required_tables(tmp_path):
    summary = generate_reports([_record(0, "layer0")], tmp_path)
    assert summary["group_rows"] == 1
    for name in (
        "consumer_group_utilization.csv",
        "consumer_cycle_summary.csv",
        "network_cycle_summary.csv",
        "correction_sensitivity.csv",
    ):
        path = tmp_path / name
        assert path.exists()
        with path.open(encoding="utf-8", newline="") as stream:
            assert list(csv.DictReader(stream))


def test_generate_primary_only_reports_skips_sensitivity_table(tmp_path):
    summary = generate_reports(
        [_record(0, "layer0")], tmp_path, include_sensitivity=False
    )
    assert summary["group_rows"] == 1
    assert summary["cycle_rows"] == 1
    assert summary["sensitivity_rows"] == 0
    assert not (tmp_path / "correction_sensitivity.csv").exists()


def test_shard_aggregate_merge_preserves_frame_ceiling_boundaries(tmp_path):
    record = _record(0, "layer0")
    shard_dirs = []
    for index in range(2):
        shard = tmp_path / f"shard_{index}"
        shard.mkdir()
        write_aggregate_records(shard / "aggregate_group_counts.json", [record])
        shard_dirs.append(shard)
    merged = _merge_aggregate_records(shard_dirs)
    counts = merged[0]["counts"]
    assert counts["n_group_total"] == 4
    # Each source record represents one frame; they must remain two schedule units.
    assert counts["schedule_histograms"]["layer_frame"]["mixed"] == {"1": 2}


def test_shard_aggregate_merge_supports_core_mode(tmp_path):
    record = _record(0, "layer0", retain_sensitivity=False)
    shard_dirs = []
    for index in range(2):
        shard = tmp_path / f"core_shard_{index}"
        shard.mkdir()
        write_aggregate_records(shard / "aggregate_group_counts.json", [record])
        shard_dirs.append(shard)
    merged = _merge_aggregate_records(shard_dirs)
    counts = merged[0]["counts"]
    assert counts["n_group_total"] == 4
    assert counts["sensitivity_retained"] is False
    assert counts["schedule_histograms"]["layer_frame"]["mixed"] == {"1": 2}
