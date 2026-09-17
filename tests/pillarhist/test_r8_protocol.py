import gzip
import json
import math
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
PILLARHIST_TOOLS = REPO / "tools" / "pillarhist"
if str(PILLARHIST_TOOLS) not in sys.path:
    sys.path.insert(0, str(PILLARHIST_TOOLS))

from r8_protocol import (
    HASH_CHAIN_INITIAL,
    audit_batch_indices,
    make_replay_entries,
    recomputed_paper_values,
    scientific_label,
    update_hash_chain,
    write_replay,
)


def test_r8_replay_is_deterministic_complete_and_epoch_local():
    first = list(make_replay_entries(dataset_size=10, batch_size=4, epochs=3, seed=666))
    second = list(make_replay_entries(dataset_size=10, batch_size=4, epochs=3, seed=666))
    assert first == second
    assert len(first) == 9
    for epoch in range(1, 4):
        entries = [item for item in first if item["epoch_one_based"] == epoch]
        indices = [index for item in entries for index in item["dataset_indices"]]
        assert sorted(indices) == list(range(10))
        assert len([seed for item in entries for seed in item["sample_seeds"]]) == 10


def test_r8_replay_file_and_audit_samples(tmp_path):
    path = tmp_path / "replay.jsonl.gz"
    manifest = write_replay(path, dataset_size=10, batch_size=4, epochs=2, seed=667)
    assert manifest["steps_per_epoch"] == 3
    assert manifest["total_optimizer_steps"] == 6
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        entries = [json.loads(line) for line in stream]
    assert len(entries) == 6
    assert audit_batch_indices(3, 667, 1) == [0, 1, 2]


def test_r8_hash_chain_uses_every_step_in_order():
    root = HASH_CHAIN_INITIAL
    for step, checksum in enumerate(("a" * 64, "b" * 64, "c" * 64)):
        root = update_hash_chain(root, step, checksum)
    reordered = HASH_CHAIN_INITIAL
    for step, checksum in enumerate(("b" * 64, "a" * 64, "c" * 64)):
        reordered = update_hash_chain(reordered, step, checksum)
    assert root != reordered


@pytest.mark.parametrize(
    "values, expected",
    [
        ((0.5, 0.1, 2, 3), "DIRECTIONALLY_SUPPORTED"),
        ((-0.5, -0.1, 1, 0), "NOT_SUPPORTED"),
        ((0.5, -0.1, 2, 1), "MIXED"),
        ((0.0, 0.0, 0, 0), "NOT_SUPPORTED"),
    ],
)
def test_r8_scientific_label(values, expected):
    assert scientific_label(*values) == expected


def test_r8_scientific_label_rejects_nonfinite():
    with pytest.raises(ValueError):
        scientific_label(math.nan, 1.0, 0, 3)


def test_r8_paper_reported_and_recomputed_values_are_both_kept():
    values = recomputed_paper_values()
    car = values["PH-PointPillars"]["Car"]
    pedestrian = values["PH-PointPillars"]["Pedestrian"]
    assert car["paper_reported_map"] == 81.42
    assert car["recomputed_from_displayed_ap"] == pytest.approx(81.41)
    assert pedestrian["paper_reported_map"] == 51.07
    assert pedestrian["recomputed_from_displayed_ap"] == pytest.approx(51.0766666667)
