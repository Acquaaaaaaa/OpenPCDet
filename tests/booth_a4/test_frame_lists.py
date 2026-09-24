from pathlib import Path

import pytest

from tools.booth_a4.build_frame_lists import build_frame_lists, select_evenly


def test_even_selection_is_deterministic_unique_and_endpoint_inclusive():
    values = [f"{index:06d}" for index in range(3712)]
    selected = select_evenly(values, 256)
    assert len(selected) == 256
    assert len(set(selected)) == 256
    assert selected[0] == values[0]
    assert selected[-1] == values[-1]
    assert selected == select_evenly(values, 256)


def test_build_frame_lists_and_refuse_changed_overwrite(tmp_path: Path):
    dataset = tmp_path / "kitti"
    image_sets = dataset / "ImageSets"
    image_sets.mkdir(parents=True)
    train_ids = [f"{index:06d}" for index in range(3712)]
    val_ids = [f"{index:06d}" for index in range(3769)]
    (image_sets / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (image_sets / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    output = tmp_path / "manifests"
    metadata = build_frame_lists(dataset, output)
    assert metadata["calibration"]["selected_frame_count"] == 256
    assert metadata["pilot"]["selected_frame_count"] == 256
    assert metadata["full_validation"]["selected_frame_count"] == 3769
    assert len((output / "kitti_val_full_3769.txt").read_text().splitlines()) == 3769

    build_frame_lists(dataset, output)
    (output / "kitti_train_calib_256.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_frame_lists(dataset, output)
