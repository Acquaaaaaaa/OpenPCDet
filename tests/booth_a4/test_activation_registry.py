import torch.nn as nn

from tools.booth_a4.activation_registry import build_activation_registry


class DummyVFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([nn.Module()])
        self.pfn_layers[0].add_module("linear", nn.Linear(10, 64, bias=False))


def _stage(in_channels, out_channels, extra_convs):
    layers = [
        nn.ZeroPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]
    for _ in range(extra_convs):
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )
    return nn.Sequential(*layers)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _stage(64, 64, 3),
                _stage(64, 128, 5),
                _stage(128, 256, 5),
            ]
        )
        self.deblocks = nn.ModuleList(
            [
                nn.Sequential(nn.ConvTranspose2d(64, 128, 1), nn.ReLU()),
                nn.Sequential(nn.ConvTranspose2d(128, 128, 2, stride=2), nn.ReLU()),
                nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=4), nn.ReLU()),
            ]
        )


class DummyDenseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_cls = nn.Conv2d(384, 18, 1)
        self.conv_box = nn.Conv2d(384, 42, 1)
        self.conv_dir_cls = nn.Conv2d(384, 12, 1)


class DummyPointPillar(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = DummyVFE()
        self.backbone_2d = DummyBackbone()
        self.dense_head = DummyDenseHead()


def test_pointpillar_registry_has_expected_consumer_and_unique_edge_counts():
    registry = build_activation_registry(DummyPointPillar())
    assert len(registry.consumers) == 23
    assert len(registry.edges) == 19


def test_shared_edges_and_padding_capture_are_explicit():
    registry = build_activation_registry(DummyPointPillar())
    edges = {edge.activation_id: edge for edge in registry.edges}

    stage_zero = edges["act__backbone_2d_stage_0_output"]
    assert stage_zero.consumer_layers == (
        "backbone_2d.blocks.1.1",
        "backbone_2d.deblocks.0.0",
    )
    assert stage_zero.capture_module == "backbone_2d.blocks.1.0"
    assert stage_zero.capture_strategy == "pre_zero_pad_input"
    assert stage_zero.exclude_boundary_padding is True

    dense = edges["act__dense_head_spatial_features_2d"]
    assert dense.shared_consumer_count == 3
    assert dense.capture_module == "dense_head.conv_cls"


def test_all_first_stage_convs_capture_before_explicit_zero_padding():
    registry = build_activation_registry(DummyPointPillar())
    first_stage_consumers = {
        "backbone_2d.blocks.0.1",
        "backbone_2d.blocks.1.1",
        "backbone_2d.blocks.2.1",
    }
    records = {
        record.consumer_layer: record
        for record in registry.consumers
        if record.consumer_layer in first_stage_consumers
    }
    assert set(records) == first_stage_consumers
    assert all(record.hook_strategy == "pre_zero_pad_input" for record in records.values())
