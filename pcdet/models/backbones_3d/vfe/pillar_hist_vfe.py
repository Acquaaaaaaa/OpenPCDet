import math

import torch
import torch.nn as nn

from .vfe_template import VFETemplate


def _integer_tensor(values, name):
    """Validate integer-valued input before converting it to int64."""
    if values.is_floating_point():
        if not torch.isfinite(values).all():
            raise ValueError(f'{name} must contain finite integer-valued entries')
        rounded = values.round().long()
        if not torch.equal(rounded.to(values.dtype), values):
            raise ValueError(f'{name} must contain integer-valued entries')
        return rounded
    return values.long()


def pillar_histogram_cpu_oracle(
    points,
    voxel_coords,
    batch_size,
    num_bins,
    voxel_size,
    point_cloud_range,
    grid_size,
):
    """Slow CPU truth implementation used by PillarHist correctness tests."""
    if points.device.type != 'cpu' or voxel_coords.device.type != 'cpu':
        raise ValueError('CPU oracle inputs must be CPU tensors')
    if points.ndim != 2 or points.shape[1] < 5:
        raise ValueError('points must have shape [P, >=5]')
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 4:
        raise ValueError('voxel_coords must have shape [V, 4]')

    batch_size = int(batch_size)
    nx, ny, nz = (int(x) for x in grid_size)
    if nz != 1:
        raise ValueError('PillarHist KITTI reference requires grid_size[2] == 1')

    coords = _integer_tensor(voxel_coords, 'voxel_coords')
    point_batch = _integer_tensor(points[:, 0], 'points[:, 0]')
    if point_batch.numel() and ((point_batch < 0) | (point_batch >= batch_size)).any():
        raise ValueError('point batch ids are outside [0, batch_size)')
    if coords.numel():
        invalid = (
            (coords[:, 0] < 0) | (coords[:, 0] >= batch_size)
            | (coords[:, 1] != 0)
            | (coords[:, 2] < 0) | (coords[:, 2] >= ny)
            | (coords[:, 3] < 0) | (coords[:, 3] >= nx)
        )
        if invalid.any():
            raise ValueError('voxel_coords contain out-of-range batch/z/y/x indices')

    active_key = coords[:, 0] * (ny * nx) + coords[:, 2] * nx + coords[:, 3]
    if torch.unique(active_key).numel() != active_key.numel():
        raise ValueError('voxel_coords contain duplicate active pillars')

    num_voxels = coords.shape[0]
    count_hist = torch.zeros((num_voxels, num_bins), dtype=torch.int32)
    intensity_sum = torch.zeros((num_voxels, num_bins), dtype=torch.float32)
    if num_voxels == 0:
        return count_hist, intensity_sum

    active_rows = {int(key): row for row, key in enumerate(active_key.tolist())}
    geometry_dtype = points.dtype if points.is_floating_point() else torch.float32
    geometry = torch.tensor(
        list(point_cloud_range) + list(voxel_size[:2]), dtype=geometry_dtype
    ).tolist()
    x_min, y_min, z_min, x_max, y_max, z_max, voxel_x, voxel_y = map(
        float, geometry
    )
    bin_height = float(
        torch.tensor((z_max - z_min) / num_bins, dtype=geometry_dtype)
    )

    for point_idx in range(points.shape[0]):
        point = points[point_idx]
        x, y, z = (float(point[i]) for i in (1, 2, 3))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise ValueError('point coordinates must be finite')
        if not (x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max):
            continue
        cx = math.floor((x - x_min) / voxel_x)
        cy = math.floor((y - y_min) / voxel_y)
        bin_id = math.floor((z - z_min) / bin_height)
        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= bin_id < num_bins):
            raise RuntimeError('valid ROI point required protective index clamping')
        key = int(point_batch[point_idx]) * (ny * nx) + cy * nx + cx
        row = active_rows.get(key)
        if row is None:
            continue
        count_hist[row, bin_id] += 1
        intensity_sum[row, bin_id] += point[4].float()

    intensity_mean = torch.zeros_like(intensity_sum)
    occupied = count_hist > 0
    intensity_mean[occupied] = (
        intensity_sum[occupied] / count_hist[occupied].to(torch.float32)
    )
    return count_hist, intensity_mean


class PillarHistVFE(VFETemplate):
    """Deterministic PillarHist reference VFE for the KITTI PointPillars path."""

    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        point_cloud_range,
        grid_size=None,
        **kwargs,
    ):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = int(num_point_features)
        if self.num_point_features < 4:
            raise ValueError('PillarHist requires x, y, z and intensity point features')

        self.num_bins = int(self.model_cfg.NUM_BINS)
        if self.num_bins <= 0:
            raise ValueError('NUM_BINS must be positive')
        self.num_filters = list(self.model_cfg.NUM_FILTERS)
        if len(self.num_filters) != 1 or int(self.num_filters[-1]) != 64:
            raise ValueError('current PointPillars backend requires NUM_FILTERS: [64]')

        self.count_mode = self.model_cfg.get('COUNT_MODE', 'raw')
        self.intensity_mode = self.model_cfg.get('INTENSITY_MODE', 'mean_raw')
        self.coord_mode = self.model_cfg.get('COORD_MODE', 'raw_meter_xy')
        self.histogram_dtype = self.model_cfg.get('HISTOGRAM_DTYPE', 'float32')
        self.reduction_mode = self.model_cfg.get(
            'REDUCTION_MODE', 'deterministic_segment'
        )
        self.use_all_points = bool(
            self.model_cfg.get('USE_ALL_POINTS_IN_ADMITTED_PILLARS', True)
        )
        self._validate_modes()

        if grid_size is None:
            raise ValueError('grid_size is required')
        self.nx, self.ny, self.nz = (int(x) for x in grid_size)
        if self.nz != 1:
            raise ValueError('PillarHist KITTI reference requires grid_size[2] == 1')

        self.voxel_x = float(voxel_size[0])
        self.voxel_y = float(voxel_size[1])
        self.voxel_z = float(voxel_size[2])
        point_cloud_range = [float(x) for x in point_cloud_range]
        if len(point_cloud_range) != 6:
            raise ValueError('point_cloud_range must have six entries')
        (
            self.x_min,
            self.y_min,
            self.z_min,
            self.x_max,
            self.y_max,
            self.z_max,
        ) = point_cloud_range
        if not (self.x_min < self.x_max and self.y_min < self.y_max and self.z_min < self.z_max):
            raise ValueError('point_cloud_range minima must be smaller than maxima')
        self.bin_height = (self.z_max - self.z_min) / self.num_bins

        expected_nx = int(round((self.x_max - self.x_min) / self.voxel_x))
        expected_ny = int(round((self.y_max - self.y_min) / self.voxel_y))
        if (self.nx, self.ny) != (expected_nx, expected_ny):
            raise ValueError('grid_size is inconsistent with point_cloud_range/voxel_size')

        projection_cfg = self.model_cfg.PROJECTION
        projection_type = projection_cfg.TYPE
        projection_bias = bool(projection_cfg.BIAS)
        input_dim = 2 * self.num_bins + 2
        output_dim = int(self.num_filters[-1])
        if projection_type == 'linear':
            self.projection = nn.Linear(input_dim, output_dim, bias=projection_bias)
        elif projection_type == 'linear_bn_relu':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim, bias=projection_bias),
                nn.BatchNorm1d(
                    output_dim,
                    eps=float(projection_cfg.get('BN_EPS', 1e-3)),
                    momentum=float(projection_cfg.get('BN_MOMENTUM', 0.01)),
                ),
                nn.ReLU(),
            )
        else:
            raise ValueError(f'unknown PROJECTION.TYPE: {projection_type}')
        self.projection_type = projection_type
        self.last_diagnostics = {}

    def _validate_modes(self):
        expected = {
            'COUNT_MODE': (self.count_mode, {'raw'}),
            'INTENSITY_MODE': (self.intensity_mode, {'mean_raw'}),
            'COORD_MODE': (self.coord_mode, {'raw_meter_xy', 'normalized_xy'}),
            'HISTOGRAM_DTYPE': (self.histogram_dtype, {'float32'}),
            'REDUCTION_MODE': (self.reduction_mode, {'deterministic_segment'}),
        }
        for name, (value, allowed) in expected.items():
            if value not in allowed:
                raise ValueError(f'{name} must be one of {sorted(allowed)}, got {value!r}')
        if not self.use_all_points:
            raise ValueError('the reference path requires all points in admitted pillars')

    def get_output_feature_dim(self):
        assert self.num_filters[-1] == 64
        return self.num_filters[-1]

    def _validate_inputs(self, points, voxel_coords, batch_size):
        if points.ndim != 2 or points.shape[1] < 5:
            raise ValueError('points must have shape [P, >=5]')
        if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 4:
            raise ValueError('voxel_coords must have shape [V, 4]')
        if points.device != voxel_coords.device:
            raise ValueError('points and voxel_coords must be on the same device')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if not torch.isfinite(points).all():
            raise ValueError('points must contain only finite values')

        coords = _integer_tensor(voxel_coords, 'voxel_coords')
        point_batch = _integer_tensor(points[:, 0], 'points[:, 0]')
        if point_batch.numel() and ((point_batch < 0) | (point_batch >= batch_size)).any():
            raise ValueError('point batch ids are outside [0, batch_size)')
        if coords.numel():
            invalid = (
                (coords[:, 0] < 0) | (coords[:, 0] >= batch_size)
                | (coords[:, 1] != 0)
                | (coords[:, 2] < 0) | (coords[:, 2] >= self.ny)
                | (coords[:, 3] < 0) | (coords[:, 3] >= self.nx)
            )
            if invalid.any():
                raise ValueError('voxel_coords contain out-of-range batch/z/y/x indices')

        active_key = coords[:, 0] * (self.ny * self.nx) + coords[:, 2] * self.nx + coords[:, 3]
        if torch.unique(active_key).numel() != active_key.numel():
            raise ValueError('voxel_coords contain duplicate active pillars')
        return coords, point_batch, active_key

    def build_histograms(self, points, voxel_coords, batch_size):
        """Build count and mean-intensity histograms in active-coordinate row order."""
        batch_size = int(batch_size)
        coords, point_batch, active_key = self._validate_inputs(
            points, voxel_coords, batch_size
        )
        num_voxels = coords.shape[0]
        device = points.device
        count_hist = torch.zeros(
            (num_voxels, self.num_bins), dtype=torch.int32, device=device
        )
        intensity_mean = torch.zeros(
            (num_voxels, self.num_bins), dtype=torch.float32, device=device
        )
        if num_voxels == 0:
            self.last_diagnostics = {
                'num_input_points': int(points.shape[0]),
                'num_roi_points': 0,
                'num_admitted_points': 0,
                'num_active_pillars': 0,
                'protective_clamp_count': 0,
                'overflow_pillars': 0,
                'overflow_points': 0,
            }
            return count_hist, intensity_mean

        with torch.autocast(device_type=device.type, enabled=False):
            points_fp32 = points.float()
            xyz = points_fp32[:, 1:4]
            in_roi = (
                (xyz[:, 0] >= self.x_min) & (xyz[:, 0] < self.x_max)
                & (xyz[:, 1] >= self.y_min) & (xyz[:, 1] < self.y_max)
                & (xyz[:, 2] >= self.z_min) & (xyz[:, 2] < self.z_max)
            )
            roi_points = points_fp32[in_roi]
            roi_batch = point_batch[in_roi]
            if roi_points.shape[0] == 0:
                self.last_diagnostics = {
                    'num_input_points': int(points.shape[0]),
                    'num_roi_points': 0,
                    'num_admitted_points': 0,
                    'num_active_pillars': int(num_voxels),
                    'protective_clamp_count': 0,
                    'overflow_pillars': 0,
                    'overflow_points': 0,
                }
                return count_hist, intensity_mean

            geometry = points_fp32.new_tensor([
                self.x_min, self.y_min, self.z_min,
                self.voxel_x, self.voxel_y, self.bin_height,
            ]).double()
            x_min, y_min, z_min, voxel_x, voxel_y, bin_height = geometry.unbind()
            cx = torch.floor(
                (roi_points[:, 1].double() - x_min) / voxel_x
            ).long()
            cy = torch.floor(
                (roi_points[:, 2].double() - y_min) / voxel_y
            ).long()
            bins_unclamped = torch.floor(
                (roi_points[:, 3].double() - z_min) / bin_height
            ).long()
            bins = bins_unclamped.clamp(0, self.num_bins - 1)
            clamp_count = int((bins != bins_unclamped).sum().item())
            if clamp_count:
                raise RuntimeError(
                    f'{clamp_count} valid ROI points required protective height-bin clamping'
                )
            if (
                ((cx < 0) | (cx >= self.nx)).any()
                or ((cy < 0) | (cy >= self.ny)).any()
                or ((bins < 0) | (bins >= self.num_bins)).any()
            ):
                raise RuntimeError('valid ROI points produced out-of-range indices')

            point_key = roi_batch * (self.ny * self.nx) + cy * self.nx + cx
            sorted_key, permutation = torch.sort(active_key, stable=True)
            positions = torch.searchsorted(sorted_key, point_key)
            safe_positions = positions.clamp(max=num_voxels - 1)
            admitted = (positions < num_voxels) & (
                sorted_key[safe_positions] == point_key
            )
            overflow_mask = ~admitted
            overflow_points = int(overflow_mask.sum().item())
            overflow_pillars = int(torch.unique(point_key[overflow_mask]).numel())
            rows = permutation[safe_positions[admitted]]
            admitted_bins = bins[admitted]
            admitted_intensity = roi_points[admitted, 4].float()
            segment_key = rows * self.num_bins + admitted_bins

            if segment_key.numel():
                sorted_segment, segment_permutation = torch.sort(
                    segment_key, stable=True
                )
                sorted_intensity = admitted_intensity[segment_permutation]
                unique_segment, segment_counts = torch.unique_consecutive(
                    sorted_segment, return_counts=True
                )
                segment_sums = torch.segment_reduce(
                    sorted_intensity, reduce='sum', lengths=segment_counts
                )
                flat_count = count_hist.view(-1)
                flat_sum = torch.zeros(
                    num_voxels * self.num_bins,
                    dtype=torch.float32,
                    device=device,
                )
                flat_count[unique_segment] = segment_counts.to(torch.int32)
                flat_sum[unique_segment] = segment_sums.float()
                occupied = flat_count > 0
                flat_mean = intensity_mean.view(-1)
                flat_mean[occupied] = (
                    flat_sum[occupied] / flat_count[occupied].to(torch.float32)
                )

        self.last_diagnostics = {
            'num_input_points': int(points.shape[0]),
            'num_roi_points': int(roi_points.shape[0]),
            'num_admitted_points': int(segment_key.shape[0]),
            'num_active_pillars': int(num_voxels),
            'protective_clamp_count': clamp_count,
            'overflow_pillars': overflow_pillars,
            'overflow_points': overflow_points,
        }
        return count_hist, intensity_mean

    def _pillar_centers(self, voxel_coords):
        coords = _integer_tensor(voxel_coords, 'voxel_coords')
        center_x = self.x_min + (coords[:, 3].float() + 0.5) * self.voxel_x
        center_y = self.y_min + (coords[:, 2].float() + 0.5) * self.voxel_y
        if self.coord_mode == 'normalized_xy':
            center_x = 2.0 * (center_x - self.x_min) / (self.x_max - self.x_min) - 1.0
            center_y = 2.0 * (center_y - self.y_min) / (self.y_max - self.y_min) - 1.0
        return torch.stack((center_x, center_y), dim=1).float()

    def forward(self, batch_dict, **kwargs):
        required = ('points', 'voxel_coords', 'batch_size')
        missing = [name for name in required if name not in batch_dict]
        if missing:
            raise KeyError(f'PillarHistVFE missing batch_dict keys: {missing}')
        points = batch_dict['points']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = int(batch_dict['batch_size'])
        count_hist, intensity_mean = self.build_histograms(
            points, voxel_coords, batch_size
        )

        if voxel_coords.shape[0] == 0:
            batch_dict['pillar_features'] = torch.empty(
                (0, self.get_output_feature_dim()),
                dtype=torch.float32,
                device=points.device,
            )
            return batch_dict

        with torch.autocast(device_type=points.device.type, enabled=False):
            count_features = count_hist.float()
            intensity_features = intensity_mean.float()
            centers = self._pillar_centers(voxel_coords).to(points.device)
            histogram_features = torch.cat(
                (count_features, intensity_features, centers.float()), dim=1
            )
            pillar_features = self.projection(histogram_features.float()).float()
        self.last_diagnostics.update({
            'count_hist_storage_dtype': str(count_hist.dtype),
            'count_feature_dtype': str(count_features.dtype),
            'intensity_mean_dtype': str(intensity_features.dtype),
            'center_dtype': str(centers.dtype),
            'histogram_feature_dtype': str(histogram_features.dtype),
            'pillar_feature_dtype': str(pillar_features.dtype),
        })
        batch_dict['pillar_features'] = pillar_features
        return batch_dict
