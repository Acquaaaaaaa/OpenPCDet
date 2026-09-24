"""Consumer-aware activation operand mapping for PointPillars MAC layers.

The mapper emits physical CIM groups in deterministic reference order without
materialising a full-frame im2col tensor.  Every physical row is classified as
exactly one of: logical valid operand, convolution boundary padding, group-tail
padding, or an invalid fixed PFN point slot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OperandGroupBatch:
    """A consecutive batch of physical CIM groups."""

    digits: torch.Tensor
    logical_valid: torch.Tensor
    boundary_padding: torch.Tensor
    tail_padding: torch.Tensor
    pfn_invalid_slot: torch.Tensor
    zero_insertion: torch.Tensor | None = None

    @property
    def zero_insertion_mask(self) -> torch.Tensor:
        if self.zero_insertion is None:
            return torch.zeros_like(self.logical_valid)
        return self.zero_insertion

    def validate(self, group_size: int) -> None:
        if self.digits.ndim != 3 or self.digits.shape[1:] != (group_size, 4):
            raise ValueError(
                f"digits must have shape [groups, {group_size}, 4], "
                f"got {tuple(self.digits.shape)}"
            )
        expected = self.digits.shape[:2]
        masks = (
            self.logical_valid,
            self.boundary_padding,
            self.tail_padding,
            self.pfn_invalid_slot,
            self.zero_insertion_mask,
        )
        if any(mask.shape != expected or mask.dtype != torch.bool for mask in masks):
            raise ValueError("all row-classification masks must be bool [groups, rows]")
        classification_count = sum(mask.to(torch.int8) for mask in masks)
        if not bool(torch.all(classification_count == 1)):
            raise ValueError("every physical row must have exactly one classification")
        padding = (
            self.boundary_padding
            | self.tail_padding
            | self.pfn_invalid_slot
            | self.zero_insertion_mask
        )
        if padding.any() and bool(torch.any(self.digits[padding] != 0)):
            raise ValueError("non-logical rows must contain zero Booth digits")


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"expected a pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _padding4(value: int | Sequence[int]) -> tuple[int, int, int, int]:
    """Return padding as left, right, top, bottom."""

    if isinstance(value, Sequence):
        if len(value) == 2:
            height, width = int(value[0]), int(value[1])
            return width, width, height, height
        if len(value) == 4:
            left, right, top, bottom = map(int, value)
            return left, right, top, bottom
        raise ValueError(f"expected padding int, pair, or quadruple, got {value!r}")
    padding = int(value)
    return padding, padding, padding, padding


def _validate_digits(digits: torch.Tensor) -> None:
    if not isinstance(digits, torch.Tensor):
        raise TypeError("digits must be a torch.Tensor")
    if digits.ndim < 2 or digits.shape[-1] != 4:
        raise ValueError(f"digits must end in a four-digit axis, got {tuple(digits.shape)}")
    if digits.numel() and not bool(torch.all((digits >= -2) & (digits <= 2))):
        raise ValueError("Booth digits must be in {-2, -1, 0, 1, 2}")


def _pack_equal_streams(
    stream_digits: torch.Tensor,
    logical_valid: torch.Tensor,
    boundary_padding: torch.Tensor,
    pfn_invalid_slot: torch.Tensor,
    *,
    group_size: int,
) -> OperandGroupBatch:
    """Pack equal-length streams while preserving stream-local tail boundaries."""

    if stream_digits.ndim != 3 or stream_digits.shape[-1] != 4:
        raise ValueError("stream_digits must have shape [streams, K, 4]")
    stream_count, reduction_length, _ = stream_digits.shape
    expected = (stream_count, reduction_length)
    if any(mask.shape != expected for mask in (logical_valid, boundary_padding, pfn_invalid_slot)):
        raise ValueError("stream masks do not match stream_digits")
    groups_per_stream = math.ceil(reduction_length / group_size)
    physical_length = groups_per_stream * group_size
    tail_length = physical_length - reduction_length

    if tail_length:
        digit_tail = torch.zeros(
            (stream_count, tail_length, 4), dtype=stream_digits.dtype, device=stream_digits.device
        )
        mask_tail = torch.zeros(
            (stream_count, tail_length), dtype=torch.bool, device=stream_digits.device
        )
        stream_digits = torch.cat((stream_digits, digit_tail), dim=1)
        logical_valid = torch.cat((logical_valid, mask_tail), dim=1)
        boundary_padding = torch.cat((boundary_padding, mask_tail), dim=1)
        pfn_invalid_slot = torch.cat((pfn_invalid_slot, mask_tail), dim=1)

    tail_padding = torch.zeros(
        (stream_count, physical_length), dtype=torch.bool, device=stream_digits.device
    )
    if tail_length:
        tail_padding[:, reduction_length:] = True

    def grouped(tensor: torch.Tensor) -> torch.Tensor:
        suffix = tensor.shape[2:]
        return tensor.reshape(stream_count, groups_per_stream, group_size, *suffix).reshape(
            stream_count * groups_per_stream, group_size, *suffix
        )

    batch = OperandGroupBatch(
        digits=grouped(stream_digits),
        logical_valid=grouped(logical_valid),
        boundary_padding=grouped(boundary_padding),
        tail_padding=grouped(tail_padding),
        pfn_invalid_slot=grouped(pfn_invalid_slot),
    )
    batch.validate(group_size)
    return batch


def iter_linear_groups(
    digits: torch.Tensor,
    module: nn.Linear,
    *,
    group_size: int = 8,
    chunk_streams: int = 4096,
    token_policy: str = "fixed_slots_included",
    valid_token_mask: torch.Tensor | None = None,
) -> Iterator[OperandGroupBatch]:
    """Map Linear operands with one independent stream per token."""

    _validate_digits(digits)
    if group_size <= 0 or chunk_streams <= 0:
        raise ValueError("group_size and chunk_streams must be positive")
    if digits.shape[-2] != module.in_features:
        raise ValueError(
            f"activation K={digits.shape[-2]} does not match Linear in_features={module.in_features}"
        )
    prefix_shape = digits.shape[:-2]
    streams = digits.reshape(-1, module.in_features, 4)
    if valid_token_mask is None:
        token_valid = torch.ones(streams.shape[0], dtype=torch.bool, device=digits.device)
    else:
        if tuple(valid_token_mask.shape) != tuple(prefix_shape):
            raise ValueError(
                f"valid_token_mask shape {tuple(valid_token_mask.shape)} does not match "
                f"Linear token shape {tuple(prefix_shape)}"
            )
        token_valid = valid_token_mask.to(device=digits.device, dtype=torch.bool).reshape(-1)

    if token_policy == "valid_points_only":
        streams = streams[token_valid]
        token_valid = token_valid[token_valid]
    elif token_policy != "fixed_slots_included":
        raise ValueError(f"unknown token_policy: {token_policy}")

    for start in range(0, streams.shape[0], chunk_streams):
        values = streams[start : start + chunk_streams]
        validity = token_valid[start : start + chunk_streams]
        logical = validity[:, None].expand(-1, module.in_features).clone()
        invalid = (~validity)[:, None].expand(-1, module.in_features).clone()
        zeros = torch.zeros_like(logical)
        if invalid.any():
            values = values.clone()
            values[invalid] = 0
        yield _pack_equal_streams(
            values,
            logical,
            zeros,
            invalid,
            group_size=group_size,
        )


def _conv2d_geometry(
    digits: torch.Tensor,
    module: nn.Conv2d,
    effective_padding: int | Sequence[int] | None,
) -> tuple[int, int, tuple[int, int, int, int]]:
    _, _, height, width, _ = digits.shape
    kernel_h, kernel_w = _pair(module.kernel_size)
    stride_h, stride_w = _pair(module.stride)
    dilation_h, dilation_w = _pair(module.dilation)
    padding = _padding4(module.padding if effective_padding is None else effective_padding)
    left, right, top, bottom = padding
    out_h = (height + top + bottom - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + left + right - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv2d geometry produces an empty output")
    return out_h, out_w, padding


def iter_conv2d_groups(
    digits: torch.Tensor,
    module: nn.Conv2d,
    *,
    group_size: int = 8,
    chunk_streams: int = 1024,
    padding_policy: str = "physical_rows_included",
    effective_padding: int | Sequence[int] | None = None,
) -> Iterator[OperandGroupBatch]:
    """Map Conv2d streams in batch/output-row/output-column/K order."""

    _validate_digits(digits)
    if digits.ndim != 5:
        raise ValueError("Conv2d digits must have shape [B, C, H, W, 4]")
    if module.groups != 1:
        raise NotImplementedError("grouped Conv2d requires an explicit group-aware mapping")
    batch_size, channels, height, width, _ = digits.shape
    if channels != module.in_channels:
        raise ValueError("activation channel count does not match Conv2d in_channels")
    if group_size <= 0 or chunk_streams <= 0:
        raise ValueError("group_size and chunk_streams must be positive")

    out_h, out_w, padding = _conv2d_geometry(digits, module, effective_padding)
    left, _, top, _ = padding
    kernel_h, kernel_w = _pair(module.kernel_size)
    stride_h, stride_w = _pair(module.stride)
    dilation_h, dilation_w = _pair(module.dilation)
    reduction_length = channels * kernel_h * kernel_w
    p = torch.arange(reduction_length, device=digits.device)
    channel_index = torch.div(p, kernel_h * kernel_w, rounding_mode="floor")
    kernel_rem = p.remainder(kernel_h * kernel_w)
    kernel_row = torch.div(kernel_rem, kernel_w, rounding_mode="floor")
    kernel_column = kernel_rem.remainder(kernel_w)
    stream_total = batch_size * out_h * out_w

    for start in range(0, stream_total, chunk_streams):
        stream_id = torch.arange(
            start, min(start + chunk_streams, stream_total), device=digits.device
        )
        batch_index = torch.div(stream_id, out_h * out_w, rounding_mode="floor")
        spatial = stream_id.remainder(out_h * out_w)
        output_row = torch.div(spatial, out_w, rounding_mode="floor")
        output_column = spatial.remainder(out_w)
        input_row = (
            output_row[:, None] * stride_h - top + kernel_row[None, :] * dilation_h
        )
        input_column = (
            output_column[:, None] * stride_w - left + kernel_column[None, :] * dilation_w
        )
        valid = (
            (input_row >= 0)
            & (input_row < height)
            & (input_column >= 0)
            & (input_column < width)
        )
        gathered = digits[
            batch_index[:, None],
            channel_index[None, :],
            input_row.clamp(0, height - 1),
            input_column.clamp(0, width - 1),
        ]
        gathered = gathered.clone()
        gathered[~valid] = 0

        if padding_policy == "physical_rows_included":
            zeros = torch.zeros_like(valid)
            yield _pack_equal_streams(
                gathered,
                valid,
                ~valid,
                zeros,
                group_size=group_size,
            )
            continue
        if padding_policy != "ideal_padding_gated":
            raise ValueError(f"unknown padding_policy: {padding_policy}")

        packed_batches: list[OperandGroupBatch] = []
        for row in range(gathered.shape[0]):
            compact = gathered[row, valid[row]]
            if compact.shape[0] == 0:
                continue
            logical = torch.ones((1, compact.shape[0]), dtype=torch.bool, device=digits.device)
            zeros = torch.zeros_like(logical)
            packed_batches.append(
                _pack_equal_streams(
                    compact.unsqueeze(0), logical, zeros, zeros, group_size=group_size
                )
            )
        if packed_batches:
            merged = OperandGroupBatch(
                digits=torch.cat([item.digits for item in packed_batches]),
                logical_valid=torch.cat([item.logical_valid for item in packed_batches]),
                boundary_padding=torch.cat([item.boundary_padding for item in packed_batches]),
                tail_padding=torch.cat([item.tail_padding for item in packed_batches]),
                pfn_invalid_slot=torch.cat([item.pfn_invalid_slot for item in packed_batches]),
            )
            merged.validate(group_size)
            yield merged


def iter_conv_transpose2d_groups(
    digits: torch.Tensor,
    module: nn.ConvTranspose2d,
    *,
    group_size: int = 8,
    chunk_streams: int = 4096,
) -> Iterator[OperandGroupBatch]:
    """Map direct-scatter ConvTranspose2d streams along input channels."""

    _validate_digits(digits)
    if digits.ndim != 5:
        raise ValueError("ConvTranspose2d digits must have shape [B, C, H, W, 4]")
    if module.groups != 1:
        raise NotImplementedError("grouped ConvTranspose2d requires an explicit mapping")
    batch_size, channels, height, width, _ = digits.shape
    if channels != module.in_channels:
        raise ValueError("activation channel count does not match ConvTranspose2d in_channels")
    if group_size <= 0 or chunk_streams <= 0:
        raise ValueError("group_size and chunk_streams must be positive")

    kernel_h, kernel_w = _pair(module.kernel_size)
    stride_h, stride_w = _pair(module.stride)
    dilation_h, dilation_w = _pair(module.dilation)
    pad_h, pad_w = _pair(module.padding)
    output_pad_h, output_pad_w = _pair(module.output_padding)
    out_h = (
        (height - 1) * stride_h - 2 * pad_h
        + dilation_h * (kernel_h - 1) + output_pad_h + 1
    )
    out_w = (
        (width - 1) * stride_w - 2 * pad_w
        + dilation_w * (kernel_w - 1) + output_pad_w + 1
    )
    base = digits.permute(0, 2, 3, 1, 4).reshape(-1, channels, 4)
    offsets_per_position = kernel_h * kernel_w
    stream_total = base.shape[0] * offsets_per_position

    for start in range(0, stream_total, chunk_streams):
        stream_id = torch.arange(
            start, min(start + chunk_streams, stream_total), device=digits.device
        )
        position_index = torch.div(stream_id, offsets_per_position, rounding_mode="floor")
        offset = stream_id.remainder(offsets_per_position)
        kernel_row = torch.div(offset, kernel_w, rounding_mode="floor")
        kernel_column = offset.remainder(kernel_w)

        spatial = position_index.remainder(height * width)
        input_row = torch.div(spatial, width, rounding_mode="floor")
        input_column = spatial.remainder(width)
        output_row = input_row * stride_h - pad_h + kernel_row * dilation_h
        output_column = input_column * stride_w - pad_w + kernel_column * dilation_w
        valid_stream = (
            (output_row >= 0)
            & (output_row < out_h)
            & (output_column >= 0)
            & (output_column < out_w)
        )
        position_index = position_index[valid_stream]
        if position_index.numel() == 0:
            continue
        values = base[position_index]
        logical = torch.ones(
            (values.shape[0], channels), dtype=torch.bool, device=digits.device
        )
        zeros = torch.zeros_like(logical)
        yield _pack_equal_streams(
            values, logical, zeros, zeros, group_size=group_size
        )


def iter_conv_transpose2d_zero_insertion_groups(
    digits: torch.Tensor,
    module: nn.ConvTranspose2d,
    *,
    group_size: int = 8,
    chunk_streams: int = 512,
) -> Iterator[OperandGroupBatch]:
    """Pilot-only zero-insertion GEMM mapping for ConvTranspose2d.

    Inserted zeros are generated analytically rather than materialising the
    expanded activation tensor.
    """

    _validate_digits(digits)
    if digits.ndim != 5:
        raise ValueError("ConvTranspose2d digits must have shape [B, C, H, W, 4]")
    if module.groups != 1:
        raise NotImplementedError("grouped ConvTranspose2d requires an explicit mapping")
    batch_size, channels, height, width, _ = digits.shape
    if channels != module.in_channels:
        raise ValueError("activation channel count does not match ConvTranspose2d in_channels")
    kernel_h, kernel_w = _pair(module.kernel_size)
    stride_h, stride_w = _pair(module.stride)
    dilation_h, dilation_w = _pair(module.dilation)
    pad_h, pad_w = _pair(module.padding)
    output_pad_h, output_pad_w = _pair(module.output_padding)
    expanded_h = (height - 1) * stride_h + 1 + output_pad_h
    expanded_w = (width - 1) * stride_w + 1 + output_pad_w
    effective_pad_h = dilation_h * (kernel_h - 1) - pad_h
    effective_pad_w = dilation_w * (kernel_w - 1) - pad_w
    if effective_pad_h < 0 or effective_pad_w < 0:
        raise NotImplementedError("zero-insertion mapping does not support negative effective padding")
    out_h = (
        (height - 1) * stride_h - 2 * pad_h
        + dilation_h * (kernel_h - 1) + output_pad_h + 1
    )
    out_w = (
        (width - 1) * stride_w - 2 * pad_w
        + dilation_w * (kernel_w - 1) + output_pad_w + 1
    )
    reduction_length = channels * kernel_h * kernel_w
    p = torch.arange(reduction_length, device=digits.device)
    channel_index = torch.div(p, kernel_h * kernel_w, rounding_mode="floor")
    kernel_rem = p.remainder(kernel_h * kernel_w)
    kernel_row = torch.div(kernel_rem, kernel_w, rounding_mode="floor")
    kernel_column = kernel_rem.remainder(kernel_w)
    stream_total = batch_size * out_h * out_w

    for start in range(0, stream_total, chunk_streams):
        stream_id = torch.arange(
            start, min(start + chunk_streams, stream_total), device=digits.device
        )
        batch_index = torch.div(stream_id, out_h * out_w, rounding_mode="floor")
        spatial = stream_id.remainder(out_h * out_w)
        output_row = torch.div(spatial, out_w, rounding_mode="floor")
        output_column = spatial.remainder(out_w)
        expanded_row = (
            output_row[:, None] - effective_pad_h
            + kernel_row[None, :] * dilation_h
        )
        expanded_column = (
            output_column[:, None] - effective_pad_w
            + kernel_column[None, :] * dilation_w
        )
        inside_expanded = (
            (expanded_row >= 0)
            & (expanded_row < expanded_h)
            & (expanded_column >= 0)
            & (expanded_column < expanded_w)
        )
        on_source_grid = (
            expanded_row.remainder(stride_h).eq(0)
            & expanded_column.remainder(stride_w).eq(0)
        )
        source_row = torch.div(
            expanded_row.clamp(min=0), stride_h, rounding_mode="floor"
        )
        source_column = torch.div(
            expanded_column.clamp(min=0), stride_w, rounding_mode="floor"
        )
        source_valid = (
            inside_expanded
            & on_source_grid
            & (source_row < height)
            & (source_column < width)
        )
        gathered = digits[
            batch_index[:, None],
            channel_index[None, :],
            source_row.clamp(0, height - 1),
            source_column.clamp(0, width - 1),
        ].clone()
        gathered[~source_valid] = 0
        boundary = ~inside_expanded
        inserted = inside_expanded & ~source_valid
        zeros = torch.zeros_like(source_valid)
        packed = _pack_equal_streams(
            gathered, source_valid | inserted, boundary, zeros, group_size=group_size
        )
        # _pack_equal_streams classified inserted zeros as logical-valid; replace
        # that classification after applying the same stream-local tail padding.
        groups_per_stream = math.ceil(reduction_length / group_size)
        physical_length = groups_per_stream * group_size
        if physical_length > reduction_length:
            inserted = torch.cat((
                inserted,
                torch.zeros(
                    (inserted.shape[0], physical_length - reduction_length),
                    dtype=torch.bool,
                    device=digits.device,
                ),
            ), dim=1)
        inserted = inserted.reshape(-1, group_size)
        corrected = OperandGroupBatch(
            digits=packed.digits,
            logical_valid=packed.logical_valid & ~inserted,
            boundary_padding=packed.boundary_padding,
            tail_padding=packed.tail_padding,
            pfn_invalid_slot=packed.pfn_invalid_slot,
            zero_insertion=inserted,
        )
        corrected.validate(group_size)
        yield corrected


def iter_logical_contiguous_groups(
    digits: torch.Tensor,
    *,
    group_size: int = 8,
    chunk_streams: int = 65536,
) -> Iterator[OperandGroupBatch]:
    """Software-only baseline that flattens the logical activation tensor."""

    _validate_digits(digits)
    flat = digits.reshape(-1, 4)
    if flat.shape[0] == 0:
        return
    # Keep a single logical stream so only the final global group has a tail.
    for start in range(0, flat.shape[0], chunk_streams * group_size):
        end = min(start + chunk_streams * group_size, flat.shape[0])
        values = flat[start:end]
        is_last = end == flat.shape[0]
        if not is_last and values.shape[0] % group_size:
            end -= values.shape[0] % group_size
            values = flat[start:end]
        logical = torch.ones((1, values.shape[0]), dtype=torch.bool, device=digits.device)
        zeros = torch.zeros_like(logical)
        yield _pack_equal_streams(
            values.unsqueeze(0), logical, zeros, zeros, group_size=group_size
        )


def iter_consumer_groups(
    digits: torch.Tensor,
    module: nn.Module,
    *,
    group_size: int = 8,
    chunk_streams: int = 1024,
    padding_policy: str = "physical_rows_included",
    token_policy: str = "fixed_slots_included",
    valid_token_mask: torch.Tensor | None = None,
    effective_padding: int | Sequence[int] | None = None,
    mapping_mode: str = "consumer_aware_reference_cim",
    conv_transpose_mapping: str = "direct_scatter",
) -> Iterator[OperandGroupBatch]:
    """Dispatch to the mapper matching the MAC consumer type."""

    if mapping_mode == "logical_contiguous":
        yield from iter_logical_contiguous_groups(
            digits, group_size=group_size, chunk_streams=chunk_streams
        )
    elif mapping_mode != "consumer_aware_reference_cim":
        raise ValueError(f"unknown mapping_mode: {mapping_mode}")
    elif isinstance(module, nn.Linear):
        yield from iter_linear_groups(
            digits,
            module,
            group_size=group_size,
            chunk_streams=chunk_streams,
            token_policy=token_policy,
            valid_token_mask=valid_token_mask,
        )
    elif isinstance(module, nn.Conv2d):
        yield from iter_conv2d_groups(
            digits,
            module,
            group_size=group_size,
            chunk_streams=chunk_streams,
            padding_policy=padding_policy,
            effective_padding=effective_padding,
        )
    elif isinstance(module, nn.ConvTranspose2d):
        if conv_transpose_mapping == "direct_scatter":
            yield from iter_conv_transpose2d_groups(
                digits,
                module,
                group_size=group_size,
                chunk_streams=chunk_streams,
            )
        elif conv_transpose_mapping == "zero_insertion":
            yield from iter_conv_transpose2d_zero_insertion_groups(
                digits,
                module,
                group_size=group_size,
                chunk_streams=chunk_streams,
            )
        else:
            raise ValueError(f"unknown ConvTranspose2d mapping: {conv_transpose_mapping}")
    else:
        raise TypeError(f"unsupported MAC consumer type: {type(module).__name__}")


def output_channels(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return int(module.out_channels)
    raise TypeError(f"unsupported MAC consumer type: {type(module).__name__}")
