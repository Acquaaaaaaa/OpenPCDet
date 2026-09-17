# PillarHist R7 Decision Report

## Decision

R7 closes as `INCONCLUSIVE`.

The two candidates rechecked with seed 667 remain inside the frozen 1.0 absolute-AP uncertainty margin. No single R8 configuration is automatically selected, and R8 must not start without human review.

## Frozen protocol

- Primary metric: full KITTI validation macro of Car/Pedestrian/Cyclist 3D AP_R40 Moderate
- Primary seed: 666; tie-review seed: 667
- 5000 optimizer steps, batch 2, workers 0, AMP disabled, native Adam OneCycle
- Tie margin: 1.0 absolute AP point
- Reduction path: `deterministic_segment`
- Validation subset: frozen 256-frame manifest; final decision uses all 3769 validation frames

## Seed 666 candidate results

| Rank | Track | Car | Pedestrian | Cyclist | Macro |
|---:|---|---:|---:|---:|---:|
| 1 | `PH_NORM_LINEAR` | 67.223 | 42.500 | 44.156 | 51.293 |
| 2 | `PH_NORM_BNRELU` | 66.129 | 40.822 | 46.727 | 51.226 |
| 3 | `PH_RAW_LINEAR` | 66.501 | 39.774 | 44.793 | 50.356 |
| 4 | `PH_RAW_BNRELU` | 66.522 | 38.044 | 46.083 | 50.216 |

The top-two gap was 0.067 AP, so only `PH_NORM_LINEAR` and `PH_NORM_BNRELU` entered the pre-registered seed 667 review.

## Tie review

| Track | Seed 666 | Seed 667 | Two-seed mean |
|---|---:|---:|---:|
| `PH_NORM_LINEAR` | 51.293 | 52.749 | 52.021 |
| `PH_NORM_BNRELU` | 51.226 | 52.053 | 51.640 |

The two-seed mean gap is 0.381 AP, still below 1.0 AP. The result is therefore `INCONCLUSIVE`, not a forced simplicity tie-break or a claim of statistical superiority.

## Fairness and validity audit

- Seed 666 full 5000-step input checksum equality across five tracks: `True`
- Seed 667 full 5000-step input checksum equality across the two reviewed tracks: `True`
- Shared replay hash and canonical backend per seed: `True`
- Optimizers were created only after canonical weights were copied.
- Every completed run has 5000 finite-loss steps with positive anchors, 50/50 passing gradient audits, finite BN running statistics, and an exact fixed-next-batch resume check.
- Overall fairness audit: `PASS`

## PP short control

`PP_SHORT_CONTROL` passed all frozen sanity gates. Its first/last 200-step median losses were 1.7137 and 0.6847; full-validation Car AP was 65.958, macro AP was 48.412, and predictions were non-empty. It is not part of the PillarHist ranking.

## Controlled retry

The first `PP_SHORT_CONTROL` attempt stopped at the frozen step-100 gradient diagnostic because the runner selected `Backbone block[0][0]`, which is `ZeroPad2d`, as though it were a trainable convolution. No checkpoint or result was produced. Amendment 1 changed only representative-parameter selection, from runner SHA-256 `8a746162...1040` to `920de583...23ea6`; replay, canonical initialization, model mathematics, optimizer, scheduler, thresholds, and validation were unchanged. A 100-step preflight passed, and the allowed `PP_SHORT_CONTROL_attempt2` completed the formal protocol.

## Efficiency records

| Run | Accounted wall (s) | Observed protocol steps/s |
|---|---:|---:|
| `seed666:PH_RAW_LINEAR` | 1829.3 | 3.117 |
| `seed666:PH_NORM_LINEAR` | 1895.1 | 3.037 |
| `seed666:PH_RAW_BNRELU` | 1891.0 | 3.025 |
| `seed666:PH_NORM_BNRELU` | 1871.2 | 3.088 |
| `seed666:PP_SHORT_CONTROL` | 1847.2 | 3.056 |
| `seed667:PH_NORM_LINEAR` | 1801.9 | 3.166 |
| `seed667:PH_NORM_BNRELU` | 1808.5 | 3.137 |

The observed protocol rate includes checkpointing and intermediate validation pauses. CUDA memory is reported separately from a fixed post-training resume-probe backward pass and is diagnostic only:

| Track | Peak allocated (MiB) | Incremental peak (MiB) |
|---|---:|---:|
| `PH_NORM_BNRELU` | 1680.4 | 1605.9 |
| `PH_NORM_LINEAR` | 1678.9 | 1604.4 |
| `PH_RAW_BNRELU` | 1680.4 | 1605.9 |
| `PH_RAW_LINEAR` | 1678.9 | 1604.4 |
| `PP_SHORT_CONTROL` | 1911.3 | 1836.9 |

## Verification

The final repository verification completed with `41 passed, 1 xfailed`. The expected failure is the separately documented official Scatter empty-input limitation and is not an R7 regression.

## Scope limits

This short-run screen does not establish paper-level or official KITTI accuracy, full-training convergence, statistical significance, PTQ/INT8 behavior, TensorRT performance, or target-hardware deployment speed. R6 did not promote the optimized reduction path, so no deployment-speedup claim is supported. Human review is required before any R8 plan.

## Evidence

- Seed 666 preparation: `/home/ksyang/OpenPCDet/outputs/pillarhist/r7/20260917T064934Z_seed666`
- Seed 667 preparation: `/home/ksyang/OpenPCDet/outputs/pillarhist/r7/20260917T095700Z_seed667`
- Machine-readable decision: `/home/ksyang/OpenPCDet/outputs/pillarhist/r7/R7_decision.json`
- Fairness audit: `/home/ksyang/OpenPCDet/outputs/pillarhist/r7/R7_fairness_audit.json`
- Efficiency record: `/home/ksyang/OpenPCDet/outputs/pillarhist/r7/R7_efficiency.json`
