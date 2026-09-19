# PillarHist R8 seed666 Partial Closure Report

## Decision

Original R8 execution status: `PARTIAL`.
Seed666 paired admission: `PASS_COMPLETE`.
Official three-seed scientific label: `NOT GENERATED`.

Seeds 667 and 668 (four tracks) were explicitly deferred for time constraints. This scope change is recorded and must be disclosed in every downstream use of these results.

## Seed666 full-validation result

| Track | Car mAP(E/M/H) | Pedestrian mAP(E/M/H) | Cyclist mAP(E/M/H) | Moderate macro |
|---|---:|---:|---:|---:|
| PP_OFFICIAL_FULL | 80.288610 | 51.067238 | 69.642738 | 64.564172 |
| PH_PAPER_LITERAL_FULL | 79.854214 | 50.836248 | 64.017427 | 62.284147 |
| PH - PP | -0.434396 | -0.230990 | -5.625311 | -2.280025 |

The seed666 observation is negative for both Car and Pedestrian and therefore does not reproduce the paper's positive direction on this seed. It is not an official `NOT_SUPPORTED` label because the frozen rule requires three paired seeds.

## Validity and fairness

- Both tracks completed 80 epochs / 74,240 optimizer steps with finite losses, positive anchors, gradient audits, recovery continuity, and full 3,769-frame validation.
- All 74240 paired optimizer-step input checksums match.
- All 240 expanded replay samples match; all epoch roots and the final root match.
- Final pytest: 51 tests, 0 failures, 0 errors, 1 skipped/xfail.
- Fresh-process final checkpoint load and inference: PASS (PP_OFFICIAL_FULL).
- Backup archive SHA-256 verification: PASS (`376e50c7451f51ae9369259c723a8bba96530b11c396e89e927c41a686128f49`).
- Primary checkpoints are epoch 80 (`LAST_EPOCH`); no best-validation selection was used.

## Paper comparison and scope

`PAPER_PROTOCOL_MISMATCH=true`. This is a local OpenPCDet FP32 controlled reproduction, not evidence for the paper's full environment, INT8, TensorRT, or target-hardware speed.

## Evidence

- `outputs/pillarhist/r8/R8_seed666_decision.json`
- `outputs/pillarhist/r8/R8_seed666_fairness_audit.json`
- `outputs/pillarhist/r8/R8_seed666_metrics.json`
- `outputs/pillarhist/r8/R8_seed666_paper_comparison.json`
- `outputs/pillarhist/r8/R8_seed666_efficiency.json`
- `outputs/pillarhist/r8/seed666/pair_admission.json`
- `outputs/pillarhist/r8/seed666/final_checkpoint_reload_probe.json`
- `outputs/pillarhist/r8/tests/seed666_final_junit.xml`
- `outputs/pillarhist/r8/R8_seed666_backup_manifest.json`
