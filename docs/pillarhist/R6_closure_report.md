# PillarHist R6 Closure Report

## Decision

R6 closes as `PASS_NOT_PROMOTED`.

- `TST-018`: `PASS`
- Implemented PH_OPT candidate: `compact_lookup_segment`
- R7 reduction path: `deterministic_segment`
- Deployment-speedup claim: not supported

The optimized candidate preserves the frozen PillarHist mathematics and full-model gradients, but it does not satisfy the pre-registered performance and memory promotion gates.

## Frozen entry

- Baseline commit: `e90de060bceba2a228095eea01718b4ded55b68b`
- Branch: `pointpillars-pillarhist`
- Specification: `PillarHist_OpenPCDet_deployment_guideline_v1.3.md`
- Pre-registration: `PillarHist_R6-R7_preregistration_v1.3.json`
- Reference profile: `outputs/pillarhist/r6/PH_PAPER_LITERAL_REF/20260917T062100Z_seed666/profile/reference_profile.json`
- Final PH_OPT run: `outputs/pillarhist/r6/PH_OPT/20260917T063154Z_seed666`

## Reference profile

Across the frozen validation and training fixtures, the mean component p50 fractions were:

| Component | Fraction |
|---|---:|
| histogram key/lookup | 55.09% |
| intensity sum/mean | 23.68% |
| count reduction | 10.76% |
| concat/center | 9.81% |
| projection | 0.66% |

This evidence justified optimizing active-pillar lookup before considering a custom CUDA kernel.

## Candidate history

The first native candidate used compact active-row lookup plus float `scatter_add`. It kept `Hp` exact and kept `Hi`/features within their direct tolerances, but failed the frozen projection-gradient tolerance on the real training fixture because atomic accumulation-order noise was amplified during backward. The failed formal run remains at:

`outputs/pillarhist/r6/PH_OPT/20260917T062950Z_seed666`

No threshold was changed. The atomic candidate was removed and replaced with `compact_lookup_segment`, which combines compact active-row lookup with the reference stable segment reduction.

## TST-018 correctness

The final candidate passed:

- exact `Hp`, active-row order and labels;
- `Hi`, projection input/features, full detector outputs and loss at the frozen tolerances;
- projection and representative backend gradients at `rtol=1e-4, atol=1e-6`;
- 20 repeated loss/backward audits from the same state;
- outer AMP with the PillarHist path remaining FP32;
- CPU structural variants, `V=0/1`, synthetic full-detector backward, and real validation/training fixtures.

In the final 20-repeat real-fixture audit, projection-gradient worst absolute error was zero; the minimum representative backend-gradient cosine remained above the frozen `0.9999` threshold.

## Pre-registered benchmark

Protocol: CUDA Event timing, explicit synchronization, 30 warmups, five rounds, REF/OPT order alternated per sample and round, 200 VFE samples per implementation per round, and 100 end-to-end samples per implementation per round.

| Metric | REF | OPT | Outcome |
|---|---:|---:|---|
| VFE p50 | 5.0694 ms | 4.9690 ms | 1.981% faster |
| VFE p95 | 7.5234 ms | 7.3634 ms | diagnostic |
| VFE throughput | 394.52 samples/s | 402.50 samples/s | diagnostic |
| End-to-end p50 | 94.1921 ms | 94.7568 ms | 0.599% slower |
| End-to-end p95 | 107.0307 ms | 109.5740 ms | diagnostic |
| End-to-end throughput | 21.233 samples/s | 21.107 samples/s | diagnostic |
| VFE incremental peak memory | 22,032,384 B | 24,442,880 B | +10.941% |
| End-to-end incremental peak memory | 496,447,488 B | 496,447,488 B | 0% |

The five VFE round p50 improvements were `6.425%`, `2.475%`, `1.385%`, `1.951%`, and `0.037%`. The end-to-end paired mean difference (`REF - OPT`) was `-1.0359 ms`, with 95% CI `[-1.7247, -0.3471] ms`.

## Gate evaluation

| Frozen gate | Result |
|---|---|
| VFE p50 improvement at least 10% | FAIL (`1.981%`) |
| Positive VFE p50 in at least 4/5 rounds | PASS (`5/5`) |
| End-to-end paired 95% CI lower bound above 0 | FAIL (`-1.7247 ms`) |
| Peak-memory increase at most 10% or justified | FAIL for VFE (`10.941%`); no compensating end-to-end benefit |

Therefore PH_OPT is not promoted. R7 must use `deterministic_segment` for all four PillarHist candidates.

## Scope limit

R6 demonstrates correctness of the retained configurable candidate and rejects its promotion on this RTX 5060 Laptop GPU environment. It does not establish a general runtime result for other GPUs, custom CUDA kernels, TensorRT, INT8, or target deployment hardware.
