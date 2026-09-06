# MLA Optimization Plan

## Objective

Speed up MLA decode latency and throughput while preserving correctness and keeping non-MLA (MHA/GQA) paths stable.

## Current Bottlenecks (Observed in Branch)

1. Python-heavy per-sequence decode loop in MLA wrapper.
2. Frequent host-side metadata handling (`tolist()` and Python list ops) on the hot path.
3. Per-sequence reconstruction of dense K/V before each attention call.
4. Multiple small attention launches instead of one batched varlen decode launch.
5. Cache read/write + reconstruction + attention split across many Python/CUDA boundaries.

## Success Criteria

1. MLA decode tokens/sec improves materially versus current branch at the same batch/context.
2. P50/P99 decode latency per step is reduced for mixed prompt/decode workloads.
3. MHA/GQA performance remains within noise vs `main` (no meaningful regression).
4. Numerical parity passes for MLA output against current reference path.
5. Allocator and cache-accounting semantics remain correct for MLA resident cache.

## Phased Execution Plan

### Phase 0: Baseline and Instrumentation

1. Add reproducible benchmark configs:
   1. single-sequence decode
   2. multi-sequence decode
   3. mixed prompt+decode
2. Collect baseline for:
   1. tokens/sec
   2. per-step latency breakdown
   3. Python vs CUDA time split
3. Add profiling hooks around:
   1. wrapper metadata prep
   2. cache read/write
   3. dense K/V reconstruction
   4. flash-attention invocation

Deliverable: baseline report and top-3 verified hotspots.

### Phase 1: Low-Risk Python Hot-Path Cleanup (Quick Wins)

1. Remove GPU-to-CPU sync points from decode path metadata handling.
2. Keep runtime metadata in tensor form; avoid repeated Python list conversions.
3. Hoist invariant computations out of inner loops.
4. Minimize temporary object construction for cache chunking.
5. Guard all MLA-only logic behind strict MLA checks (no non-MLA path pollution).

Deliverable: measurable improvement without changing kernel behavior.

### Phase 2: Move Cache Runtime Ops into C++ Extension

1. Implement C++/CUDA-side MLA runtime helpers for:
   1. cache chunk addressing
   2. component cache read/write
   3. sequence-offset handling
2. Replace Python cache-manipulation helpers on decode hot path with extension calls.
3. Expose compact extension API taking device tensors (no host metadata dependency).

Deliverable: Python becomes orchestration layer; runtime cache operations become native.

### Phase 3: Batched MLA Decode Attention Path

1. Replace per-sequence flash-attention loop with batched varlen decode execution.
2. Build packed Q/K/V view generation for all active sequences in one step.
3. Ensure metadata format supports both prefill+decode mixed batches.
4. Keep fallback path available behind a feature flag for safe rollback.

Deliverable: one batched decode attention execution path for MLA.

### Phase 4: Fusion of Reconstruction + Attention (Major Gain)

1. Prototype fused path that avoids materializing full dense K/V in Python-level flow.
2. Implement either:
   1. custom CUDA kernel sequence (reconstruct-on-the-fly + attention), or
   2. Triton/CUDA fused pre-kernel feeding FA-compatible buffers with minimal staging.
3. Validate numerical parity and memory usage behavior.

Deliverable: reduced memory movement and kernel launch overhead in MLA decode.

### Phase 5: Hardening, Regression Gates, and Rollout

1. Add CI/perf gates:
   1. MLA minimum throughput threshold
   2. MHA/GQA non-regression threshold
2. Add correctness suite:
   1. paged MLA vs contiguous MLA parity
   2. TP=1 and TP>1 coverage
3. Add runtime flags:
   1. `--mla-decode-impl=legacy|batched|fused`
   2. safe default with easy rollback
4. Ship incremental rollout:
   1. enable batched path first
   2. enable fused path after soak and parity confidence

Deliverable: production-safe MLA acceleration with guardrails.

## Non-MLA Safety Plan

1. Keep MHA/GQA code paths unchanged unless required for shared interfaces.
2. Branch early by cache architecture/model type to isolate MLA logic.
3. Track MHA/GQA benchmarks for every MLA optimization PR.
4. Block merge on non-MLA regression outside agreed tolerance.

## Recommended Implementation Order (Practical)

1. Phase 0 and Phase 1 in the next PR.
2. Phase 2 in one focused extension PR.
3. Phase 3 in one decode-kernel integration PR.
4. Phase 4 as an experimental branch, then merge behind a flag.
5. Phase 5 as final stabilization and default switch.

## Risks and Mitigations

1. Risk: correctness drift from aggressive fusion.
   Mitigation: maintain legacy reference path and enforce parity tests per layer/step.
2. Risk: accidental regressions for MHA/GQA.
   Mitigation: explicit architecture gating + required perf regression checks.
3. Risk: high implementation complexity in custom kernels.
   Mitigation: stage via batched varlen path first, then fuse incrementally.

## Exit Condition

This plan is complete when MLA decode is consistently faster than current branch under representative workloads, with validated correctness and no meaningful MHA/GQA regression.
