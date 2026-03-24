# Member 1 Plan V4

## Purpose

This is the updated execution plan for getting `DeepSeek-V2-Lite` running in this repo with `vAttention`, while preserving the actual research goal:

- compare fragmentation and allocator behavior for MHA, GQA, and MLA
- measure MLA residency using only the persistent resident MLA payload
- keep dense reconstructed K/V transient

This plan is intentionally incremental and preserves the historical record from earlier versions:

- [member-1-mla-plan.md](/home/anodyine/repos/vattention/docs/member-1-mla-plan.md)
- [member-1-mla-plan-v2.md](/home/anodyine/repos/vattention/docs/member-1-mla-plan-v2.md)
- [member-1-mla-plan-v3.md](/home/anodyine/repos/vattention/docs/member-1-mla-plan-v3.md)

V4 keeps the core direction from V3, but updates the remaining work based on what we learned during implementation:

- runtime-wrapper integration is its own major phase
- paged MLA attention needs explicit gated milestones
- “attention works” and “model inference works” must stay separate
- allocator/accounting validation is part of the main path, not cleanup

## Goal

Reach a working path that can run `DeepSeek-V2-Lite` on the target `4 x RTX 3090` hardware in this repo using `vAttention`, and do it in a way that preserves valid MLA memory-accounting and fragmentation measurements.

This requires both:

1. a correct MLA execution path under tensor parallelism
2. correct allocator-visible accounting of only the resident MLA cache payload

## Design Constraints We Are Keeping From V3

These remain correct and should not change:

1. Keep FlashAttention.
2. Reconstruct dense K/V immediately before the attention call.
3. Count only resident MLA cache state as persistent KV-cache residency.
4. Keep reconstructed dense K/V transient.
5. Use the shared spec layer as the boundary between Python and CUDA.
6. Reuse the repo’s existing tensor-parallel framework rather than building a separate MLA-specific parallel stack.

## What We Learned Since V3

The current implementation work clarified several boundaries that V3 did not separate strongly enough.

### 1. Runtime boundary stabilization is its own phase

We now have:

- a contiguous MLA reference path
- a model-side DeepSeek MLA scaffold
- a Sarathi attention-wrapper bridge
- a layer-cache object carrying runtime `kv_cache` plus resident MLA cache
- an explicit MLA wrapper input contract

This is enough to show that model-side MLA execution and runtime-wrapper integration are different problems and should be tracked separately.

### 2. Paged MLA attention should be gated before broader model bring-up

The first real paged milestone is not “DeepSeek inference works.”

The real gating order is:

1. one wrapper can consume resident MLA inputs directly
2. paged MLA attention works
3. paged MLA matches contiguous MLA
4. then model/runtime inference wiring becomes meaningful

### 3. The research goal depends on wrapper/accounting correctness

The objective is not only to run the model.

It is also to ensure that:

- allocator-visible bytes come from resident MLA components
- transient dense K/V is not counted as persistent residency
- fragmentation results remain valid after the runtime path moves from dense assumptions to MLA-aware wrappers

### 4. “Attention-only execution” is useful but not the same as inference

We now have attention-only decoder/model execution scaffolding, but:

- weights are not loaded
- MoE is not implemented
- full model semantics are not present

So the plan needs separate milestones for:

- attention correctness
- attention-only model execution
- first runnable non-MoE inference path
- full DeepSeek-V2-Lite inference

## Current Status

### Completed or largely in place

#### Phase 1 groundwork from V3

- Python/CUDA cache/init boundary for MLA component-spec initialization
- extension entrypoint for component-spec KV cache init
- allocator sizing updates based on resident MLA cache geometry
- allocator/CUDA integration tests

#### Phase 2 groundwork from V3

- DeepSeek-V2 config recognition
- DeepSeek-V2 model registration
- tensor-parallel-aware model scaffold

#### Phase 3 contiguous/reference work

- resident MLA cache helpers
- dense local K/V reconstruction helpers
- projection path from hidden states into MLA query/cache components
- contiguous MLA attention reference path
- mixed-length and `tp>1` correctness checks
- attention-only decoder/model stack wiring

#### Runtime boundary work discovered after V3

- backend-bridge path for DeepSeek MLA attention
- Sarathi attention-wrapper bridge
- combined layer-cache object carrying runtime `kv_cache` plus resident MLA cache
- explicit MLA wrapper input contract for wrapper-native execution

### Not yet complete

- actual MLA-capable paged attention wrapper implementation
- real paged MLA attention execution under `vAttention`
- parity validation between paged MLA and contiguous MLA
- first runnable DeepSeek inference path without MoE
- MoE support
- full DeepSeek-V2-Lite inference
- final telemetry/experiment support on top of validated MLA accounting

## Revised Milestone Order

The revised milestone order is:

1. component-spec init works in the extension
2. contiguous MLA attention works
3. runtime MLA boundary is stabilized
4. first MLA-capable paged wrapper works
5. paged MLA attention matches contiguous MLA
6. first runnable DeepSeek path without MoE
7. paged DeepSeek execution works under tensor parallelism
8. full DeepSeek-V2-Lite with MoE executes
9. allocator/fragmentation experiments run on trusted MLA accounting

## Tensor Parallelism Requirement

This remains mandatory.

The repo already supports tensor parallelism for existing models, but MLA-specific correctness still has to be implemented and validated explicitly.

The relevant requirements remain:

1. projection compatibility with existing tensor-parallel layers
2. explicit local-vs-global MLA shape definitions
3. per-rank resident cache specification
4. per-rank allocator sizing and page-capacity calculations
5. per-rank wrapper reconstruction behavior
6. correctness validation for both `tp=1` and `tp>1`

Success should still be defined in terms of the intended multi-GPU target, not just single-rank execution.

## Work Plan V4

### Phase 1: Finish the Python-to-CUDA cache/init boundary

Keep from V3:

1. add an MLA-aware extension initialization entrypoint
2. teach the CUDA extension to understand resident MLA cache components
3. update allocator sizing to use the shared resident-cache model
4. add allocator/CUDA integration tests

Status:

- effectively complete for the current implementation direction

### Phase 2: Add DeepSeek-V2-Lite config/model support

Keep from V3:

5. add DeepSeek-V2-Lite config recognition
6. register a new DeepSeek-V2-Lite model implementation

Status:

- effectively complete as scaffolding
- not complete as full inference support

### Phase 3: Build a contiguous MLA reference path

Keep from V3:

7. implement DeepSeek-V2-Lite MLA attention in a contiguous reference path
8. implement the projection and reconstruction path for MLA attention
9. add attention-only correctness tests

Status:

- substantially complete
- this phase should remain the reference baseline for later paged parity tests

### Phase 4: Stabilize the runtime MLA boundary

New phase in V4.

10. define the wrapper-facing MLA contract explicitly

- package:
  - query activations
  - new resident MLA cache components
  - past resident MLA cache
  - runtime `kv_cache` handle
  - local KV up-projection weights
  - local MLA dimensions

11. define the runtime layer-cache object for MLA execution

- ensure each layer can carry:
  - the runtime cache handle used by existing execution paths
  - the resident MLA cache state required for reconstruction

12. preserve a controlled fallback boundary

- if a wrapper does not support MLA natively, the model-side bridge may still reconstruct dense local K/V and use the dense wrapper contract
- this fallback exists only to preserve bring-up velocity and should not become the final paged MLA design

Why this phase exists:

- this is the seam between model correctness work and paged runtime work
- it must be explicit before implementing a real MLA wrapper

Status:

- underway and partially complete

### Phase 5: Add the first MLA-capable paged wrapper

This phase replaces the older, broader V3 wrapper step with a more explicit gating milestone.

13. implement `forward_mla(...)` on one runtime wrapper path

- start with one backend only
- the wrapper should accept resident MLA inputs directly
- the wrapper should reconstruct dense local K/V at the wrapper boundary, not require permanent model-side dense-KV bridging

14. define wrapper-side MLA cache read/write behavior

- read resident MLA components from paged cache
- append current resident MLA state
- reconstruct dense local K/V immediately before attention
- ensure only resident MLA components are written back persistently

15. add wrapper-focused MLA tests

- validate input contract handling
- validate writeback behavior
- validate decode reuse
- validate tensor-parallel-local shapes

Why this phase is separate:

- “wrapper supports MLA” is the real first paged MLA milestone

### Phase 6: Validate paged MLA attention and accounting

16. reach first working paged MLA attention execution

- run paged MLA attention successfully under `vAttention`
- confirm:
  - cache init succeeds
  - prefill attention succeeds
  - decode attention succeeds

17. compare paged MLA against contiguous MLA

- validate parity between:
  - contiguous MLA reference path
  - paged MLA wrapper path
- validate both:
  - `tp=1`
  - at least one meaningful `tp>1` setting

18. validate resident-memory accounting at the wrapper boundary

- confirm:
  - allocator-visible persistent bytes are resident MLA bytes only
  - transient dense reconstructed K/V is not counted as persistent cache
  - wrapper-side MLA execution does not silently restore dense-KV accounting assumptions

19. add allocator/fragmentation validation cases for MLA paging

- sequence-length sweeps
- expected page-growth checks
- expected tokens-per-page behavior
- expected fragmentation accounting
- per-rank sizing checks under tensor parallelism

Why this phase is central:

- this is where the implementation becomes useful for the actual fragmentation study

### Phase 7: Reach first runnable DeepSeek path without MoE

This phase sharpens the old V3 “first working inference without MoE” milestone.

20. add the minimal remaining non-MoE model path needed for bring-up

- if necessary:
  - temporary feedforward fallback
  - minimal output path
  - weight-loading subset sufficient for early execution

21. reach first runnable contiguous DeepSeek path without MoE

- run prompt prefill
- run at least one decode step

22. reach first runnable paged DeepSeek path without MoE

- run the same basic execution flow using the paged MLA wrapper path

Why this phase is separate:

- it distinguishes “paged MLA attention works” from “a runnable model path exists”

### Phase 8: Add full DeepSeek-V2-Lite support

Keep from V3, but make the dependency explicit.

23. implement MoE support for DeepSeek-V2-Lite

- expert routing
- expert parameter loading
- MoE execution path
- compatibility with tensor/pipeline-parallel expectations

24. re-run MLA validation with MoE enabled

- ensure MoE integration does not perturb the validated MLA attention/runtime path

25. reach first full DeepSeek-V2-Lite inference

- end-to-end inference through the real model architecture in this repo

### Phase 9: Add telemetry and experiment support

Keep from V3, but only after the runtime/accounting path is trusted.

26. add asynchronous telemetry reporting after MLA accounting is validated

- sequence-length milestone reporting
- resident MLA cache metrics
- no persistent accounting of transient dense K/V

27. run MLA experiments for the fragmentation study

- compare MHA, GQA, and MLA under the corrected resident-cache accounting model

## Success Criteria

There are several different “done” states, and the plan should keep them distinct.

### Attention correctness done

- contiguous MLA reference path is correct
- paged MLA wrapper path matches it

### Runtime integration done

- a real wrapper consumes resident MLA inputs directly
- paged MLA attention works under `vAttention`
- allocator-visible accounting remains resident-cache correct

### Bring-up done

- DeepSeek path runs without MoE
- prefill and decode both work

### Full model done

- DeepSeek-V2-Lite runs with MoE under the intended tensor-parallel setup

### Research readiness done

- fragmentation/accounting results are collected on the trusted paged MLA implementation

## Recommended Immediate Next Steps

From the current repository state, the highest-priority next tasks are:

1. implement `forward_mla(...)` on one wrapper path
2. move dense local K/V reconstruction to the wrapper boundary for that path
3. validate paged MLA attention parity against the contiguous reference path
4. validate that persistent accounting still reflects resident MLA bytes only

## Summary

V4 keeps the core design and most of the milestone logic from V3.

What changed is the structure of the remaining work:

- runtime-wrapper stabilization is now a first-class phase
- paged MLA attention has explicit gating milestones
- model bring-up is separated from attention correctness
- accounting validation is elevated because it is part of the research objective, not optional polish

This should make the remaining path clearer while preserving the historical intent and technical direction of V3.
