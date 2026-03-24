# Member 1 Plan V3

## Purpose

This is the execution-oriented version of the MLA plan.

The earlier plan versions are being preserved:

- [member-1-mla-plan.md](/home/anodyine/repos/vattention/docs/member-1-mla-plan.md)
- [member-1-mla-plan-v2.md](/home/anodyine/repos/vattention/docs/member-1-mla-plan-v2.md)

V3 answers a narrower and more operational question:

> What tasks are required to actually run inference with `DeepSeek-V2-Lite` in this repo?

This version therefore includes not only allocator and cache-layout work, but also the missing model-execution, MLA attention, validation, and serving steps needed to reach a first working inference path.

## Current Status

Completed so far:

- cache architecture abstraction
- resident cache byte formulas
- page-capacity helpers
- cache block size centralization
- cache layout and init spec layers
- MLA attention-dimension spec
- resident cache component spec
- explicit Python-to-CUDA init modes and init requests
- Docker-backed unit-test harness for config/cache spec behavior

Not yet completed:

- CUDA support for component-spec MLA cache initialization
- MLA-aware virtual tensor layout in the extension
- DeepSeek-V2-Lite config/model registration
- DeepSeek-V2-Lite attention implementation
- contiguous MLA reference execution
- paged MLA attention wrapper
- attention correctness harness
- full model inference path
- MoE support

## Goal

Reach a working path that can run `DeepSeek-V2-Lite` inference in this repo.

The recommended milestone order is:

1. first working attention-only execution
2. first working contiguous MLA model inference
3. first working paged MLA model inference with `vAttention`
4. validated parity against a reference implementation
5. full DeepSeek-V2-Lite inference including MoE

## Design Constraints

The core design constraints remain:

1. Keep FlashAttention.
2. Reconstruct dense K/V immediately before the attention call.
3. Count only resident MLA cache state as KV-cache residency.
4. Keep reconstructed dense K/V transient.
5. Use the shared spec layer as the boundary between Python and CUDA.

## Work Plan V3

### Phase 1: Finish the Python-to-CUDA cache/init boundary

1. Add an MLA-aware extension initialization entrypoint.

- Add a new extension API for component-based initialization, such as:
  - `init_kvcache_component_spec(...)`
- Make it consume the structured payload already produced by:
  - `VAttentionCacheSpec.to_extension_dict()`
  - `VAttentionInitSpec.get_extension_init_request()`

Why this is required:

- Right now MLA specs intentionally cannot initialize the extension.
- This is the first blocker to paged MLA execution.

2. Teach the CUDA extension to understand resident cache components.

- Update the extension-side init path to interpret component-based cache structure:
  - dense KV uses `k`, `v`
  - MLA uses `kv_latent`, `k_rope`
- Do not re-derive MLA structure ad hoc in C++.
- Consume the Python-provided spec as directly as possible.

Why this is required:

- The extension must know what persistent cache structure it is allocating and paging.

3. Update allocator sizing in the CUDA path to use the shared MLA resident-cache model.

- Replace remaining dense-KV assumptions in:
  - `vattention/vattention.cu`
  - `vattention/utils.h`
- Update:
  - `tokens_per_page`
  - virtual buffer size per token
  - page growth calculations
  - free-block accounting
  - fragmentation calculations

Why this is required:

- MLA paging is not correct until the allocator computes capacity from the resident MLA cache payload rather than dense K/V.

4. Add allocator/CUDA integration tests.

- Validate:
  - Python spec vs CUDA tokens-per-page agreement
  - Python spec vs CUDA resident-byte agreement
  - page-growth boundaries
  - component-spec initialization success

Why this is required:

- This is the first place where Python-only correctness stops being enough.

### Phase 2: Add DeepSeek-V2-Lite model/config support

5. Add DeepSeek-V2-Lite config recognition.

- Extend config loading and model selection so the repo can identify `DeepSeek-V2-Lite` as an MLA model.
- Ensure the needed config fields are exposed:
  - `q_lora_rank`
  - `kv_lora_rank`
  - `qk_nope_head_dim`
  - `qk_rope_head_dim`
  - `v_head_dim`
  - expert-routing / MoE config fields for later

Why this is required:

- The model loader cannot run DeepSeek-V2-Lite until it is registered and understood.

6. Register a new DeepSeek-V2-Lite model implementation.

- Add a new model module under `sarathi-lean/sarathi/model_executor/models`
- Register it in the model loader
- Keep the first version attention-first if necessary

Why this is required:

- There is currently no DeepSeek-V2-Lite model class in the repo.

### Phase 3: Build a contiguous MLA reference path

7. Implement DeepSeek-V2-Lite MLA attention in a contiguous reference path.

- Start with a contiguous cache, not paged `vAttention`
- Represent the resident cache in component form:
  - `kv_latent`
  - `k_rope`
- Reconstruct dense K/V immediately before FlashAttention

Why this is required:

- This isolates model-attention correctness from paging/allocator behavior.

8. Implement the projection and reconstruction path for MLA attention.

- Add the attention projections needed for DeepSeek-V2-Lite
- Reconstruct:
  - dense key
  - dense value
  from resident cache components and current-step projections
- Keep all dense K/V reconstruction transient

Why this is required:

- This is the core MLA attention computation.

9. Add attention-only correctness tests.

- Compare local MLA attention against a reference implementation using:
  - identical weights
  - identical hidden states
  - identical positions
  - identical past cache
- Validate:
  - prefill
  - decode
  - multi-step cache reuse
  - mixed-length batches

Why this is required:

- This is the correctness gate before any MoE work.

### Phase 4: Get first working DeepSeek-V2-Lite inference without MoE

10. Add a temporary non-MoE inference path if needed.

- If MoE support is not yet implemented, add a temporary attention-first execution path or a minimal feedforward fallback sufficient for early inference bring-up.

Why this is required:

- The first practical milestone is getting the model stack to execute around the MLA attention path.

11. Reach first working contiguous DeepSeek-V2-Lite inference.

- Run a prompt through the local model path
- Confirm the model can:
  - prefill
  - decode at least one or more tokens

Why this is required:

- This is the first true “it runs” milestone.

### Phase 5: Add paged MLA support in the runtime path

12. Implement an MLA-specific attention wrapper for paged execution.

- Add a dedicated wrapper that:
  - reads resident componentized cache state from paged cache
  - reconstructs dense K/V right before FlashAttention
  - writes only resident MLA components back to cache

Why this is required:

- This is the bridge between paged `vAttention` cache and the MLA model path.

13. Wire the DeepSeek-V2-Lite model path to use the MLA wrapper.

- Ensure the model runner, attention backend selection, and cache engine work together for MLA execution.

Why this is required:

- Without backend integration, the model cannot use paged MLA cache during real runtime execution.

14. Reach first working paged DeepSeek-V2-Lite inference.

- Run a short prompt with paged MLA cache under `vAttention`
- Confirm:
  - cache initialization succeeds
  - prefill succeeds
  - decode succeeds

Why this is required:

- This is the first milestone where DeepSeek-V2-Lite runs in the actual intended architecture.

### Phase 6: Validate correctness and memory behavior

15. Compare paged MLA against contiguous MLA.

- Validate output parity between:
  - contiguous MLA path
  - paged MLA path

Why this is required:

- This is the main correctness check for the `vAttention` MLA integration.

16. Validate resident memory accounting.

- Confirm that:
  - allocator-visible resident bytes match MLA component payload
  - reconstructed dense K/V is not counted as persistent cache
  - fragmentation metrics reflect resident MLA state only

Why this is required:

- This is the core measurement requirement for the research goal.

17. Add allocator/fragmentation test cases for MLA.

- Sweep sequence lengths and validate:
  - expected page growth
  - expected tokens-per-page behavior
  - expected fragmentation accounting

Why this is required:

- This verifies that MLA memory behavior under `vAttention` is being measured correctly.

### Phase 7: Add full DeepSeek-V2-Lite support

18. Implement MoE support for DeepSeek-V2-Lite.

- Add expert routing
- add expert parameter loading
- add MoE execution path
- integrate with tensor/pipeline-parallel expectations in this repo

Why this is required:

- The full DeepSeek-V2-Lite model cannot run end-to-end without MoE.

19. Re-run inference validation with MoE enabled.

- Confirm that the previously validated MLA attention path still behaves correctly after MoE integration.

Why this is required:

- MoE integration should not silently perturb the MLA attention path.

20. Reach first full DeepSeek-V2-Lite inference.

- Run end-to-end inference through the actual DeepSeek-V2-Lite architecture in this repo.

Why this is required:

- This is the final milestone for “able to run inference with DeepSeek-V2-Lite.”

### Phase 8: Add telemetry and experiment support

21. Add asynchronous telemetry reporting after MLA allocator accounting is correct.

- Add sequence-length milestone reporting
- ensure telemetry reports resident MLA cache metrics, not transient dense-K/V buffers

Why this comes last:

- Telemetry is only useful once allocator accounting is trustworthy.

22. Run MLA experiments for the fragmentation study.

- Compare MHA, GQA, and MLA under the now-correct resident-cache accounting model.

Why this comes last:

- The research results depend on all prior correctness work.

## First Working Inference Milestones

The plan should be treated as a sequence of concrete bring-up milestones:

1. component-spec init works in the extension
2. contiguous MLA attention works
3. contiguous DeepSeek-V2-Lite executes
4. paged MLA attention works
5. paged DeepSeek-V2-Lite executes
6. full DeepSeek-V2-Lite with MoE executes

Only milestone 6 means we can truly say:

> We can run inference with DeepSeek-V2-Lite in this repo.

## Recommended Immediate Next Steps

From the current repository state, the highest-priority next tasks are:

1. add `init_kvcache_component_spec(...)` to the extension
2. refactor CUDA allocator sizing to use the shared resident-cache model
3. add allocator/CUDA integration tests
4. add DeepSeek-V2-Lite model/config registration
5. implement contiguous MLA attention

## Deliverables for V3

- a preserved record of the earlier plans
- this full path-to-inference plan
- a task list that covers both:
  - MLA allocator support
  - DeepSeek-V2-Lite runtime support
- a concrete set of milestones for reaching first working inference

## Notes

- V3 does not replace the earlier plans historically; it extends them operationally.
- The core addition in V3 is that it includes all missing runtime tasks, not just allocator and cache-layout tasks.
- The key practical takeaway is:
  - current work is necessary groundwork
  - but actual DeepSeek-V2-Lite inference still requires model, attention, wrapper, and MoE implementation work beyond the cache/allocator layer
