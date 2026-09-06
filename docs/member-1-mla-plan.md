# Member 1 Plan

## Role

MLA Architecture Integration and Experimental Bring-Up

This role is responsible for integrating Multi-Head Latent Attention (MLA) into the `vAttention` stack in a way that preserves the validity of the memory-fragmentation study. The implementation target is `DeepSeek-V2-Lite`, with FlashAttention retained as the attention kernel and dense K/V reconstructed immediately before the attention call.

## Primary Goals

- add `DeepSeek-V2-Lite` support to the serving stack
- implement MLA cache support in `vAttention`
- ensure MLA resident cache accounting is distinct from transient dense K/V reconstruction
- validate MLA attention correctness before adding MoE support
- produce an implementation that supports fragmentation experiments comparing MHA, GQA, and MLA

## Core Design Decision

We will keep FlashAttention and reconstruct dense K/V immediately before the attention call.

This is acceptable for the fragmentation study because:

- MHA and GQA will continue to page their true persistent dense K/V cache
- MLA will page its true persistent compressed cache payload
- transient reconstructed dense K/V buffers used only for the attention call will not be counted as KV-cache residency

This means the fragmentation comparison remains valid as long as all allocator sizing, paging, and telemetry are based on the MLA resident cache payload rather than the reconstructed dense K/V tensors.

## Key Engineering Constraint

The current `vAttention` code assumes that cached bytes per token are derived from dense K/V geometry:

- `num_kv_heads`
- `head_dim`
- `dtype`
- two sides: `K` and `V`

That assumption is embedded in:

- `sarathi-lean/sarathi/engine/arg_utils.py`
- `sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py`
- `vattention/vattention.cu`
- `vattention/utils.h`

For MLA, this assumption is incorrect. The persistent cache per token is not dense per-head K/V. It is the compressed MLA resident state, which should be treated as the canonical per-token cache payload throughout the allocator and telemetry stack.

## Work Plan

1. Introduce a cache-architecture abstraction.

- Add an explicit cache-architecture concept to the Python configuration path.
- Distinguish between:
  - `DENSE_KV` for MHA and GQA
  - `MLA` for DeepSeek-V2-Lite
- Add helpers to `ModelConfig` or a nearby cache-layout utility module, such as:
  - `is_mla_model()`
  - `get_cache_architecture()`
  - `get_cached_token_bytes_per_layer(parallel_config)`
  - `get_cached_token_bytes_local(parallel_config, megacache=False)`

2. Define a canonical `bytes per cached token` abstraction.

- Centralize all resident-cache sizing formulas in one place.
- For MHA and GQA:
  - cached bytes per token per layer = `2 * num_local_kv_heads * head_dim * dtype_size`
- For MLA:
  - cached bytes per token per layer = `mla_latent_dim_bytes + mla_rope_cache_bytes`
- These formulas must describe only persistent paged cache state.
- They must not include any temporary dense K/V tensors materialized just before FlashAttention.

3. Refactor every dense-KV sizing callsite to use the shared abstraction.

- Replace all inline dense-KV cache math in:
  - `sarathi-lean/sarathi/engine/arg_utils.py`
  - `sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py`
  - `vattention/vattention.cu`
  - `vattention/utils.h`
- Ensure that:
  - page capacity
  - block size
  - number of free blocks
  - cache block size
  - fragmentation telemetry
  all derive from the same cache-architecture-aware definition.

4. Keep Python and CUDA cache math synchronized.

- Ensure Python and C++ agree on the same MLA resident cache dimensions.
- Either:
  - pass the derived MLA cache dimensions from Python into the CUDA extension
  - or duplicate only a very small, stable formula set in both places
- Do not allow Python and CUDA to disagree on tokens per page or bytes per token.

5. Add `DeepSeek-V2-Lite` model support to the model executor.

- Implement a new model module for `DeepSeek-V2-Lite`.
- Register it in the model loader.
- Extend config handling to expose the MLA-specific dimensions needed by the cache and wrapper logic.
- Initial support should prioritize the attention path over the full MoE stack.

6. Implement the attention path before MoE.

- Add the `DeepSeek-V2-Lite` attention projections and MLA cache flow first.
- Do not start with expert routing or MoE execution.
- For the first milestone, use an attention-focused path with either:
  - a temporary dense feedforward substitute
  - or a minimal non-MoE fallback
- The point is to isolate MLA attention correctness from MoE complexity.

7. Build a contiguous MLA reference path first.

- Before touching `vAttention` paging, implement MLA using a simple contiguous cache in PyTorch.
- This reference path should:
  - store only MLA resident cache state
  - reconstruct dense K/V immediately before the FlashAttention call
  - produce correct outputs for both prefill and decode
- This stage isolates model math from allocator behavior.

8. Add an MLA-specific attention wrapper.

- Create a new wrapper for MLA rather than forcing the existing dense-KV wrapper to absorb all MLA logic.
- The wrapper should:
  - write only compressed MLA resident state to cache
  - reconstruct dense K/V right before the FlashAttention call
  - keep reconstructed dense K/V transient
  - never treat reconstructed dense K/V as resident paged cache

9. Validate attention correctness before proceeding to MoE.

- Use a standalone attention correctness gate before MoE support.
- Compare a reference DeepSeek-V2-Lite attention implementation against the local MLA attention path with identical:
  - weights
  - hidden states
  - positions
  - cache contents
- Validate:
  - prefill outputs
  - decode outputs
  - multi-step cache reuse
  - RoPE-sensitive positions
  - batched decode with differing context lengths

10. Define the MLA attention correctness test harness.

- Build tests that compare:
  - reference attention output vs local attention output
  - reference reconstructed K/V vs local reconstructed K/V
- Use deterministic inputs:
  - fixed random seed
  - dropout disabled
  - fixed positions
- Cover:
  - single-sequence prefill
  - single-sequence decode over multiple steps
  - mixed batch decode
  - cache append behavior after each step
- Require close numerical agreement before moving forward.

11. Add structural cache tests.

- Verify that MLA resident cache shapes reflect compressed cache state, not dense KV-head geometry.
- Verify that reconstructed dense K/V tensors appear only in the attention forward path.
- Verify that allocator-visible resident bytes do not increase to dense-KV scale.

12. Add a dedicated MLA backend in the attention/backend registry.

- Add an explicit MLA path rather than silently overloading the current dense-KV `vAttention` backend.
- Keep the architecture separation clear in:
  - backend selection
  - wrapper logic
  - cache engine behavior

13. Extend the CUDA API with MLA cache initialization.

- Add an MLA-specific initialization path to the `vAttention` extension.
- Allocate virtual tensors according to MLA resident payload layout instead of dense K/V layout.
- Keep dense-KV and MLA cache initialization paths separate enough to remain debuggable.

14. Update allocator paging logic for MLA resident bytes.

- Change all page-capacity and growth calculations in the CUDA allocator to use the MLA resident cache payload definition.
- Ensure the allocator answers the correct question:
  - how many MLA resident tokens fit per page
  - how many pages are needed for a sequence of length `L`
- Do not base MLA paging decisions on reconstructed dense K/V size.

15. Update fragmentation accounting for MLA semantics.

- Update useful-bytes and allocated-bytes calculations in the allocator telemetry path so that MLA fragmentation is measured using resident MLA payload bytes.
- This is necessary to keep MHA, GQA, and MLA fragmentation measurements directly comparable under `vAttention`.

16. Reuse the existing block manager if possible.

- Keep the existing block-manager behavior where practical.
- Let the refactored bytes-per-cached-token abstraction determine the meaning of a block for MLA.
- Avoid unnecessary scheduler changes if the same logical block interface can be retained.

17. Validate paged MLA against contiguous MLA.

- Once MLA paging is implemented, compare:
  - contiguous MLA outputs
  - paged `vAttention` MLA outputs
- Require output parity before moving to performance or telemetry analysis.

18. Run MLA-specific allocator and fragmentation tests.

- Sweep sequence length and verify:
  - page growth occurs at expected boundaries
  - free-block counts match expected capacity
  - fragmentation calculations reflect MLA resident payload
- Confirm that reconstructed dense K/V does not affect persistent cache accounting.

19. Add telemetry integration after MLA paging is correct.

- Only after allocator accounting is correct, add the asynchronous telemetry reporting path.
- Ensure telemetry reports:
  - resident MLA cache bytes
  - allocated physical bytes
  - fragmentation metrics
  - sequence-length milestones
- Do not report transient dense-K/V reconstruction buffers as cache residency.

20. Add `DeepSeek-V2-Lite` MoE support only after the attention gate passes.

- Once MLA attention is numerically validated and paged MLA is working, add MoE support.
- Re-run the attention parity tests afterward to confirm that MoE integration did not perturb the attention path.

## Attention Correctness Gate Before MoE

To mitigate the risk of debugging MLA attention and MoE at the same time, the project should enforce an explicit attention correctness gate before any MoE work proceeds.

The gate should require:

- a working contiguous MLA reference path
- a working MLA wrapper that reconstructs dense K/V only at the FlashAttention call
- parity against a reference DeepSeek-V2-Lite attention implementation for:
  - prefill
  - decode
  - multi-step cache reuse
  - RoPE-sensitive positions
  - mixed-length batched decode

Only after that gate passes should the MoE feedforward path be added.

## Recommended Execution Order

1. add the cache-architecture abstraction
2. centralize `bytes per cached token`
3. refactor Python and CUDA cache sizing to use the shared abstraction
4. add `DeepSeek-V2-Lite` model support
5. implement contiguous MLA attention
6. implement the MLA attention wrapper with reconstruct-before-FlashAttention
7. pass the attention correctness gate
8. add MLA cache initialization to `vAttention`
9. implement paged MLA resident cache support
10. validate paged MLA against contiguous MLA
11. validate MLA fragmentation accounting
12. add telemetry integration
13. add MoE support

## Deliverables

- a repo-integrated plan for MLA support in `vAttention`
- a cache-architecture-aware bytes-per-cached-token abstraction
- `DeepSeek-V2-Lite` attention-path support
- an MLA-specific attention wrapper
- a contiguous MLA correctness harness
- paged MLA support in `vAttention`
- fragmentation measurements for MLA under correct resident-cache accounting
- telemetry integration at sequence-length milestones
- a validated path to MoE integration after attention correctness is established

## Suggested First Milestones

- finalize the cache-architecture abstraction
- define the canonical resident-cache bytes-per-token formulas
- refactor allocator sizing and block sizing to use the new abstraction
- bring up contiguous MLA attention without MoE
- build and pass the MLA attention correctness gate

## Notes

- The key measurement rule is that only resident paged cache bytes count as KV-cache usage.
- Dense K/V reconstructed immediately before FlashAttention are transient compute buffers, not persistent cache state.
- If this distinction is preserved throughout the allocator and telemetry paths, the comparison among MHA, GQA, and MLA remains valid under `vAttention`.
