# Member 1 Plan V2

## Purpose

This document is a revised version of the original MLA implementation plan in [member-1-mla-plan.md](/home/anodyine/repos/vattention/docs/member-1-mla-plan.md).

The original plan is being kept unchanged for project recordkeeping.

This V2 plan reflects what we learned while implementing the first several refactors in the Python configuration and cache-sizing path. The broad strategy has not changed:

- keep FlashAttention
- reconstruct dense K/V immediately before the attention call
- treat only resident MLA cache state as persistent KV-cache memory

What changed is the level of precision in the implementation sequence and the interface boundaries.

## What We Learned

### 1. We need stronger abstraction boundaries than the original plan implied.

From the initial refactor work, it became clear that MLA integration is much safer if we separate three different concepts that were previously easy to blur together:

- model-attention dimensions
- resident cache structure
- allocator/init sizing

This led to the introduction of distinct spec layers:

- `MLAAttentionSpec`
- `CacheComponentSpec`
- `CacheLayout`
- `VAttentionCacheSpec`
- `VAttentionInitSpec`

Why this changes the plan:

- The original plan said to centralize cache sizing, which was correct.
- What we learned is that “centralize cache sizing” is not enough by itself.
- We also need to avoid coupling DeepSeek-specific attention fields directly to allocator math.
- So V2 explicitly stages the work through those intermediate spec objects.

### 2. Resident cache structure must be described structurally, not just as a byte count.

We initially focused on `bytes per cached token`, which was necessary.

But during implementation, it became obvious that byte counts alone are not enough for MLA. We also need a stable structural description of what is resident in cache:

- Dense KV: `k`, `v`
- MLA: `kv_latent`, `k_rope`

Why this changes the plan:

- The original plan described MLA resident payload conceptually, but not as an explicit API surface.
- V2 makes resident cache components a first-class step because that will directly shape both:
  - MLA cache tensor layout
  - MLA attention-wrapper inputs

### 3. Python/CUDA synchronization should happen through a spec object, not through duplicated formulas.

We now have enough evidence that the safest next CUDA step is not “add some MLA branches” but:

- define the Python-side allocator/init spec cleanly
- make the CUDA side consume that same structure or a direct serialization of it

Why this changes the plan:

- The original plan allowed either passing dimensions from Python or duplicating formulas in C++.
- After the refactor work, the Python side is already organizing the relevant data cleanly.
- So V2 biases more strongly toward spec-driven synchronization instead of parallel formula maintenance.

### 4. Test layering needs to be more explicit.

We learned that:

- package-level `sarathi` imports are heavy and pull in unrelated runtime paths
- direct module-focused unit tests are much easier to keep stable
- Docker-based execution is the correct primary validation environment

Why this changes the plan:

- The original plan called for validation, but it did not clearly separate:
  - config/cache-layout unit tests
  - allocator/CUDA integration tests
  - MLA attention parity tests
- V2 makes those layers explicit so failures are easier to localize.

### 5. The future MLA wrapper should be designed around resident cache components.

The original plan correctly said that reconstructed dense K/V should stay transient.

What we learned is that the wrapper should be described more concretely as:

- read resident components from cache
- reconstruct dense attention inputs from those components
- call FlashAttention

Why this changes the plan:

- This is the cleanest bridge from the new component-level cache abstraction to the model-execution path.
- It also gives the attention correctness harness clearer test targets.

## Updated Design Principles

The implementation should now follow these explicit principles:

1. Keep attention-dimension logic separate from cache-allocation logic.
2. Treat resident cache structure as a first-class API surface.
3. Derive allocator math from shared spec objects, not ad hoc formulas.
4. Keep reconstructed dense K/V transient and out of persistent memory accounting.
5. Stage testing from pure-Python layout logic to allocator integration to attention parity.

## Work Plan V2

1. Finish the Python-side cache specification layer.

- Keep the new spec hierarchy as the source of truth:
  - `MLAAttentionSpec`
  - `CacheComponentSpec`
  - `CacheLayout`
  - `VAttentionCacheSpec`
  - `VAttentionInitSpec`
- Ensure all Python-side sizing and initialization paths derive from these objects.

Why this is earlier and more explicit than before:

- We now know this structure is the cleanest way to prevent dense-KV assumptions from leaking into MLA.

2. Complete the remaining dense-KV sizing callsite migration in Python.

- Audit Python paths for any remaining dense-KV inline assumptions that should instead consume the shared spec objects.
- Keep these changes small and test-backed.

Why this remains in scope:

- Before touching CUDA, the Python side should present one coherent definition of resident cache structure and allocator sizing.

3. Add a serialization-friendly boundary for the CUDA allocator.

- Define exactly what part of `VAttentionInitSpec` or `VAttentionCacheSpec` will be passed into the extension.
- Prefer a spec-driven handoff over duplicating MLA formulas in CUDA code.

Why this changed:

- The original plan left the Python/CUDA sync choice more open.
- Based on what we built, V2 favors using the spec layer directly.

4. Update the CUDA allocator to consume MLA-aware cache/init data.

- Refactor allocator sizing logic in:
  - `vattention/vattention.cu`
  - `vattention/utils.h`
- Replace dense-KV assumptions for:
  - `tokens_per_page`
  - virtual buffer size per token
  - page growth calculations
  - fragmentation accounting
- Make those values correspond to resident cache structure, not reconstructed dense K/V.

Why this is framed differently:

- V1 described this mainly as formula replacement.
- V2 frames it as “make CUDA consume the shared resident-cache model.”

5. Introduce an MLA-specific cache initialization path in the extension.

- Keep dense-KV and MLA initialization paths conceptually separate.
- Use the shared spec to determine:
  - resident token payload
  - tensor layout
  - page capacity semantics

Why this is more explicit:

- We now understand that tensor shape and token-payload structure need to follow the component-level cache model, not just byte counts.

6. Add `DeepSeek-V2-Lite` config/model support with attention-first scope.

- Add the model registration and config support needed for DeepSeek-V2-Lite.
- Do not add MoE execution yet.
- Start with the MLA attention path and a minimal feedforward fallback if needed.

Why this remains the same:

- The earlier reasoning still holds: attention correctness must be isolated from MoE complexity.

7. Build the MLA wrapper around resident cache components.

- Design the wrapper to work from componentized cache state:
  - `kv_latent`
  - `k_rope`
- Reconstruct dense K/V immediately before FlashAttention.
- Ensure dense K/V tensors remain transient.

Why this changed:

- V1 said this conceptually.
- V2 now ties it directly to the `CacheComponentSpec` abstraction.

8. Build the contiguous MLA reference path before paged MLA execution.

- Implement a non-paged MLA reference path in PyTorch first.
- Use the same component-based resident cache representation as much as possible.
- Validate:
  - prefill
  - decode
  - multi-step cache reuse

Why this remains unchanged:

- It is still the best way to isolate model math from allocator behavior.

9. Split validation into three explicit layers.

### Layer A: config and cache-layout unit tests

- Validate:
  - architecture detection
  - MLA attention dimensions
  - resident cache components
  - tokens per page
  - cache block size
  - allocator init spec

### Layer B: allocator/CUDA integration tests

- Validate:
  - page growth boundaries
  - free-block accounting
  - resident-byte accounting
  - consistency between Python-derived spec and CUDA behavior

### Layer C: MLA attention parity tests

- Validate:
  - reference attention output vs local attention output
  - reference reconstructed K/V vs local reconstructed K/V
  - prefill
  - decode
  - mixed-length batches

Why this changed:

- We now know test layering needs to be explicit to keep failures debuggable.

10. Enforce the MLA attention correctness gate before MoE.

- Do not proceed to MoE until:
  - contiguous MLA attention is correct
  - paged MLA matches the contiguous reference
  - reconstructed dense K/V is confirmed transient

Why this remains unchanged:

- The risk of debugging MoE and MLA attention simultaneously is still high.

11. Add MoE support only after the gate passes.

- Once attention and cache behavior are correct, integrate the DeepSeek-V2-Lite MoE path.
- Re-run attention parity and cache-accounting tests after MoE integration.

Why this remains unchanged:

- It is still the right sequencing decision.

12. Add telemetry after allocator accounting is correct.

- Once MLA resident bytes and fragmentation are correct under paging, integrate:
  - milestone-based telemetry
  - fragmentation reporting
  - sequence-length event reporting

Why this remains unchanged:

- Telemetry is only useful once allocator accounting is trustworthy.

## Revised Near-Term Execution Order

1. finish Python-side cache spec consolidation
2. complete remaining Python callsite migration
3. define the Python-to-CUDA spec boundary
4. refactor CUDA allocator sizing around the shared spec
5. add MLA-aware cache initialization in the extension
6. add DeepSeek-V2-Lite model/config support
7. build contiguous MLA reference execution
8. implement MLA wrapper using resident cache components
9. validate paged MLA against the contiguous reference
10. add MoE support
11. add telemetry

## Current Status Relative to V2

Completed so far:

- cache architecture abstraction
- canonical cached-token byte helpers
- page-capacity helpers
- centralized cache block sizing
- shared cache layout descriptor
- shared vAttention cache spec
- shared vAttention init spec
- explicit MLA attention spec
- explicit resident cache component spec
- Docker-validated unit-test harness for the config/cache layer

Not yet done:

- CUDA allocator refactor to consume the shared spec
- MLA-aware extension init path
- DeepSeek-V2-Lite model implementation
- contiguous MLA reference path
- MLA wrapper
- attention parity tests
- MoE support
- telemetry integration

## Deliverables for V2

- the preserved original plan
- this revised plan with recorded lessons learned
- a spec-driven Python cache-sizing layer
- a staged path to CUDA integration and MLA model bring-up
- a clearer test strategy that separates layout logic, allocator behavior, and attention correctness

## Notes

- V2 does not replace the original plan historically; it refines it operationally.
- The central lesson so far is that MLA integration needs explicit representations for:
  - attention dimensions
  - resident cache structure
  - allocator/init sizing
- That separation is now the main design constraint guiding the remaining work.
