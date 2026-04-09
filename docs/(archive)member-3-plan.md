# Member 3 Plan

## Role

Mathematical Modeling and Empirical Validation

This role is responsible for formally characterizing the amortization behavior of KV-cache fragmentation and validating those predictions against the existing `vAttention` baselines before MLA is introduced.

## Primary Goals

- derive the theoretical amortization curves for MHA and GQA
- validate those curves empirically using the current `vAttention` stack
- identify baseline model candidates that `vAttention` already supports and that are realistic to run on the available hardware
- explain the architectural difference between MHA and GQA in terms of fragmentation behavior

## Work Plan

1. Define the exact fragmentation quantities.

- Write precise definitions for `W_avg` and `W_%`.
- State all assumptions clearly: block size, page size, number of KV heads, head dimension, layers, data type, and whether waste is measured in tokens, blocks, or bytes.
- Make sure the theoretical definitions match how the system computes fragmentation in practice.

2. Derive the MHA and GQA formulas.

- Express allocated KV memory and useful KV memory as functions of sequence length `L`.
- Derive closed-form or piecewise expressions for `W_avg(L)` and `W_%(L)` for MHA.
- Repeat the derivation for GQA, making the KV-head-count reduction explicit.
- Record the assumptions under which each derivation holds.

3. Identify amortization thresholds.

- Solve for the smallest sequence length where fragmentation falls below important targets like `10%`, `5%`, and `2%`.
- Produce a compact comparison table for MHA vs GQA.
- Highlight the difference in the number of tokens needed before fragmentation drops below `2%`.

4. Identify candidate baseline models.

- Find examples of both MHA and GQA models that the current `vAttention` codebase already supports or is likely to support with minimal setup.
- Document which candidates are the best fit for the available hardware, especially the `4 x RTX 3090` setup.
- Prefer models that give a fair architectural comparison between MHA and GQA without introducing unrelated confounders.
- Produce a short recommendation list with:
  - model name
  - architecture type: MHA or GQA
  - approximate size
  - expected fit on current hardware
  - reason it is a good comparison candidate

5. Run empirical validation on the existing system.

- Use the current `vAttention` codebase, without MLA integration, to collect initial fragmentation results for MHA and GQA baselines.
- Sweep over sequence length and any other parameters needed for the theoretical comparison.
- Confirm whether empirical fragmentation curves match the predicted amortization behavior.

6. Compare theory against measurement.

- Overlay the theoretical and empirical curves for MHA and GQA.
- Quantify the mismatch where it exists.
- Identify whether any deviations are caused by implementation details, scheduling effects, batching behavior, or allocator details.

7. Write the architectural delta.

- Produce a short explanation of why MHA and GQA differ in fragmentation amortization.
- Emphasize the difference in KV structure and how that affects the amount of useful memory per additional token.
- Summarize the practical consequence for the comparison section of the paper.

## Deliverables

- a theory note defining `W_avg` and `W_%`
- derivations for MHA and GQA amortization behavior
- a threshold comparison table, including the `< 2%` crossover point
- a recommended baseline model list for MHA and GQA on current hardware
- empirical baseline results from the current `vAttention` system
- a short writeup explaining the architectural difference between MHA and GQA

## Suggested First Milestones

- finalize the formal definitions of `W_avg` and `W_%`
- identify one strong MHA candidate and one strong GQA candidate for the cluster
- derive the first-pass amortization expressions
- run one initial baseline experiment for each architecture

## Notes

- The model-selection step is part of this role, not an afterthought.
- The goal is not just to prove the math in the abstract, but to connect it to realistic baselines that can run well on the available hardware and give a meaningful comparison.
