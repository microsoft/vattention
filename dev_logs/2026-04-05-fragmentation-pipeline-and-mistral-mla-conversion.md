# Dev Log: 2026-04-05

## Scope

This log covers:

- the merged `adding-client-request-loop` work that landed on `main` today via PR #9
- the follow-up experimentation work on this branch: `mistral-gqa-to-mla-conversion`
- the current experimental story for the report
- concrete commands, artifacts, and caveats needed to restart in a fresh chat tomorrow

Branch context at end of day:

- `main` includes merge commit `04cc61d` (`Merge pull request #9 from Anodyine/adding-client-request-loop`)
- this branch includes follow-up commit `3a36006` (`added 4 models graph`)

Working tree status at time of writing:

- clean

## High-Level Outcome

By the end of today we had:

- a working client-side request sweep that drives exact prompt lengths against the OpenAI-compatible server
- a master pipeline that starts a model server, waits for readiness, runs the sweep, shuts the server down gracefully, and plots fragmentation
- host-visible metrics under `server-output/<model-slug>/`
- repo-visible plots under `server_plots/<model-slug>/`
- four experimental model tracks:
  - `Qwen/Qwen-14B` as MHA
  - `mistralai/Mistral-Nemo-Base-2407` as GQA
  - `deepseek-ai/DeepSeek-V2-Lite` as real MLA
  - `mistralai/Mistral-Nemo-Base-2407` converted into a synthetic MLA runtime path for fragmentation-only study
- a four-model allocated-cache comparison plot that makes the architectural story much clearer

The most important result is:

- DeepSeek-V2-Lite did **not** support the simple hypothesis that MLA always uses less cache than GQA in our current implementation.
- The synthetic Mistral GQA->MLA conversion **did** support the intended hypothesis in a controlled apples-to-apples comparison on the same backbone.

This means the report story should distinguish:

1. a general allocator/fragmentation story
2. a current implementation result for DeepSeek MLA
3. a controlled synthetic GQA->MLA conversion result showing the expected cache-layout advantage

## Git / Commit Context

Recent relevant commits:

- `1461f9a` `added request sweep script`
- `636ad74` `added client request sweep`
- `6244a77` `added basic plotting`
- `bd84559` `added MLA vs MHA comparison`
- `db51bb7` `generated some graphs`
- `04cc61d` merge of PR #9 `adding-client-request-loop`
- `3a36006` `added 4 models graph`

PR #9 (`04cc61d`) added the core experimentation framework:

- `scripts/fragmentation_context_sweep.py`
- `scripts/run_fragmentation_pipeline.sh`
- `scripts/plotting/plot_context_vs_fragmentation.py`
- `scripts/plotting/plot_cache_bytes_comparison.py`
- server wrappers for Qwen, Llama-3-8B, Mistral-Nemo-12B, DeepSeek-V2-Lite
- the dedicated venv setup scripts for the sweep runner and plotting
- OpenAI-serving changes to allow token-array prompts so the client can control context length exactly
- tests for the sweep script

This branch (`3a36006`) added the synthetic Mistral MLA conversion and the four-model comparison plot:

- `sarathi-lean/sarathi/model_executor/models/mistral_mla.py`
- `sarathi-lean/tests/test_mistral_mla_conversion.py`
- `scripts/docker/start-server-mistral-nemo-12b-mla.sh`
- pipeline support for `--model-key mistral-nemo-12b-mla`
- multi-series cache-bytes comparison plotting

## What Was Implemented Today

### 1. Fragmentation sweep and orchestration flow

The client request loop now works end to end:

- one request at a time
- exact prompt token lengths
- `max_tokens=1`
- manifest written for every run
- lengths automatically clamped to the model server's advertised `max_model_len`

Main files:

- `/home/anodyine/repos/vattention/scripts/fragmentation_context_sweep.py`
- `/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh`

Important behavioral details:

- the sweep queries `/v1/models` before running
- it drops requested context lengths above the server's max context
- the pipeline starts the containerized server, waits for readiness, runs the sweep, sends `SIGINT`, waits for `sequence_metrics.csv`, then waits a short settle window before plotting

### 2. Dedicated experiment environments

Two task-specific venvs were created:

- sweep runner env: `/home/anodyine/repos/vattention/.venv-frag-sweep`
- plotting env: `/home/anodyine/repos/vattention/.venv-londy`

Helpers:

- `/home/anodyine/repos/vattention/scripts/setup-fragmentation-context-sweep-venv.sh`
- `/home/anodyine/repos/vattention/scripts/plotting/setup-context-fragmentation-venv.sh`

### 3. Host-visible output directories

Server wrappers were updated so metrics land in host-visible repo paths:

- `/home/anodyine/repos/vattention/server-output/qwen-14b`
- `/home/anodyine/repos/vattention/server-output/mistral-nemo-12b`
- `/home/anodyine/repos/vattention/server-output/mistral-nemo-12b-mla`
- `/home/anodyine/repos/vattention/server-output/deepseek-v2-lite`

Plots are written separately under:

- `/home/anodyine/repos/vattention/server_plots/...`

This separation matters because the container writes `server-output/...` as `nobody`, so those files are readable from the host but not always writable by the host user. `server_plots/...` is the user-owned place for derived figures.

### 4. Model selection outcome

By end of day the practically relevant models were:

- MHA baseline: `Qwen/Qwen-14B`
- GQA baseline: `mistralai/Mistral-Nemo-Base-2407`
- real MLA baseline: `deepseek-ai/DeepSeek-V2-Lite`
- synthetic MLA conversion: `mistralai/Mistral-Nemo-Base-2407` through `MistralMLAForCausalLM`

Llama-3-8B was initially considered for GQA but proved less useful:

- gated on Hugging Face
- smaller than desired
- eventually replaced by Mistral-Nemo-12B as the better GQA baseline

### 5. Mistral-Nemo support fixes

Mistral-Nemo required runtime fixes before it would load and serve:

- the repo had assumed `head_dim = hidden_size / num_attention_heads`, which is false for Mistral-Nemo
- support was added for `config.head_dim`
- Mistral checkpoint naming had to be normalized across legacy and HF-style names

Relevant files:

- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/config.py`
- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/model_executor/models/mistral.py`

Without those fixes, the server failed at model load or reshape time.

### 6. Synthetic GQA->MLA conversion for Mistral-Nemo

This branch added a quality-agnostic synthetic MLA conversion path for Mistral-Nemo:

- file: `/home/anodyine/repos/vattention/sarathi-lean/sarathi/model_executor/models/mistral_mla.py`

Design:

- subclasses the existing DeepSeek MLA model path
- consumes Mistral source weights
- rewrites them into MLA-shaped weights needed by the runtime
- preserves enough shape correctness to run forward passes and exercise the MLA cache path
- does **not** aim to preserve model quality

This is acceptable for the current goal because we only care about allocator behavior and fragmentation, not output quality.

The conversion is activated through env vars in:

- `/home/anodyine/repos/vattention/scripts/docker/start-server-mistral-nemo-12b-mla.sh`

and the model loader hook in:

- `/home/anodyine/repos/vattention/sarathi-lean/sarathi/model_executor/model_loader.py`

Current synthetic MLA dimensions:

- `kv_lora_rank = 128`
- `qk_rope_head_dim = 64`
- `qk_nope_head_dim = 64`
- `v_head_dim = 128`

These were chosen specifically to make the resident MLA cache smaller than the dense GQA resident representation for the same Mistral backbone.

## Important Experimental Findings

### A. Qwen-14B MHA vs DeepSeek-V2-Lite MLA

This comparison showed a clear sawtooth difference when both were clipped to `8192` tokens:

- same qualitative sawtooth mechanism
- DeepSeek and Qwen both exhibit diminishing fragmentation within pages and jumps on page allocation
- the matched-axis comparison was visually strong

However, the raw block-count and allocated-cache comparisons were not enough by themselves to claim that DeepSeek MLA is smaller than all dense baselines, because page capacity differs by architecture.

### B. Why DeepSeek looked worse than expected against Mistral

This was the key surprise of the day.

The DeepSeek vs Mistral comparison showed that DeepSeek allocated **more** cache bytes than Mistral over the same context range.

This is not a plotting bug. It comes directly from the current implementation's resident cache geometry.

From the server logs:

- Qwen-14B:
  - Architecture: `dense_kv`
  - Tokens Per Page: `819`
  - Page Buffer Token Bytes: `2560`

- Mistral-Nemo-12B:
  - Architecture: `dense_kv`
  - Tokens Per Page: `4096`
  - Page Buffer Token Bytes: `512`

- DeepSeek-V2-Lite:
  - Architecture: `mla`
  - Tokens Per Page: `1820`
  - Page Buffer Token Bytes: `1152`

- Mistral-Nemo-12B (Synthetic MLA):
  - Architecture: `mla`
  - Tokens Per Page: `5461`
  - Page Buffer Token Bytes: `384`

Interpretation:

- in our current implementation, DeepSeek MLA stores a resident per-token state that is larger than Mistral GQA's local dense-KV page-buffer state
- specifically, DeepSeek stores `kv_latent + k_rope`
- this makes its page-buffer footprint `1152` bytes per token, versus Mistral's `512`
- because page size is fixed, fewer tokens fit per page for DeepSeek
- that makes DeepSeek's curve bumpier and increases allocated cache bytes

This is why the report must **not** claim:

- "MLA always uses less cache than GQA"

based on the real DeepSeek comparison.

### C. k_rope investigation

We explicitly revisited whether `k_rope` must be stored.

Conclusion:

- in the current DeepSeek MLA implementation, `k_rope` is part of the resident key state and is needed to reconstruct the key used at attention time
- dropping it would break exact behavior unless another representation were retained to recompute it
- it is therefore not simply "extra waste" that can be deleted without changing the algorithm

So the DeepSeek result is partly a deeper MLA representation issue, not just a vAttention paging mistake.

### D. Controlled GQA->MLA conversion result

The synthetic Mistral-Nemo GQA->MLA conversion produced the result we wanted for a controlled apples-to-apples architectural comparison:

- same backbone
- same model family
- GQA and MLA compared under the same serving stack
- synthetic MLA layout allocates less cache than GQA

This is the cleanest support for the intended hypothesis.

## Most Important Report-Ready Numbers

From:

- `/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context_summary.csv`

### Max allocated cache over the measured range

- Qwen-14B (MHA), max context `8192`: `22 MiB`
- Mistral-Nemo-12B (GQA), max context `32768`: `16 MiB`
- Mistral-Nemo-12B (Synthetic MLA), max context `32768`: `14 MiB`
- DeepSeek-V2-Lite (MLA), max context `32768`: `38 MiB`

### Mean estimated waste over the measured range

- Qwen-14B (MHA): `1.1004 MiB`
- Mistral-Nemo-12B (GQA): `1.1708 MiB`
- Mistral-Nemo-12B (Synthetic MLA): `1.0778 MiB`
- DeepSeek-V2-Lite (MLA): `1.0243 MiB`

Important interpretation:

- the waste MiB numbers are not the main story for the comparison figure anymore
- the main comparison figure now focuses only on allocated cache vs context length
- the fragmentation plots already communicate the sawtooth and waste patterns better

### Fragmentation summary snapshots

From the saved summary CSVs:

- Qwen-14B at 8192:
  - mean fragmentation `21.48%`
  - median `15.34%`
  - p90 `46.08%`

- DeepSeek-V2-Lite at 8192:
  - mean fragmentation `32.18%`
  - median `23.81%`
  - p90 `72.57%`

- Mistral-Nemo-12B at 32768:
  - mean fragmentation `25.03%`
  - median `16.67%`
  - p90 `53.75%`

- Mistral-Nemo-12B (Synthetic MLA) at 32768:
  - mean fragmentation `27.48%`
  - median `18.75%`
  - p90 `65.31%`

These fragmentation summaries are useful context, but the stronger storyline comes from pairing them with the allocated-cache comparison.

## Figures and Artifacts Worth Keeping for the Report

### Main per-model fragmentation figures

- Qwen MHA:
  - `/home/anodyine/repos/vattention/server_plots/qwen-14b-8192-max-context/context_vs_fragmentation.png`

- DeepSeek MLA at 8192:
  - `/home/anodyine/repos/vattention/server_plots/deepseek-v2-lite-8192-max-context/context_vs_fragmentation.png`

- Mistral GQA:
  - `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/context_vs_fragmentation.png`

- Mistral Synthetic MLA:
  - `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b-mla/context_vs_fragmentation.png`

### Main comparison figure

- four-model allocated cache figure:
  - `/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context.png`

### Supporting comparison figure

- earlier DeepSeek-vs-Mistral cache figure:
  - `/home/anodyine/repos/vattention/server_plots/comparisons/deepseek-vs-mistral/cache_bytes_vs_context.png`

This older two-model plot is still useful as backup evidence for explaining why DeepSeek behaved differently than expected.

## Current Interpretation for the Report

Recommended narrative:

1. show fragmentation-vs-context sawtooth plots to illustrate allocator behavior
2. use Qwen-14B vs DeepSeek-V2-Lite for the MHA-vs-MLA visual comparison at a shared `8192` limit
3. be explicit that DeepSeek-V2-Lite's current MLA runtime representation has larger page-buffer bytes per token than Mistral-Nemo GQA in this codebase
4. use the synthetic Mistral GQA->MLA conversion to show the cleaner controlled architectural comparison
5. use the four-model allocated-cache figure to place all runs on one chart

Suggested careful claim:

- "Our allocator experiments show the expected sawtooth fragmentation dynamics across dense KV and MLA layouts. In the current DeepSeek-V2-Lite implementation, MLA does not automatically reduce allocated cache relative to the Mistral-Nemo GQA baseline because the resident MLA cache representation has a larger page-buffer footprint. However, when the same Mistral-Nemo backbone is converted into a synthetic MLA layout, the allocated cache drops relative to the original GQA layout, supporting the intended architectural hypothesis under a controlled comparison."

What not to claim:

- "DeepSeek-V2-Lite proves MLA is always smaller than GQA"
- "MLA universally reduces bytes/token in our current runtime"

## Commands We Actually Used / Need Tomorrow

### Run the pipeline for a model

Qwen:

```bash
/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh \
  --model-key qwen-14b
```

Mistral GQA:

```bash
/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh \
  --model-key mistral-nemo-12b
```

Mistral Synthetic MLA:

```bash
/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh \
  --model-key mistral-nemo-12b-mla
```

DeepSeek:

```bash
/home/anodyine/repos/vattention/scripts/run_fragmentation_pipeline.sh \
  --model-key deepseek-v2-lite
```

### Run the four-model comparison plot

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/anodyine/repos/vattention/.venv-londy/bin/python \
/home/anodyine/repos/vattention/scripts/plotting/plot_cache_bytes_comparison.py \
  --series 'server-output/qwen-14b/sequence_metrics.csv|server-output/qwen-14b/benchmark_config.yml|Qwen-14B (MHA)|#d73a49' \
  --series 'server-output/mistral-nemo-12b/sequence_metrics.csv|server-output/mistral-nemo-12b/benchmark_config.yml|Mistral-Nemo-12B (GQA)|#1f6feb' \
  --series 'server-output/mistral-nemo-12b-mla/sequence_metrics.csv|server-output/mistral-nemo-12b-mla/benchmark_config.yml|Mistral-Nemo-12B (Synthetic MLA)|#2da44e' \
  --series 'server-output/deepseek-v2-lite/sequence_metrics.csv|server-output/deepseek-v2-lite/benchmark_config.yml|DeepSeek-V2-Lite (MLA)|#8250df' \
  --out-plot '/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context.png' \
  --out-summary '/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context_summary.csv' \
  --title 'Allocated Cache vs Context Length Across MHA, GQA, and MLA Runs'
```

## Verification Status

Verified today:

- `test_fragmentation_context_sweep.py` passed earlier in the day
- `test_mistral_mla_conversion.py` passes in-container
- the Mistral synthetic MLA server now starts and runs the pipeline successfully
- the four-model allocated-cache plot was regenerated after simplifying it to remove the waste panel

Also fixed along the way:

- wrapper execute bit issue on `start-server-mistral-nemo-12b-mla.sh`
- invalid `--tokenizer` argument in that wrapper
- pipeline stale-server issue by making the readiness check verify the served model name

## Known Caveats

- The synthetic Mistral MLA conversion is **not** a quality-preserving converted model.
- It should be described as a cache-layout / fragmentation experiment, not a pretrained MLA model.
- `server-output/...` contents may be owned by `nobody` because the server writes from inside the container.
- `server_plots/...` is the right place for host-generated report figures.
- DeepSeek's behavior is real for this runtime; do not smooth it away or reinterpret it as a plotting artifact.

## Best Next Steps Tomorrow

1. Decide which figures are the primary ones for the report.
   - likely one MHA-vs-MLA fragmentation comparison
   - one GQA-vs-synthetic-MLA fragmentation comparison
   - one four-model allocated-cache figure

2. Write caption text while the interpretation is fresh.

3. If needed, add a clipped shared-range comparison figure at `8192` for all models.

4. If needed, add another plotter that overlays only:
   - Mistral GQA
   - Mistral Synthetic MLA
   so the controlled same-backbone comparison is front and center.

5. Consider whether to save a short markdown note translating the main figure takeaways directly into report prose.

## Fresh-Chat Restart Notes

If starting from scratch tomorrow, the most important facts to tell the new chat are:

- PR #9 already merged the sweep + pipeline + plotting system.
- The branch `mistral-gqa-to-mla-conversion` adds a synthetic MLA conversion for Mistral-Nemo-12B.
- The synthetic conversion is meant only for fragmentation/cache experiments, not quality.
- The key new model key is:
  - `mistral-nemo-12b-mla`
- The most important artifacts are:
  - `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b/context_vs_fragmentation.png`
  - `/home/anodyine/repos/vattention/server_plots/mistral-nemo-12b-mla/context_vs_fragmentation.png`
  - `/home/anodyine/repos/vattention/server_plots/qwen-14b-8192-max-context/context_vs_fragmentation.png`
  - `/home/anodyine/repos/vattention/server_plots/deepseek-v2-lite-8192-max-context/context_vs_fragmentation.png`
  - `/home/anodyine/repos/vattention/server_plots/comparisons/four-models/cache_bytes_vs_context.png`
- The key implementation insight is:
  - DeepSeek-V2-Lite uses `Page Buffer Token Bytes = 1152`
  - Mistral GQA uses `512`
  - synthetic Mistral MLA uses `384`
  - therefore DeepSeek being larger than Mistral is a real current-runtime result, while the synthetic GQA->MLA conversion shows the intended architectural cache savings.
