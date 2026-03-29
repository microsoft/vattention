# Josh Metrics Plan: Fragmentation vs Context Length

This plan explains how to capture vAttention fragmentation metrics in Sarathi so you can compare fragmentation against request context length across many requests.

## Goal

Measure fragmentation as a function of context length by:

- sending requests with different prompt lengths
- generating only a few decode tokens per request (to keep MLA decode cost low)
- saving per-request fragmentation metrics in Sarathi's existing metrics outputs

The target result is a request-level dataset where each request row includes:

- context length (`request_num_prefill_tokens`)
- fragmentation metrics (for example `kv_blocks_mapped`, `kv_fragmentation_percent`)

## Codebase background you need first

### 1. vAttention allocator and fragmentation source

- [vattention/vattention.cu](/home/anodyine/repos/vattention/vattention/vattention.cu)
- [vattention/apis.h](/home/anodyine/repos/vattention/vattention/apis.h)

What matters:

- Fragmentation math is implemented in `compute_fragmentation_metrics(...)`.
- Python can already call `debug_fragmentation_metrics(seq_len, mapped_blocks)`.
- This means you do **not** need a new C++ binding for first-pass integration.

### 2. Cache engine runtime state (where seq lengths live)

- [vATTN_cache_engine.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py)

What matters:

- `curr_seq_lens` tracks current sequence lengths by batch index.
- `seq_to_batch_idx` maps request `seq_id` to allocator batch index.
- `num_free_blocks()` already exposes allocator free blocks.
- This is the best place to build a Python helper that returns allocator metrics per active request.

### 3. Where request metrics are written

- [metrics_store.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/metrics/metrics_store.py)
- [constants.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/metrics/constants.py)

What matters:

- `sequence_metrics.csv` is keyed by `Request Id` and is built from per-request `DataSeries`.
- Request-level fields like `request_num_prefill_tokens` are already recorded in `_on_request_end(...)`.
- If you add fragmentation series keyed by `Request Id`, they can be emitted into the same CSV.

### 4. Worker and engine hooks

- [base_worker.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/worker/base_worker.py)
- [base_llm_engine.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/engine/base_llm_engine.py)

What matters:

- Requests are finalized in the normal batch/step flow.
- Metrics should be captured at a deterministic point in that flow (same place request metrics are already finalized) so request IDs and final lengths are stable.

### 5. Docker output path

- [scripts/docker/start-server.sh](/home/anodyine/repos/vattention/scripts/docker/start-server.sh)

What matters:

- Docker already passes `--output_dir` into the in-container server process.
- Sarathi metrics write to that output dir.
- Ray exporter warnings are separate and should not block this lab.

## Implementation steps (what to change and why)

1. Add allocator metric names in `constants.py`.

- Add names for at least:
  - `KV_BLOCKS_MAPPED`
  - `KV_FRAGMENTATION_PERCENT`
- Why: metric constants prevent string drift and keep plots/CSV naming consistent.

2. Add a request-level allocator snapshot helper in `vATTN_cache_engine.py`.

- Add a method (for example `get_request_allocator_metrics(seq_id)` or `get_allocator_metrics_for_requests(seq_ids)`) that:
  - resolves `seq_id -> batch_idx` using `seq_to_batch_idx`
  - reads current sequence length from `curr_seq_lens[batch_idx]`
  - computes mapped blocks from allocator state for that request
  - calls `vattention.debug_fragmentation_metrics(seq_len, mapped_blocks)`
  - returns structured values
- Why: the cache engine already owns allocator-facing state and is the correct layer to translate runtime request state into metrics.

3. Add storage for request-level allocator metrics in `MetricsStore`.

- Create new `DataSeries` entries keyed by `Request Id` (same key used by sequence metrics).
- Add a method like `push_request_allocator_metric(metric_name, request_id, value)`.
- Why: request-keyed storage is required to compare against `request_num_prefill_tokens` per request.

4. Hook allocator metric capture at request-finalization time.

- In the request completion path (where `_on_request_end(...)` is effectively finalized), push fragmentation fields for that request ID.
- Ensure request ID uses the same format as existing sequence metrics (`replica_id + seq_id` via `_get_seq_id`).
- Why: this guarantees one aligned row per completed request.

5. Emit allocator request metrics into `sequence_metrics.csv`.

- Include new request-level allocator `DataSeries` in `_store_seq_metrics(...)` so output lands with existing request columns.
- Optional: also keep a separate `allocator_metrics.csv` if you want batch-level debugging.
- Why: the experiment needs one table that already contains both context length and fragmentation.

6. Keep current allocator `printf` temporarily.

- Do not remove existing stdout fragmentation prints during rollout.
- Why: use stdout to spot-check values while validating CSV integration.

## Validation procedure (Docker)

1. Start server:

- `scripts/docker/start-server-yi6b.sh`

2. Send requests across a context-length sweep:

- vary prompt length significantly
- keep generation short (`max_tokens` small, e.g. 1-4)

3. Verify output files under `/tmp/vattention/<container-name>`:

- `sequence_metrics.csv` exists
- new fragmentation columns are present

4. Verify row-level alignment:

- for each `Request Id`, check `request_num_prefill_tokens` and `kv_fragmentation_percent` are both populated

5. Spot-check against stdout:

- compare a few requests against allocator print values to ensure parity

## First milestone (minimum useful result)

Deliver this first:

- add request-level `KV_BLOCKS_MAPPED` and `KV_FRAGMENTATION_PERCENT`
- write both into `sequence_metrics.csv` keyed by `Request Id`
- validate on a short Docker run with mixed context lengths

This is enough to produce the fragmentation-vs-context curve for MLA-focused experiments.

## What not to block on

- Ray internal metrics exporter warnings
- Ray dashboard/Prometheus integration
- adding every possible allocator field before first analysis

Those can be addressed later after request-level fragmentation capture is stable.
