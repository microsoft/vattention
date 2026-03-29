# Metrics Plan: Fragmentation vs Context Length

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

- [vattention/vattention.cu](../vattention/vattention.cu#L509)
- [vattention/apis.h](../vattention/apis.h#L64)

What matters:

- Fragmentation math is implemented in `compute_fragmentation_metrics(...)`.
- Python can already call `debug_fragmentation_metrics(seq_len, mapped_blocks)`.
- This means you do **not** need a new C++ binding for first-pass integration.

### 2. Cache engine runtime state (where seq lengths live)

- [vATTN_cache_engine.py](../sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L753)

What matters:

- `curr_seq_lens` tracks current sequence lengths by batch index.
- `seq_to_batch_idx` maps request `seq_id` to allocator batch index.
- `num_free_blocks()` already exposes allocator free blocks.
- This is the best place to build a Python helper that returns allocator metrics per active request.

### 3. Where request metrics are written

- [metrics_store.py](../sarathi-lean/sarathi/metrics/metrics_store.py#L155)
- [constants.py](../sarathi-lean/sarathi/metrics/constants.py#L79)

What matters:

- `sequence_metrics.csv` is keyed by `Request Id` and is built from per-request `DataSeries`.
- Request-level fields like `request_num_prefill_tokens` are already recorded in `_on_request_end(...)`.
- If you add fragmentation series keyed by `Request Id`, they can be emitted into the same CSV.

### 4. Worker and engine hooks

- [base_worker.py](../sarathi-lean/sarathi/worker/base_worker.py#L135)
- [base_llm_engine.py](../sarathi-lean/sarathi/engine/base_llm_engine.py#L293)

What matters:

- Requests are finalized in the normal batch/step flow.
- Metrics should be captured at a deterministic point in that flow (same place request metrics are already finalized) so request IDs and final lengths are stable.

### 5. Docker output path

- [scripts/docker/start-server.sh](../scripts/docker/start-server.sh#L7)

What matters:

- Docker already passes `--output_dir` into the in-container server process.
- Sarathi metrics write to that output dir.
- Ray exporter warnings are separate and should not block this work.

## Implementation steps (what to change and why)

1. Add allocator metric names in `constants.py`.

- Add names for at least:
  - `KV_BLOCKS_MAPPED`
  - `KV_FRAGMENTATION_PERCENT`
- Why: metric constants prevent string drift and keep plots/CSV naming consistent.

Example starter shape in [constants.py](../sarathi-lean/sarathi/metrics/constants.py#L79):

```python
class SequenceMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_NUM_RESTARTS = "request_num_restarts"
    REQUEST_NUM_PAUSES = "request_num_pauses"
    REQUEST_NUM_IGNORED = "request_num_ignored"
    KV_BLOCKS_MAPPED = "kv_blocks_mapped"
    KV_FRAGMENTATION_PERCENT = "kv_fragmentation_percent"
```

This works well because `sequence_metrics.csv` is already built from request-level metric series keyed by `Request Id`.

2. Add a request-level allocator snapshot helper in `vATTN_cache_engine.py`.

- Add a method (for example `get_request_allocator_metrics(seq_id)` or `get_allocator_metrics_for_requests(seq_ids)`) that:
  - resolves `seq_id -> batch_idx` using `seq_to_batch_idx`
  - reads current sequence length from `curr_seq_lens[batch_idx]`
  - computes mapped blocks from allocator state for that request
  - calls `vattention.debug_fragmentation_metrics(seq_len, mapped_blocks)`
  - returns structured values
- Why: the cache engine already owns allocator-facing state and is the correct layer to translate runtime request state into metrics.

Example starter shape in [vATTN_cache_engine.py](../sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L753):

```python
def get_request_allocator_metrics(self, seq_id: int) -> dict | None:
    batch_idx = self.seq_to_batch_idx.get(seq_id)
    if batch_idx is None:
        return None

    seq_len = int(self.curr_seq_lens[batch_idx])
    if seq_len <= 0:
        return None

    mapped_blocks = int(vattention.debug_tokens_to_pages(seq_len))
    metrics = dict(vattention.debug_fragmentation_metrics(seq_len, mapped_blocks))
    return {
        "mapped_blocks": mapped_blocks,
        "fragmentation_percent": float(metrics["token_frag_pct"]),
    }
```

This is a good first version because it only returns the two fields needed for the initial experiment.

3. Add storage for request-level allocator metrics in `MetricsStore`.

- Create new `DataSeries` entries keyed by `Request Id` (same key used by sequence metrics).
- Add a method like `push_request_allocator_metric(metric_name, request_id, value)`.
- Why: request-keyed storage is required to compare against `request_num_prefill_tokens` per request.

Example helper in [metrics_store.py](../sarathi-lean/sarathi/metrics/metrics_store.py#L155):

```python
@check_enabled
@if_write_metrics
def push_request_metric(self, metric_name, request_id: str, value: float) -> None:
    self.seq_metrics_histogram[metric_name].put(request_id, value)
```

This keeps the request-finalization call site small and reuses the existing request-keyed `DataSeries`.

4. Hook allocator metric capture at request-finalization time.

- In the request completion path (where `_on_request_end(...)` is effectively finalized), push fragmentation fields for that request ID.
- Ensure request ID uses the same format as existing sequence metrics (`replica_id + seq_id` via `_get_seq_id`).
- Why: this guarantees one aligned row per completed request.

Example call shape near [`_on_request_end(...)`](../sarathi-lean/sarathi/metrics/metrics_store.py#L295):

```python
request_id = self._get_seq_id(seq.seq_id)

self.push_request_metric(
    SequenceMetricsHistogram.KV_BLOCKS_MAPPED,
    request_id,
    allocator_metrics["mapped_blocks"],
)
self.push_request_metric(
    SequenceMetricsHistogram.KV_FRAGMENTATION_PERCENT,
    request_id,
    allocator_metrics["fragmentation_percent"],
)
```

This is the key step that ensures `request_num_prefill_tokens` and fragmentation end up on the same request row.

5. Emit allocator request metrics into `sequence_metrics.csv`.

- Include new request-level allocator `DataSeries` in `_store_seq_metrics(...)` so output lands with existing request columns.
- Optional: also keep a separate `allocator_metrics.csv` if you want batch-level debugging.
- Why: the experiment needs one table that already contains both context length and fragmentation.

Success should look conceptually like this:

```csv
Request Id,request_num_prefill_tokens,kv_blocks_mapped,kv_fragmentation_percent
0_17,32000,47,5.9
0_18,64000,94,3.0
0_19,128000,188,1.5
```

This is the main reason to prefer request-level storage over a batch-only CSV for this experiment.

6. Keep current allocator `printf` temporarily.

- Do not remove existing stdout fragmentation prints during rollout.
- Why: use stdout to spot-check values while validating CSV integration.

Quick sanity check before full wiring:

```python
import vattention

seq_len = 32000
mapped_blocks = vattention.debug_tokens_to_pages(seq_len)
metrics = vattention.debug_fragmentation_metrics(seq_len, mapped_blocks)

print("mapped_blocks:", mapped_blocks)
print("token_frag_pct:", metrics["token_frag_pct"])
print("mapped_physical_bytes:", metrics["mapped_physical_bytes"])
```

If this output already looks wrong, debug the allocator path before touching CSV-writing code.

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
