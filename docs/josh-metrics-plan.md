# Josh Metrics Plan

This note captures the current plan for wiring vAttention allocator stats into Sarathi's in-repo metrics system.

## Goal

Capture the vAttention allocator information that is currently printed to stdout, especially:

- number of mapped/free KV blocks
- fragmentation percentage

and save it through Sarathi's metrics pipeline so it lands in the run output directory alongside the other benchmark metrics.

## What is already working

- The Docker multiuser server starts successfully with `scripts/docker/start-server-yi6b.sh`.
- The OpenAI-compatible API is reachable on `/v1/completions`.
- Ray logs warnings about its own metrics exporter agent, but the server still starts.

For this task, the better target is Sarathi's own metrics system, not Ray's internal metrics exporter.

## Where the fragmentation information is printed today

The current fragmentation print is in:

- [vattention/vattention.cu](/home/anodyine/repos/vattention/vattention/vattention.cu)

Relevant lines:

- [vattention.cu](/home/anodyine/repos/vattention/vattention/vattention.cu#L414): computes `frag_percent`
- [vattention.cu](/home/anodyine/repos/vattention/vattention/vattention.cu#L418): prints the fragmentation line

The current print format includes:

- request id
- total sequence length
- mapped blocks
- resident useful tokens
- useful MB
- physical MB
- internal fragmentation percent

Related block and cache logging already exists here:

- [base_llm_engine.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/engine/base_llm_engine.py#L246) logs `# GPU blocks`
- [vATTN_cache_engine.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L48) prints KV-cache init details
- [vATTN_cache_engine.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py#L85) prints reserved physical memory

## Recommended implementation plan

1. Add a Python-callable stats getter in the `vattention` extension.

   In [vattention/vattention.cu](/home/anodyine/repos/vattention/vattention/vattention.cu), factor the current fragmentation calculation into a helper that returns structured values instead of only printing them.

   Suggested fields:

   - `mapped_blocks`
   - `resident_useful_tokens`
   - `useful_mb`
   - `physical_mb`
   - `frag_percent`

   Expose the helper through the existing pybind module near functions like `num_free_kvblocks`.

2. Add a cache-engine wrapper for those stats.

   In [vATTN_cache_engine.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py), add a method such as `get_allocator_metrics()` that calls the new `vattention` binding and returns the current allocator snapshot.

3. Sample the metrics once per batch.

   The cleanest hook is in [base_worker.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/worker/base_worker.py#L199), right next to:

   - `self.metrics_store.on_batch_stage_end(...)`

   After the batch completes, pull the allocator stats from the cache engine and push them into the metrics store.

4. Add new metric names to Sarathi.

   Extend [constants.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/metrics/constants.py) with allocator-specific metric names.

   Suggested initial metrics:

   - `KV_BLOCKS_FREE`
   - `KV_BLOCKS_MAPPED`
   - `KV_FRAGMENTATION_PERCENT`
   - `KV_USEFUL_MB`
   - `KV_PHYSICAL_MB`

5. Extend `MetricsStore` for allocator metrics.

   In [metrics_store.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/metrics/metrics_store.py), add a new per-batch `DataSeries` collection for allocator metrics, similar to the existing per-batch metric handling.

   Add a helper like:

   - `push_allocator_metric(metric_name, value)`

   keyed by the current batch id.

6. Save allocator metrics to their own CSV output.

   Follow the existing CSV-writing patterns in [metrics_store.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/metrics/metrics_store.py).

   Recommended output:

   - `allocator_metrics.csv`

   in the same run output directory used by the rest of Sarathi's metrics.

7. Keep the current `printf` during rollout.

   Do not remove the existing fragmentation print immediately. Keep it while validating the new metrics path so stdout can be compared against the saved CSV values.

8. Validate end to end.

   - Start the server with `scripts/docker/start-server-yi6b.sh`
   - Send one or more requests to `/v1/completions`
   - Inspect the output directory under `/tmp/vattention/<container-name>`
   - Confirm `allocator_metrics.csv` contains the expected block and fragmentation values
   - Compare those values with the existing stdout prints

## Suggested first milestone

The first useful milestone is:

- expose allocator stats from `vattention`
- record just `KV_BLOCKS_MAPPED` and `KV_FRAGMENTATION_PERCENT`
- save them once per batch to `allocator_metrics.csv`

That is enough to prove the integration path works before adding extra fields.

## What not to block on

- Ray's internal metrics exporter warnings
- Ray dashboard or Prometheus integration

Those are separate from Sarathi's in-repo metrics output and do not need to be solved first for this task.
