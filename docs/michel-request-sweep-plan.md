# Michel Request Sweep Plan: Sequential Context-Length Driver

This plan defines Michel's work for automating sequential request submission across context lengths, while relying on Josh's metrics pipeline to record fragmentation results.

## Goal

Build a script that:

- accepts a model name
- sends requests sequentially (never concurrently)
- sweeps through target context lengths
- keeps decode short (small `max_tokens`) to reduce MLA decode cost
- leaves metrics capture to Josh's system (`sequence_metrics.csv` with fragmentation columns)

## Integration contract:

- Michel guarantees deterministic, sequential request ordering and known target context lengths.
- Josh guarantees request-level fragmentation metrics are emitted alongside request context length in metrics output.

## Codebase background you should understand

### 1. Server API path to call

- [api_server.py](/home/anodyine/repos/vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py)

What matters:

- Use `/v1/completions` endpoint.
- Requests should be synchronous and sequential from the client side.

### 2. Docker launch and metrics output location

- [start-server.sh](/home/anodyine/repos/vattention/scripts/docker/start-server.sh)
- [start-server-yi6b.sh](/home/anodyine/repos/vattention/scripts/docker/start-server-yi6b.sh)

What matters:

- Server runs inside container.
- Output directory is passed via `--output_dir` and defaults under `/tmp/vattention/<container-name>`.
- Michel does not need to implement metrics writing in his script.

### 3. Existing request-driving scripts style

- [benchmark_e2e_static_trace.py](/home/anodyine/repos/vattention/scripts/benchmark_e2e_static_trace.py)
- [utils.py](/home/anodyine/repos/vattention/scripts/utils.py)

What matters:

- Follow existing script style for arguments/logging where useful.
- Keep Michel's script focused on deterministic sequential API calls rather than benchmark framework integration.

### 4. Metrics columns Michel depends on from Josh

- [josh-metrics-plan.md](/home/anodyine/repos/vattention/docs/josh-metrics-plan.md)

What matters:

- The experiment depends on request-level columns including context length and fragmentation.
- Michel's script should not duplicate allocator logic.

## Implementation steps (what to build and why)

1. Create a dedicated sweep script.

- Suggested path: `scripts/fragmentation_context_sweep.py`.
- Why: keeps experiment harness simple, reproducible, and separate from general benchmark tooling.

2. Add required CLI arguments.

- `--model` (required)
- `--base-url` (default `http://127.0.0.1:8000`)
- `--context-lengths` (comma-separated or repeated values)
- `--max-tokens` (default `1`)
- `--temperature` (default `0.0`)
- `--timeout-seconds` (default reasonable value)
- `--output-manifest` (path for request log, e.g. JSONL)
- Why: these are the minimum controls needed to reproduce runs and compare models.

3. Generate deterministic prompts at exact target lengths.

- For each target context length, create a prompt template and trim/extend deterministically.
- Keep content stable across runs and models as much as practical.
- Why: stable prompt construction reduces experimental noise.

4. Enforce strictly sequential execution.

- Send one request.
- Wait for completion (or timeout/failure handling) before sending the next.
- No async fan-out, no thread pool.
- Why: avoids batch interaction effects and preserves clean fragmentation-vs-length interpretation.

5. Keep decode intentionally short.

- Use small `max_tokens` (e.g., `1` to `4`).
- Why: decode is slow in current MLA path; experiment focus is context-length effect on fragmentation.

6. Write a per-request manifest file.

- Log one line per attempted request with at least:
  - run timestamp
  - model name
  - target context length
  - request index
  - HTTP status / error
  - latency
- Why: gives traceability and an audit trail to match with metrics outputs.

7. Add robust failure handling and continue policy.

- On single-request failure, log error and continue to next context length unless `--fail-fast` is set.
- Why: long sweeps should not be lost due to one transient request failure.

8. Print concise end-of-run summary.

- Total requests attempted/succeeded/failed.
- Success rate and latency summary.
- Manifest path reminder.
- Why: fast sanity check before downstream analysis.

## Validation procedure

1. Start server in Docker.

- Example: `scripts/docker/start-server-yi6b.sh --model_name <target-model>` if needed.

2. Run a short dry sweep.

- Use 3-5 context lengths and `max_tokens=1`.

3. Confirm Michel script behavior.

- Requests are clearly sequential in logs.
- Manifest file contains one record per request.

4. Confirm Josh metrics output is present.

- `sequence_metrics.csv` exists in run output directory.
- Request-level fragmentation columns exist (from Josh's work).

5. Check data usability.

- Verify each intended context length appears in metrics rows.
- Verify fragmentation columns are populated for those requests.

## First milestone

Deliver this first:

- sequential sweep script with `--model`, `--context-lengths`, and `--max-tokens`
- one manifest output file per run
- successful short sweep against a running Docker server

After this milestone, Josh's metrics data can be joined/analyzed for fragmentation vs context length.

## What not to block on

- adding concurrency modes
- integrating into broader benchmark pipelines
- plotting in the same script

Those can be added later after sequential sweep reliability is confirmed.
