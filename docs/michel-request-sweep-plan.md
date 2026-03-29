# Request Sweep Plan: Sequential Context-Length Driver

This plan defines your work for automating sequential request submission across context lengths, while relying on Josh's metrics pipeline to record fragmentation results.

## Goal

Build a script that:

- accepts a model name
- sends requests sequentially (never concurrently)
- sweeps through target context lengths
- keeps decode short (small `max_tokens`) to reduce MLA decode cost
- leaves metrics capture to Josh's system (`sequence_metrics.csv` with fragmentation columns)

## Integration contract:

- You guarantee deterministic, sequential request ordering and known target context lengths.
- Josh guarantees request-level fragmentation metrics are emitted alongside request context length in metrics output.

## Codebase background you should understand

### 1. Server API path to call

- [api_server.py](~/repos/vattention/sarathi-lean/sarathi/entrypoints/openai_server/api_server.py)

What matters:

- Use `/v1/completions` endpoint.
- Requests should be synchronous and sequential from the client side.

### 2. Docker launch and metrics output location

- [start-server.sh](~/repos/vattention/scripts/docker/start-server.sh)
- [start-server-yi6b.sh](~/repos/vattention/scripts/docker/start-server-yi6b.sh)

What matters:

- Server runs inside container.
- Output directory is passed via `--output_dir` and defaults under `/tmp/vattention/<container-name>`.
- You do not need to implement metrics writing in your script.

### 3. Existing request-driving scripts style

- [benchmark_e2e_static_trace.py](~/repos/vattention/scripts/benchmark_e2e_static_trace.py)
- [utils.py](~/repos/vattention/scripts/utils.py)

What matters:

- Follow existing script style for arguments/logging where useful.
- Keep your script focused on deterministic sequential API calls rather than benchmark framework integration.

### 4. Metrics columns you depend on from Josh

- [josh-metrics-plan.md](~/repos/vattention/docs/josh-metrics-plan.md)

What matters:

- The experiment depends on request-level columns including context length and fragmentation.
- Your script should not duplicate allocator logic.

## Implementation steps (what to build and why)

1. Create a dedicated sweep script.

- Suggested path: `scripts/fragmentation_context_sweep.py`.
- Why: keeps experiment harness simple, reproducible, and separate from general benchmark tooling.

2. Keep the CLI very small.

- Require only `--model`.
- Keep the following as script constants unless there is a strong reason to expose them later:
  - base URL: `http://127.0.0.1:8000`
  - sweep context lengths
  - `max_tokens = 1`
  - `temperature = 0.0`
  - timeout
- Why: this script has one narrow job, and reducing arguments makes it easier to run consistently.

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

6. Write a simple run log file.

- A manifest here just means a small file that records what the script tried to send.
- Log one line per attempted request with at least:
  - run timestamp
  - model name
  - target context length
  - request index
  - HTTP status or error
  - latency
- Suggested path: put it next to the script output as a `.jsonl` or `.csv` file.
- Why: this makes it easy to confirm which requests were sent if metrics output later looks incomplete.

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

2. Run one manual smoke-test request first.

- Optional: check served model name:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq .
```

- Minimal completion call:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "01-ai/Yi-6B-200k",
    "prompt": "Say hello in five words.",
    "max_tokens": 4,
    "temperature": 0.0,
    "stream": false
  }' | jq .
```

3. Run a short dry sweep.

- Use 3-5 context lengths and `max_tokens=1`.

4. Confirm your script behavior.

- Requests are clearly sequential in logs.
- Run log file contains one record per request.

5. Confirm Josh metrics output is present.

- `sequence_metrics.csv` exists in run output directory.
- Request-level fragmentation columns exist (from Josh's work).

6. Check data usability.

- Verify each intended context length appears in metrics rows.
- Verify fragmentation columns are populated for those requests.

## Small code examples to get started

These are intentionally small so you can paste them in, run them, and then expand them into the real sweep script.

### Example 1: single request with `curl`

Use this to verify the server is up before writing any Python.

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "01-ai/Yi-6B-200k",
    "prompt": "Say hello in five words.",
    "max_tokens": 4,
    "temperature": 0.0,
    "stream": false
  }' | jq .
```

### Example 2: smallest useful Python request

This is the simplest Python version of the same call.

```python
#!/usr/bin/env python3
import json
import urllib.request


payload = {
    "model": "01-ai/Yi-6B-200k",
    "prompt": "Say hello in five words.",
    "max_tokens": 1,
    "temperature": 0.0,
    "stream": False,
}

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read().decode("utf-8"))

print(body["choices"][0]["text"])
print(body["usage"])
```

### Example 3: tiny sequential sweep skeleton

This shows the exact control flow you want for the real script: build prompt, send request, wait, then move to the next context length.

```python
#!/usr/bin/env python3
import json
import time
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
MODEL = "01-ai/Yi-6B-200k"
MAX_TOKENS = 1
TEMPERATURE = 0.0
CONTEXT_LENGTHS = [128, 512, 1024]


def make_prompt(target_tokens: int) -> str:
    # Simple deterministic starter prompt.
    return " ".join(["token"] * target_tokens)


def send_request(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


for idx, context_len in enumerate(CONTEXT_LENGTHS):
    prompt = make_prompt(context_len)
    started = time.time()
    response = send_request(prompt)
    elapsed = time.time() - started

    print(
        {
            "request_index": idx,
            "context_len": context_len,
            "latency_sec": round(elapsed, 3),
            "finish_reason": response["choices"][0].get("finish_reason"),
        }
    )
```

### Example 4: writing a tiny run log

This is the smallest version of the run log idea. It appends one JSON object per request to a `.jsonl` file.

```python
from pathlib import Path


RUN_LOG = Path("~/repos/vattention/tmp/context_sweep_run_log.jsonl").expanduser()
RUN_LOG.parent.mkdir(parents=True, exist_ok=True)


def append_run_log(record: dict) -> None:
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

Use it after each request:

```python
append_run_log(
    {
        "request_index": idx,
        "context_len": context_len,
        "latency_sec": elapsed,
        "finish_reason": response["choices"][0].get("finish_reason"),
    }
)
```

### Example 5: handling one failed request and continuing

This is the behavior you want in the real sweep.

```python
try:
    response = send_request(prompt)
except Exception as exc:
    append_run_log(
        {
            "request_index": idx,
            "context_len": context_len,
            "error": str(exc),
        }
    )
    continue
```

If you start from Examples 3, 4, and 5 together, you already have the basic shape of the final script.

## First milestone

Deliver this first:

- sequential sweep script with just `--model`
- one run log file per run
- successful short sweep against a running Docker server

After this milestone, Josh's metrics data can be joined/analyzed for fragmentation vs context length.

## What not to block on

- adding concurrency modes
- integrating into broader benchmark pipelines
- plotting in the same script

Those can be added later after sequential sweep reliability is confirmed.
