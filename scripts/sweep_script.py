
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

# --- Constants (not exposed as CLI args) ---
BASE_URL = "http://127.0.0.1:8000"
MAX_TOKENS = 1
TEMPERATURE = 0.0
TIMEOUT = 180  # seconds per request
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

def make_prompt(target_tokens: int) -> str:
    return " ".join(["token"] * target_tokens)

def send_request(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
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
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))
def append_run_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential context-length sweep for fragmentation analysis."
    )
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--fail-fast", action="store_true", help="Abort on first failure")
    parser.add_argument("--output", default=None, help="JSONL output file (default: auto-named)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path(f"sweep_{args.model.replace('/', '_')}_{timestamp}.jsonl")
    print(f"Logging results to {out_path}")

    attempted = 0
    succeeded = 0
    failed = 0
    latencies = []

    for ctx_len in CONTEXT_LENGTHS:
        prompt = make_prompt(ctx_len)
        print(f"  context={ctx_len:>6} tokens ... ", end="", flush=True)
        t0 = time.monotonic()
        attempted += 1
        try:
            response = send_request(args.model, prompt)
            elapsed = time.monotonic() - t0
            record = {
                "context_length": ctx_len,
                "status": "ok",
                "latency_s": round(elapsed, 3),
                "response": response,
            }
            print(f"ok ({elapsed:.2f}s)")
            succeeded += 1
            latencies.append(elapsed)
        except urllib.error.HTTPError as e:
            elapsed = time.monotonic() - t0
            body = e.read().decode("utf-8", errors="replace")
            record = {
                "context_length": ctx_len,
                "status": "http_error",
                "latency_s": round(elapsed, 3),
                "http_status": e.code,
                "error": body,
            }
            print(f"HTTP {e.code} ({elapsed:.2f}s)")
            failed += 1
            if args.fail_fast:
                append_run_log(out_path, record)
                raise SystemExit(1)
        except Exception as e:
            elapsed = time.monotonic() - t0
            record = {
                "context_length": ctx_len,
                "status": "error",
                "latency_s": round(elapsed, 3),
                "error": str(e),
            }
            print(f"ERROR: {e}")
            failed += 1
            if args.fail_fast:
                append_run_log(out_path, record)
                raise SystemExit(1)

        append_run_log(out_path, record)

    print()
    print("=== Sweep summary ===")
    print(f"  attempted:    {attempted}")
    print(f"  succeeded:    {succeeded}")
    print(f"  failed:       {failed}")
    print(f"  success rate: {succeeded/attempted*100:.0f}%")
    if latencies:
        print(f"  latency (s):  min={min(latencies):.3f}  avg={sum(latencies)/len(latencies):.3f}  max={max(latencies):.3f}")
    print(f"  run log:      {out_path}")


if __name__ == "__main__":
    main()
