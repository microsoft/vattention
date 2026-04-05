#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 600
MAX_TOKENS = 1
TEMPERATURE = 0.0
HF_CACHE_DIR = Path(os.environ.get("HF_HOME", "/tmp/vattention-hf-home"))
RUNNER_OUTPUT_ROOT = Path(
    os.environ.get("VATTN_FRAGMENTATION_SWEEP_OUTPUT_DIR", "/tmp/vattention-frag-sweep")
)
CONTEXT_LENGTHS = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)

PROMPT_SEED_TEXT = (
    "This request is part of a deterministic context-length sweep for "
    "fragmentation analysis. The content is intentionally repetitive so the "
    "prompt can be expanded to an exact token count while staying stable "
    "across runs. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a sequential context-length sweep against the local OpenAI-compatible server."
    )
    parser.add_argument("--model", required=True, help="Served model name.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after the first failed request.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_hf_cache_dirs() -> None:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))


def load_tokenizer(model_name: str):
    ensure_hf_cache_dirs()

    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency `transformers`. Activate `.venv-frag-sweep` "
            "or run `scripts/setup-fragmentation-context-sweep-venv.sh` first."
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except KeyError:
        tokenizer_path = Path(model_name)
        if not (tokenizer_path / "tokenizer.json").exists():
            raise
        return PreTrainedTokenizerFast.from_pretrained(model_name)


def encode_without_special_tokens(tokenizer: Any, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def build_prompt_token_pool(tokenizer: Any) -> List[int]:
    repeated_seed = PROMPT_SEED_TEXT * 256
    token_ids = encode_without_special_tokens(tokenizer, repeated_seed)
    if not token_ids:
        raise ValueError("Tokenizer produced no prompt tokens for the fixed sweep seed text.")
    return token_ids


def build_exact_prompt_token_ids(
    target_length: int, token_pool: Sequence[int]
) -> List[int]:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if not token_pool:
        raise ValueError("token_pool must not be empty")

    repeats = (target_length + len(token_pool) - 1) // len(token_pool)
    prompt_token_ids = list(token_pool) * repeats
    return prompt_token_ids[:target_length]


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return {
            "status_code": response.getcode(),
            "body": json.loads(body) if body else {},
        }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def create_run_dir(model_name: str) -> Path:
    safe_model = model_name.replace("/", "__")
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_model}"
    run_dir = RUNNER_OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def summarize_attempts(attempts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    succeeded = [attempt for attempt in attempts if attempt["ok"]]
    failed = [attempt for attempt in attempts if not attempt["ok"]]
    latencies = [attempt["latency_seconds"] for attempt in succeeded]
    prompt_token_mismatches = [
        attempt
        for attempt in succeeded
        if attempt.get("actual_prompt_tokens") != attempt["target_context_length"]
    ]

    return {
        "attempted": len(attempts),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "success_rate": (len(succeeded) / len(attempts)) if attempts else 0.0,
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "prompt_token_mismatches": len(prompt_token_mismatches),
    }


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    token_pool = build_prompt_token_pool(tokenizer)

    run_dir = create_run_dir(args.model)
    manifest_path = run_dir / "request_manifest.jsonl"
    metadata_path = run_dir / "run_metadata.json"

    metadata = {
        "started_at_utc": utc_timestamp(),
        "model": args.model,
        "base_url": BASE_URL,
        "context_lengths": list(CONTEXT_LENGTHS),
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "manifest_path": str(manifest_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    attempts: List[Dict[str, Any]] = []
    url = f"{BASE_URL}/v1/completions"

    print(f"Run directory: {run_dir}")
    print(f"Manifest: {manifest_path}")

    for request_index, target_context_length in enumerate(CONTEXT_LENGTHS, start=1):
        prompt_token_ids = build_exact_prompt_token_ids(target_context_length, token_pool)
        payload = {
            "model": args.model,
            "prompt": prompt_token_ids,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": False,
        }

        started_at = time.perf_counter()
        record: Dict[str, Any] = {
            "timestamp_utc": utc_timestamp(),
            "request_index": request_index,
            "model": args.model,
            "target_context_length": target_context_length,
            "submitted_prompt_tokens": len(prompt_token_ids),
            "ok": False,
        }

        print(
            f"[{request_index}/{len(CONTEXT_LENGTHS)}] sending request with "
            f"{target_context_length} prompt tokens"
        )

        try:
            response = post_json(url, payload, timeout=REQUEST_TIMEOUT_SECONDS)
            latency_seconds = time.perf_counter() - started_at
            body = response["body"]
            usage = body.get("usage", {})
            record.update(
                {
                    "ok": True,
                    "status_code": response["status_code"],
                    "latency_seconds": round(latency_seconds, 6),
                    "request_id": body.get("id"),
                    "actual_prompt_tokens": usage.get("prompt_tokens"),
                    "actual_completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": (
                        body.get("choices", [{}])[0].get("finish_reason")
                        if body.get("choices")
                        else None
                    ),
                }
            )
        except urllib.error.HTTPError as exc:
            latency_seconds = time.perf_counter() - started_at
            error_body = exc.read().decode("utf-8", errors="replace")
            record.update(
                {
                    "status_code": exc.code,
                    "latency_seconds": round(latency_seconds, 6),
                    "error": error_body,
                }
            )
        except urllib.error.URLError as exc:
            latency_seconds = time.perf_counter() - started_at
            record.update(
                {
                    "status_code": None,
                    "latency_seconds": round(latency_seconds, 6),
                    "error": str(exc.reason),
                }
            )
        except Exception as exc:
            latency_seconds = time.perf_counter() - started_at
            record.update(
                {
                    "status_code": None,
                    "latency_seconds": round(latency_seconds, 6),
                    "error": str(exc),
                }
            )

        append_jsonl(manifest_path, record)
        attempts.append(record)

        if record["ok"]:
            print(
                f"  success in {record['latency_seconds']:.3f}s "
                f"(usage.prompt_tokens={record.get('actual_prompt_tokens')})"
            )
        else:
            print(
                f"  failed in {record['latency_seconds']:.3f}s "
                f"(status={record.get('status_code')}, error={record.get('error')})"
            )
            if args.fail_fast:
                break

    summary = summarize_attempts(attempts)
    summary["finished_at_utc"] = utc_timestamp()
    summary["manifest_path"] = str(manifest_path)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("")
    print("Sweep summary")
    print(f"  attempted: {summary['attempted']}")
    print(f"  succeeded: {summary['succeeded']}")
    print(f"  failed: {summary['failed']}")
    print(f"  success_rate: {summary['success_rate']:.2%}")
    if summary["mean_latency_seconds"] is not None:
        print(f"  mean_latency_seconds: {summary['mean_latency_seconds']:.3f}")
        print(f"  median_latency_seconds: {summary['median_latency_seconds']:.3f}")
    print(f"  prompt_token_mismatches: {summary['prompt_token_mismatches']}")
    print(f"  manifest_path: {manifest_path}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
