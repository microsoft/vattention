#!/usr/bin/env python3

import argparse
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Initialize vAttention and run a minimal fragmentation sanity check. "
			"Intended for GPU boxes (CUDA required)."
		)
	)
	parser.add_argument("--seq-len", type=int, default=32000)

	# vAttention init params (match your model for exact sizing)
	parser.add_argument("--num-layers", type=int, default=24)
	parser.add_argument("--num-kv-heads", type=int, default=8)
	parser.add_argument("--head-size", type=int, default=128)
	parser.add_argument("--max-batch-size", type=int, default=8)
	parser.add_argument("--max-context-length", type=int, default=131072)
	parser.add_argument("--device-idx", type=int, default=0)
	parser.add_argument(
		"--dtype",
		choices=["fp16", "bf16"],
		default="fp16",
		help="KV cache dtype passed to vattention.init_kvcache",
	)
	parser.add_argument(
		"--page-size-mb",
		type=int,
		default=2,
		help="vAttention page size (MiB)",
	)
	parser.add_argument(
		"--megacache",
		action="store_true",
		help="Enable megacache sizing in vAttention init",
	)

	# Optional behaviors
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable vAttention verbose logging (stdout)",
	)
	parser.add_argument(
		"--show-allocator-state",
		action="store_true",
		help="Call vattention.show_allocator_state() after init and after step.",
	)
	parser.add_argument(
		"--reserve-mb",
		type=int,
		default=0,
		help=(
			"If set, call vattention.reserve_physical_pages(reserve_mb * 1024**2). "
			"Typically needed if you want to run vattention.step()."
		),
	)
	parser.add_argument(
		"--trigger-step",
		action="store_true",
		help=(
			"Run one vattention.step() for req0 to trigger allocator mapping/printfs. "
			"Usually requires --reserve-mb."
		),
	)
	return parser.parse_args()


def _dtype_from_string(torch_mod: Any, dtype: str):
	if dtype == "fp16":
		return torch_mod.float16
	if dtype == "bf16":
		return torch_mod.bfloat16
	raise ValueError(f"Unsupported dtype: {dtype}")


def main() -> int:
	args = _parse_args()

	try:
		import torch
	except ModuleNotFoundError:
		print("ERROR: torch is not installed in this environment.", file=sys.stderr)
		return 2

	if not torch.cuda.is_available():
		print("ERROR: CUDA is required (torch.cuda.is_available() is false).", file=sys.stderr)
		return 2

	try:
		import vattention
	except ModuleNotFoundError:
		print(
			"ERROR: vattention extension is not importable. Build/install it first.",
			file=sys.stderr,
		)
		return 2

	# Ensure CUDA context exists.
	torch.empty(1, device=f"cuda:{args.device_idx}")

	dtype = _dtype_from_string(torch, args.dtype)
	page_size_bytes = args.page_size_mb * 1024 * 1024

	kv_tensors = None
	try:
		if args.verbose:
			vattention.set_verbose(True)

		# Keep returned tensors alive for the duration of the process.
		kv_tensors = vattention.init_kvcache(
			args.num_layers,
			args.num_kv_heads,
			args.head_size,
			args.max_batch_size,
			args.max_context_length,
			args.device_idx,
			dtype,
			page_size_bytes,
			bool(args.megacache),
		)

		if args.reserve_mb > 0:
			vattention.reserve_physical_pages(args.reserve_mb * 1024 * 1024)

		if args.show_allocator_state:
			vattention.show_allocator_state()

		seq_len = int(args.seq_len)
		mapped_blocks = int(vattention.debug_tokens_to_pages(seq_len))
		metrics = dict(vattention.debug_fragmentation_metrics(seq_len, mapped_blocks))

		print("\n=== vAttention Fragmentation Sanity Check ===")
		print(f"seq_len: {seq_len}")
		print(f"mapped_blocks (debug_tokens_to_pages): {mapped_blocks}")
		print(f"token_frag_pct: {metrics.get('token_frag_pct')}")
		print(f"mapped_physical_bytes: {metrics.get('mapped_physical_bytes')}")

		if args.trigger_step:
			seq_lens = [0] * int(args.max_batch_size)
			seq_lens[0] = seq_len
			vattention.step(seq_lens, True)
			if args.show_allocator_state:
				vattention.show_allocator_state()

		return 0
	finally:
		try:
			vattention.cleanup()
		except Exception:
			pass
		_ = kv_tensors


if __name__ == "__main__":
	raise SystemExit(main())

