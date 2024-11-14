from typing import Optional
import argparse
import random
import time
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plots = os.path.join(src, "plots")

from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, create_kv_caches_with_random
from vllm._C import ops

NUM_BLOCKS = 1024
PARTITION_SIZE = 512

@torch.inference_mode()
def main(
    version: str,
    num_seqs: int,
    context_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    kv_cache_dtype: Optional[str] = None,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [context_len for _ in range(num_seqs)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV cache.
    key_caches, value_caches = create_kv_caches_with_random(
        NUM_BLOCKS, block_size, 1, num_kv_heads, head_size, kv_cache_dtype,
        dtype)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Prepare for the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        tmp_output = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_query_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)

    def run_benchmark(num_iters: int, profile: bool = False) -> float:
        torch.cuda.synchronize()
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        latencies = []

        for _ in range(5):
            start_time = time.perf_counter()
            for _ in range(num_iters):
                if version == "v1":
                    ops.paged_attention_v1(
                        output,
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        scale,
                        block_tables,
                        context_lens,
                        block_size,
                        max_context_len,
                        alibi_slopes,
                        kv_cache_dtype,
                    )
                elif version == "v2":
                    ops.paged_attention_v2(
                        output,
                        exp_sums,
                        max_logits,
                        tmp_output,
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        scale,
                        block_tables,
                        context_lens,
                        block_size,
                        max_context_len,
                        alibi_slopes,
                        kv_cache_dtype,
                    )
                else:
                    raise ValueError(f"Invalid version: {version}")
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) / num_iters)

        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        return min(latencies)

    # Warmup.
    #print("Warming up...")
    run_benchmark(num_iters=3, profile=False)

    # Benchmark.
    if do_profile:
        latency = run_benchmark(num_iters=1, profile=True)
    else:
        latency = run_benchmark(num_iters=100, profile=False)
    runtime = round(latency * 1000000, 3)
    #print(f"Kernel running time: {runtime} us")
    return runtime

perf_record = {}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8_e5m2"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type.')
    args = parser.parse_args()
    #print(args)

    num_query_heads = 32
    num_kv_heads = 8
    head_size=128
    batch_sizes = [1, 2, 4, 8, 16]
    context_length=16384
    block_sizes = [16, 32, 64, 128]
    max_latency = -1
    for bs in batch_sizes:
        perf_record[bs] = {}
        for block_size in block_sizes:
            latency = main(
                        version="v2",
                        num_seqs=bs,
                        context_len=context_length,
                        num_query_heads=num_query_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        block_size=block_size,
                        use_alibi=False,
                        dtype=torch.float16,
                        seed=0,
                        do_profile=True,
                        kv_cache_dtype="auto",
                        )
            perf_record[bs][f"{block_size}"] = latency / 1000
            max_latency = max(max_latency, latency / 1000)
    df = pd.DataFrame(perf_record).transpose()
    df.to_csv(os.path.join(src, "logs/figure_3.csv"))
    #plot_figure(df, max_latency)


