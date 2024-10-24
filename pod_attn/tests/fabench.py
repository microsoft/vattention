import sys
import os
import fused_ampere as fused_amp
import torch
import math
from einops import rearrange
import time

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
def launch_big_kernel():
    #m, n, k = 48000, 48000, 48000
    #a = torch.randn(m, k, device='cuda', dtype=torch.float16)
    #b = torch.randn(k, n, device='cuda', dtype=torch.float16)
    #c = torch.matmul(a, b)
    #return c
    return 0

repeat = 100

@torch.inference_mode
def do_fa_prefill(q, k, v, seq_lens_k=None, fused_params=0):
    try:
        output = None
        output = fused_amp.flash_attn_with_kvcache(q, k, v, causal=True, cache_seqlens=seq_lens_k, fused_params=fused_params)
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output = fused_amp.flash_attn_with_kvcache(q, k, v, causal=True, cache_seqlens=seq_lens_k, fused_params=fused_params)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output, duration
    except Exception as e:
        print(e)
        return None, -1

@torch.inference_mode
def do_fa_decode(q, k, v, k_new=None, v_new=None, seq_lens_k=None, cache_batch_idx=None, splits=0, fused_params=0):
    try:
        output = fused_amp.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k, cache_batch_idx=cache_batch_idx, num_splits=splits, fused_params=fused_params)
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output = fused_amp.flash_attn_with_kvcache(q, k, v, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k, cache_batch_idx=cache_batch_idx, num_splits=splits, fused_params=fused_params)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output, duration
    except Exception as e:
        print(e)
        return None, -1

@torch.inference_mode
def do_fa_prefill_decode(q_p, k_p, v_p, q_d, k_d, v_d, k_new=None, v_new=None, seq_lens_k_p=None, seq_lens_k_d=None, cache_batch_idx=None, splits=0, fused_params=0):
    try:
        output_pref = fused_amp.flash_attn_with_kvcache(q_p, k_p, v_p, causal=True, cache_seqlens=seq_lens_k_p)
        output_dec = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, causal=False, fused_params=fused_params)
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output_pref = fused_amp.flash_attn_with_kvcache(q_p, k_p, v_p, causal=True, cache_seqlens=seq_lens_k_p)
            output_dec = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, causal=False, fused_params=fused_params)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output_pref, output_dec, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output_pref, output_dec, duration
    except Exception as e:
        print(e)
        return None, None, -1

@torch.inference_mode
def do_fa3_prefill(q, k, v):
    import fused_hopper as fused_hop
    try:
        output = None
        output = fused_hop.flash_attn_func(q, k, v, causal=True)[0]
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output = fused_hop.flash_attn_func(q, k, v, causal=True)[0]
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output, duration
    except Exception as e:
        print(e)
        return None, -1

@torch.inference_mode
def do_fa3_decode(q, k, v, k_new=None, v_new=None, seq_lens_k=None, cache_batch_idx=None, splits=0):
    import fused_hopper as fused_hop
    try:
        output = fused_hop.flash_attn_func(q, k, v, causal=False)[0]
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output = fused_hop.flash_attn_func(q, k, v, causal=False)[0]
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output, duration
    except Exception as e:
        print(e)
        return None, -1

@torch.inference_mode
def do_fa3_prefill_decode(q_p, k_p, v_p, q_d, k_d, v_d, k_new=None, v_new=None, seq_lens_k_d=None, cache_batch_idx=None, fused_params=0):
    import fused_hopper as fused_hop
    try:
        output_pref = fused_hop.flash_attn_func(q_p, k_p, v_p, causal=True)[0]
        output_dec = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, causal=False, fused_params=fused_params)
        #launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output_pref = fused_hop.flash_attn_func(q_p, k_p, v_p, causal=True)[0]
            output_dec = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, causal=False, fused_params=fused_params)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output_pref, output_dec, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output_pref, output_dec, duration
    except Exception as e:
        print(e)
        return None, None, -1

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
@torch.inference_mode()
def do_fa_prefill_decode_streams(q_p, k_p, v_p, q_d, k_d, v_d, k_new=None, v_new=None, seq_lens_k_p=None, seq_lens_k_d=None, cache_batch_idx=None, splits=0, fused_params=0):
    try:
        with torch.cuda.stream(stream1):
            output_pref = fused_amp.flash_attn_with_kvcache(q_p, k_p, v_p, causal=True, cache_seqlens=seq_lens_k_p)
        with torch.cuda.stream(stream2):
            output_dec = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, causal=False, fused_params=fused_params)

        stream1.synchronize()
        stream2.synchronize()
        start.record()
        for _ in range(repeat):
            with torch.cuda.stream(stream1):
                output_pref = fused_amp.flash_attn_with_kvcache(q_p, k_p, v_p, causal=True, cache_seqlens=seq_lens_k_p)
            with torch.cuda.stream(stream2):
                output_dec = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, causal=False, fused_params=fused_params)
            stream1.synchronize()
            stream2.synchronize()

        end.record()
        # Synchronize the device to ensure all operations are complete
        torch.cuda.synchronize()
        return output_pref, output_dec, round(start.elapsed_time(end) / repeat, 3)
    except Exception as e:
        print(e)
        return None, None, -1

def do_mps_prefill(queue, bs, cl, k_cl, num_heads, num_kv_heads, head_size, reps, seq_lens_k_p=None, fused_params=0):
    try:
        q_p = torch.randn(bs, cl, num_heads, head_size, device='cuda', dtype=torch.float16)
        k_p = torch.randn(bs, k_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
        v_p = torch.randn(bs, k_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
        output = fused_amp.flash_attn_with_kvcache(q_p, k_p, v_p, causal=True, cache_seqlens=seq_lens_k_p, fused_params=fused_params)
        launch_big_kernel()
        queue.put(("prefill", time.perf_counter_ns()))
        for _ in range(reps):
            output = fused_amp.flash_attn_with_kvcache(q_p, k_p, v_p, causal=True, cache_seqlens=seq_lens_k_p, fused_params=fused_params)
        torch.cuda.synchronize()
        queue.put(("prefill", time.perf_counter_ns()))
    except Exception as e:
        print(e)
    quit()

def do_mps_decode(queue, bs, cl, k_cl, num_heads, num_kv_heads, head_size, reps, k_new=None, v_new=None, seq_lens_k_d=None, cache_batch_idx=None, fused_params=0):
    try:
        q_d = torch.randn(bs, cl, num_heads, head_size, device='cuda', dtype=torch.float16)
        k_d = torch.randn(bs, k_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
        v_d = torch.randn(bs, k_cl, num_kv_heads, head_size, device='cuda', dtype=torch.float16)
        output = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params)
        launch_big_kernel()
        queue.put(("decode", time.perf_counter_ns()))
        for _ in range(reps):
            output = fused_amp.flash_attn_with_kvcache(q_d, k_d, v_d, causal=False, k=k_new, v=v_new, cache_seqlens=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params)
        torch.cuda.synchronize()
        queue.put(("decode", time.perf_counter_ns()))
    except Exception as e:
        print(e)

def start_mps():
    import subprocess
    #subprocess.run(["nvidia-smi", "-i", "0", "-c", "EXCLUSIVE_PROCESS"])
    subprocess.run(["nvidia-cuda-mps-control", "-d"])

def end_mps():
    import subprocess
    subprocess.run(["echo quit | nvidia-cuda-mps-control"], shell=True)
    #subprocess.run(["nvidia-smi", "-i", "0", "-c", "DEFAULT"])

# Merge using MPS
@torch.inference_mode
def do_mps_attn(prefill_params, decode_params):
    import multiprocessing
    (bs_p, cl_p, num_heads_p, head_size_p, k_cl_p, num_kv_heads_p) = prefill_params
    (bs_d, cl_d, num_heads_d, head_size_d, k_cl_d, num_kv_heads_d) = decode_params
    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass
    reps = 1000
    queue = multiprocessing.Queue()
    prefill_proc = multiprocessing.Process(target=do_mps_prefill, args=(queue, bs_p, cl_p, k_cl_p, num_heads_p, num_kv_heads_p, head_size_p, reps))
    decode_proc = multiprocessing.Process(target=do_mps_decode, args=(queue, bs_d, cl_d, k_cl_d, num_heads_d, num_kv_heads_d, head_size_d, reps))
    prefill_proc.start()
    decode_proc.start()
    prefill_proc.join()
    decode_proc.join()
    dec_time = []
    pref_time = []
    mintime = 9999999999999999999
    maxtime = -1
    while not queue.empty():
        entry = queue.get()
        mintime = min(mintime, entry[1])
        maxtime = max(maxtime, entry[1])
        if entry[0] == "decode":
            dec_time.append(entry[1])
        else:
            pref_time.append(entry[1])
    # Return time in ms
    return (maxtime - mintime) / reps / 1000000

# Ignore the below function. It is a dummy just for testing
@torch.inference_mode
def do_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, ratio=0):
    try:
        output = None
        fused_amp.fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, causal=True, ratio=ratio)
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output = fused_amp.fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, causal=True, ratio=ratio)
        end.record()
        torch.cuda.synchronize()
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output, duration
    except Exception as e:
        print(e)
        return None, -1

# Below functions are the real fused operations
@torch.inference_mode
def do_true_fused_attn(q_p, k_p, v_p, q_d, k_d, v_d, fused_params=0, k_new=None, v_new=None, seq_lens_k_p=None, seq_lens_k_d=None, cache_batch_idx=None, num_splits_p=0, num_splits_d=0):
    try:
        output_pref, output_dec = fused_amp.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=True, \
                cache_seqlens_p=seq_lens_k_p, cache_seqlens_d=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d)
        launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output_pref, output_dec = fused_amp.true_fused_attn_with_kvcache(q_p, k_p, v_p, q_d, k_d, v_d, k=k_new, v=v_new, causal=True, \
                cache_seqlens_p=seq_lens_k_p, cache_seqlens_d=seq_lens_k_d, cache_batch_idx=cache_batch_idx, fused_params=fused_params, num_splits_p=num_splits_p, num_splits_d=num_splits_d)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output_pref, output_dec, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output_pref, output_dec, duration
    except Exception as e:
        print(e)
        return None, None, -1

@torch.inference_mode
def do_true_fused_attn3(q_p, k_p, v_p, q_d, k_d, v_d, fused_params=0, k_new=None, v_new=None, seq_lens_k_p=None, seq_lens_k_d=None, cache_batch_idx=None):
    import fused_hopper as fused_hop
    try:
        output_pref, output_dec = fused_hop.fused_attn_func(q_p, k_p, v_p, q_d, k_d, v_d, causal=True, \
                fused_params=fused_params)
        #launch_big_kernel()
        start.record()
        for _ in range(repeat):
            output_pref, output_dec = fused_hop.fused_attn_func(q_p, k_p, v_p, q_d, k_d, v_d, causal=True, \
                fused_params=fused_params)
        end.record()
        torch.cuda.synchronize()
        if repeat == 0:
            return output_pref, output_dec, 0
        duration = round(start.elapsed_time(end) / repeat, 3)
        return output_pref, output_dec, duration
    except Exception as e:
        print(e)
        return None, None, -1
