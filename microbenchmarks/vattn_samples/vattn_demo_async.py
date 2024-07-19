import torch
import vattention
import sys
import time
import random
import utils

BLOCK_SIZE = utils.PAGE_SIZE // (utils.NUM_KV_HEADS * utils.HEAD_DIM * 2)

seqlens = [0 for i in range(utils.MAX_BATCH_SIZE)]
active_ids = []

def allocate_req_id():
    for id in range(utils.MAX_BATCH_SIZE):
        if id not in active_ids:
            return id

def get_mem_usage():
    num_blocks = 0
    for req_id in range(utils.MAX_BATCH_SIZE):
        num_blocks += (seqlens[req_id] + BLOCK_SIZE - 1) // BLOCK_SIZE

    num_pages = num_blocks * 2 * utils.NUM_LAYERS
    mem_usage = (num_pages * utils.PAGE_SIZE) // utils.MB
    return mem_usage

"""
def access_kv_cache(kv_cache, seqlens):
    for req_id in range(utils.MAX_BATCH_SIZE):
        if seqlens[req_id] == 0:
            continue
        #print(f"accessing req_id: {req_id} seq_len: {seqlens[req_id]}", flush=True)
        for l in range(2 * utils.NUM_LAYERS):
            kv_buff = kv_cache[l][req_id]
            kv_buff[:seqlens[req_id]].fill_(1.0)
"""

def do_kvcache_management_pass_1(kv_cache, nr_steps):
    total_sync_ms = 0
    # now, add/remove requests once every few steps to simulate a dynamic workload
    for i in range(nr_steps):
        # this allocates prefill kv cache synchronously and launches a background thread to
        # allocate memory for the next decoding step asynchronously, optimistically assuming
        # that all active requests are going to continue in the next iteration
        start = time.time()
        vattention.step_async(seqlens)
        end = time.time()
        sync_ms = round((end - start) * 1000, 3)

        # do not include the first step in the average sync time
        total_sync_ms += sync_ms if i > 1 else 0

        if sync_ms > 1:
            print(f"step: {i} sync time (ms): {sync_ms}", flush=True)

        #ensure we can access the cache in each step
        utils.access_kv_cache(kv_cache, seqlens)
 
        if i % 3 == 0:
            # add request
            new_req_id = allocate_req_id()
            seqlens[new_req_id] = utils.get_new_req_seq_len()
            active_ids.append(new_req_id)
            # remove request
            new_req_id = random.choice(active_ids)
            active_ids.remove(new_req_id)
            seqlens[new_req_id] = 0
            #print("freed req_id: ", new_req_id, flush=True)

        for id in active_ids:
            seqlens[id] += 1

        utils.do_matmul()

    # --- end of auto-regressive oop
    print(f"avg sync time pass 1 (ms): {round(total_sync_ms / (nr_steps - 1), 3)}", flush=True)

def do_kvcache_management_pass_2(kv_cache, nr_steps):
    total_sync_ms = 0
    for i in range(nr_steps):
        start = time.time()
        vattention.step_async(seqlens)
        end = time.time()
        sync_ms = round((end - start) * 1000, 3)
        if sync_ms > 1 or i % 100 == 0:
            mem_usage = get_mem_usage()
            print(f"step: {i} mem_usage(MB): {mem_usage} sync_time(ms): {sync_ms}", flush=True)

        utils.access_kv_cache(kv_cache, seqlens)
        total_sync_ms += sync_ms
        for id in active_ids:
            seqlens[id] += 1
        if i % 45 == 0:
            id = random.choice(active_ids)
            orig = seqlens[id]
            seqlens[id] = 0
            nr_tokens = sum(seqlens)
            #print(f"reducing from {orig} to 0. new nr_tokens: {nr_tokens}", flush=True)
        utils.do_matmul()
    print(f"avg sync time pass 2 (ms): {round(total_sync_ms / nr_steps, 3)}", flush=True)

def do_kvcache_management(kv_cache, nr_steps):
    # warm up the batch size
    while len(active_ids) < utils.MAX_BATCH_SIZE // 2:
        new_req_id = allocate_req_id()
        active_ids.append(new_req_id)
        seqlens[new_req_id] = utils.get_new_req_seq_len()

    print('***********************************************************************')
    do_kvcache_management_pass_1(kv_cache, nr_steps)
    do_kvcache_management_pass_2(kv_cache, nr_steps*100)
    # release memory and sync with the background thread
    utils.cleanup_kvcache_manager()

kv_cache = utils.init_kvcache()
do_kvcache_management(kv_cache, nr_steps=100)