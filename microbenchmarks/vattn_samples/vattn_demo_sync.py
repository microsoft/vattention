import torch
import vattention
import time
import random
import utils

def do_kvcache_management(kv_cache, nr_steps):
    seqlens = [0 for i in range(utils.MAX_BATCH_SIZE)]
    active_ids = []

    # warm up the batch size
    while len(active_ids) < utils.MAX_BATCH_SIZE // 2:
        new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
        if new_req_id not in active_ids:
            active_ids.append(new_req_id)
            #seqlens[new_req_id] = random.randint(1, 1024)
            seqlens[new_req_id] = utils.INIT_SEQ_LEN

    total_sync_ms = 0
    # now, add/remove requests once every few steps to simulate a dynamic workload
    for i in range(nr_steps):
        # sync is necesaaary to ensure the background thread is in sync with the main thread
        # otherwise, we the background thread may free some memory that is accessed in the loop below
        torch.cuda.synchronize()
        start = time.time()
        vattention.step(seqlens, True)
        end = time.time()
        sync_ms = round((end - start) * 1000, 3)
        total_sync_ms += sync_ms
        if sync_ms > 1:
            print(f"step: {i} sync time (ms): {sync_ms}", flush=True)
        
        #ensure we can access the cache in each step
        utils.access_kv_cache(kv_cache, seqlens)
        if i % 3 == 0:
            # add request
            new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
            while new_req_id in active_ids:
                new_req_id = random.randint(0, utils.MAX_BATCH_SIZE-1)
            active_ids.append(new_req_id)
            #seqlens[new_req_id] = random.randint(1, 1024)
            seqlens[new_req_id] = utils.INIT_SEQ_LEN # fixed length to make the measurement more deterministic

            # remove request
            new_req_id = random.choice(active_ids)
            active_ids.remove(new_req_id)
            seqlens[new_req_id] = 0

        for id in active_ids:
            seqlens[id] += 1

        utils.do_matmul()

    # --- end of auto-regressive oop
    print(f"avg sync time (ms): {round(total_sync_ms / nr_steps, 3)}", flush=True)
    # release memory and sync with the background thread
    utils.cleanup_kvcache_manager()

kv_cache = utils.init_kvcache()
do_kvcache_management(kv_cache, nr_steps=1000)