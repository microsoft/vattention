static vAttentionCachingAllocator vattn;

std::vector<at::Tensor> init_kvcache(unsigned long num_layers, unsigned long num_kv_heads,
                        unsigned long head_size, unsigned long max_batch_size,
                        unsigned long max_context_length, int device,
                        py::object dtype, u64 page_size) {
    std::vector<at::Tensor> tensors;
    vattn.init_kvcache(num_layers, num_kv_heads, head_size,
                                    max_batch_size, max_context_length,
                                    device, dtype, page_size);
    tensors = vattn.init_kvcache_virtual();
    return tensors;
}

void show_kvcache_config() {
    vattn.show_kvcache_config();
}

void show_allocator_state() {
    vattn.show_allocator_state();
}

int reserve_physical_pages(u64 free_memory) {
    return vattn.reserve_physical_pages(free_memory);
}

void step(std::vector<u64> seq_lens, bool eager_reclaim) {
    vattn.step_sync(seq_lens, eager_reclaim);
}

void step_async(std::vector<u64> seq_lens)  {
    Py_BEGIN_ALLOW_THREADS
    vattn.step_async(seq_lens);
    Py_END_ALLOW_THREADS
}

void set_verbose(bool val) {
    verbose = val;
}

void cleanup() {
    vattn.cleanup();
}

void set_deferred_reclamation(bool val) {
    vattn.set_deferred_reclamation(val);
}

void map_common_pages(u64 num_tokens) {
    vattn.map_common_pages_in_batch(num_tokens);
}

int alloc_new_batch_idx(unsigned long seqlen) {
    return vattn.alloc_new_batch_idx(seqlen);
}

void free_batch_idx(int reqId) {
    vattn.free_batch_idx(reqId);
}

u64 num_free_kvblocks() {
    return vattn.num_free_kvblocks();
}