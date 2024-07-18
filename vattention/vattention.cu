#include <torch/extension.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/ScalarType.h>
#include <ATen/ArrayRef.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/static_tracepoint.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <Python.h>
#include <utility>

#include <thread>
#include <atomic>

#include "common.h"
#include "logger.h"
#include "cudaInternal.h"
#include "vtensor.h"

std::vector<CUdeviceptr> k_ptr;
std::vector<CUdeviceptr> v_ptr;

std::atomic<bool> mem_manager_running(false);

class vAttentionCachingAllocator {
private:
    // model specifc params
    long num_kv_heads;
    long head_size;
    size_t num_layers;
    size_t bytes_per_elem;
    int device;
    py::object dtype;

    // framework specific params
    long max_batch_size;
    long max_context_length;

    // kv cache for the full model, one elem per layer
    std::vector<at::Tensor> k_tensors;
    std::vector<at::Tensor> v_tensors;
  
    // virtual memory metadata
    size_t virt_buff_size;
    size_t virt_buff_size_per_req;
    size_t virt_buff_size_per_token;
    size_t virt_buff_num_kv_tokens;
  
    // allocator and GPU specific data
    size_t tokens_per_page; // number of elements per physical block

    // a request can be allocated only this many pages per buffer
    size_t max_pages_per_req;

    // physical memory metadata
    std::vector<size_t> mapped_pages; // per request allocated block count
    //std::vector<size_t> new_req_seq_lengths; // seq lengths of requests joining this batch
    std::vector<size_t> curr_seq_lengths; // per request allocated block count

    // track physical memory handles (required while unmapping)
    std::map<std::tuple<size_t, size_t, size_t>, std::pair<CUHandle, CUHandle>> page_handles_map;
    std::map<std::tuple<size_t, size_t, size_t>, std::pair<NvU64, NvU64>> uvm_page_handles_map;

    // garbage collection thread to utilize slack
    std::thread gc_thread;
    Log log;

    // defer free optimization
    bool deferred_reclaim = true;
    // use uvm backend
    bool use_uvm_backend = false;

    /* custom virtual tensor allocator */
    VirtualTensorAllocator *allocator;

    /* whether kv cache has been configured yet */
    bool is_configured = false;

public:

    void init_kv_block_size() {
        page_size_bytes = do_cuda_init(device, use_uvm_backend);
        tokens_per_page = page_size_bytes / (num_kv_heads * head_size * bytes_per_elem);
        log.log("Initialized CUDA context and memory config etc...");
        log.log("num_tokens_per_kvblock: " + std::to_string(tokens_per_page));
    }

    void init_buffer_sizes() {
        size_t remainder;

        virt_buff_size_per_token = num_kv_heads * head_size * bytes_per_elem;
        virt_buff_size_per_req = virt_buff_size_per_token * max_context_length;

        assert(virt_buff_size_per_req % page_size_bytes == 0);
        max_pages_per_req = virt_buff_size_per_req / page_size_bytes;

        /* align to ensure that each request begins at a mappable offset */
        remainder = virt_buff_size_per_req % page_size_bytes;
        if (remainder != 0)
            virt_buff_size_per_req += page_size_bytes - remainder;

        virt_buff_size = virt_buff_size_per_req * max_batch_size;
        virt_buff_num_kv_tokens = max_batch_size * max_context_length;
        log.log("virt_buff_num_kv_tokens: " + std::to_string(virt_buff_num_kv_tokens));
        log.log("virt_buff_size_per_req: " + std::to_string(virt_buff_size_per_req / MB) + " MB");
        log.log("virt_buff_size: " + std::to_string(virt_buff_size / MB) + " MB");
    }

    void init_kvcache_batch_metadata() {
        mapped_pages.resize(max_batch_size);
        curr_seq_lengths.resize(max_batch_size);
        //new_req_seq_lengths.resize(max_batch_size);
        for(unsigned long i = 0; i < max_batch_size; i++) {
            curr_seq_lengths[i] = 0;
            mapped_pages[i] = 0;
        }
    }

    bool check_kvcache_config() const {
        return !(num_layers == 0 || num_kv_heads == 0 || head_size == 0 ||
                    max_batch_size == 0 || max_context_length == 0);
    }

    inline size_t tokens_to_pages(size_t num_tokens) const {
        return (num_tokens + tokens_per_page - 1) / tokens_per_page;
    }

    inline size_t pages_to_tokens(size_t num_blocks) const {
        return num_blocks * tokens_per_page;
    }

    inline size_t get_req_mapped_tokens(int reqId) const {
        return pages_to_tokens(mapped_pages[reqId]);
    }

    inline size_t get_req_pages(int reqId) const {
        return mapped_pages[reqId];
    }

    inline size_t inc_req_page_count(int reqId) {
        return ++mapped_pages[reqId];
    }

    inline size_t dec_req_page_count(int reqId) {
        return --mapped_pages[reqId];
    }

    inline size_t get_req_begin_offset_virt(int reqId) const {
        return reqId * virt_buff_size_per_req;
    }

    inline bool is_active_req(int reqId) const {
        return curr_seq_lengths[reqId] != 0;
    }

    inline size_t get_req_seq_length(int reqId) {
        return curr_seq_lengths[reqId];
    }

    inline void set_req_seq_length(int reqId, size_t seq_len) {
        curr_seq_lengths[reqId] = seq_len;
    }

    inline void set_curr_seq_lengths(std::vector<size_t> seq_lens) {
        curr_seq_lengths = seq_lens;
    }

    inline void wait_kvcache_manager_sync() {
        while (mem_manager_running);
    }

    inline void set_seq_lengths_for_next_step(std::vector<size_t> seq_lens) {
        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            if (!is_active_req(reqId))
                continue;

            set_req_seq_length(reqId, seq_lens[reqId] + 1);
        }
    }

    inline void show_allocator_state() {
        std::stringstream ss;
        size_t pages = !use_uvm_backend ? page_handles.size() : uvm_page_handles.size();
        log.log("Free pool: " + std::to_string(PAGES_TO_KVBLOCKS(pages)) + " KV blocks");
        log.log("reqId : seqlen: mapped: required");
        for (int i = 0; i < max_batch_size; i++) {
            ss.str(std::string());
            ss << std::setw(8) << i << ": "
                << std::setw(8) << get_req_seq_length(i) << " : "
                << std::setw(8) << mapped_pages[i] << " : "
                << std::setw(8) << tokens_to_pages(get_req_seq_length(i));
            log.log(ss.str());
        }
    }

    inline size_t get_req_current_map_offset(int reqId) const {
        int num_mapped_blocks = get_req_pages(reqId);
        size_t block_offset_within_req = num_mapped_blocks * page_size_bytes;

        return get_req_begin_offset_virt(reqId) + block_offset_within_req;
    }

    inline size_t get_req_current_unmap_offset(int reqId) const {
        int num_mapped_blocks;
        size_t block_offset_within_req;

        num_mapped_blocks = get_req_pages(reqId);
        /* unmap should not be called if nothing is mapped */
        assert(num_mapped_blocks > 0);
        /* return the offset to unmap, which is the beginning of the last mapped block */
        block_offset_within_req = (num_mapped_blocks-1) * page_size_bytes;
        return get_req_begin_offset_virt(reqId) + block_offset_within_req;
    }

    void init_kvcache(unsigned long num_layers_,
                            unsigned long num_kv_heads_,
                            unsigned long head_size_,
                            unsigned long max_batch_size_,
                            unsigned long max_context_length_,
                            int device_,
                            py::object dtype_,
                            bool use_uvm_backend_) {
        assert(max_batch_size_ > 0 && max_batch_size_ < 1000);
        assert(max_context_length_ > 0 && max_context_length_ < 1000000);
        assert(num_layers_ > 0 && num_layers_ < 100);
        assert(num_kv_heads_ > 0 && num_kv_heads_ < 256);
        num_layers = num_layers_;
        num_kv_heads = num_kv_heads_;
        head_size = head_size_;
        max_batch_size = max_batch_size_;
        max_context_length = max_context_length_;
        device = device_;
        dtype = dtype_;
        use_uvm_backend = use_uvm_backend_;
        bytes_per_elem = dtype.attr("itemsize").cast<int>();
        init_kv_block_size();
        init_buffer_sizes();
        init_kvcache_batch_metadata();
        k_ptr.resize(num_layers);
        v_ptr.resize(num_layers);
        allocator = new VirtualTensorAllocator(device, use_uvm_backend);
        is_configured = true;
    }

    void show_kvcache_config() {
        log.log("Num layers: " + std::to_string(num_layers));
        log.log("Num kv_heads: " + std::to_string(num_kv_heads));
        log.log("Head size: " + std::to_string(head_size));
        log.log("Max batch size: " + std::to_string(max_batch_size));
        log.log("Max context length: " + std::to_string(max_context_length));
        log.log("Bytes per elem: " + std::to_string(bytes_per_elem));
        log.log("virt_buff_size_per_req: " + std::to_string(virt_buff_size_per_req));
        log.log("virt_buff_size: " + std::to_string(virt_buff_size));
    }

    at::Tensor allocate_virtual_buffer() {
        at::ScalarType type_ = torch::python::detail::py_object_to_dtype(dtype);
        at::IntArrayRef shape = {max_batch_size, max_context_length, num_kv_heads, head_size};
        //at::Tensor t = at::detail::empty_cuda_virtual(shape, page_size_bytes, type_, device, use_uvm_backend);
        at::Tensor t = alloc_vtensor(shape, page_size_bytes, type_, allocator, device);

        return t;
    }

    std::vector<at::Tensor> init_kvcache_virtual() {
        if (!check_kvcache_config()) {
            log.log("Invalid kv cache configuration...");
            return std::vector<at::Tensor>();
        }

        for(int i = 0; i < 2 * num_layers; i++) {
            at::Tensor t = allocate_virtual_buffer();
            if (i < num_layers)
                k_tensors.push_back(t);
            else
                v_tensors.push_back(t);
        }
        std::vector<at::Tensor> tensors;
        tensors.insert(tensors.end(), k_tensors.begin(), k_tensors.end());
        tensors.insert(tensors.end(), v_tensors.begin(), v_tensors.end());
        return tensors;
    }

    int reserve_physical_pages(size_t free_memory) {
        return reserve_gpu_pages(num_layers, free_memory, use_uvm_backend);
    }

    inline size_t get_num_overcommitted_kvblocks() {
        size_t overcommitted_kvblocks = 0;
        for (int reqId = 0; reqId < max_batch_size; reqId++)
            overcommitted_kvblocks += mapped_pages[reqId] - tokens_to_pages(get_req_seq_length(reqId));
        return overcommitted_kvblocks;
    }

    inline size_t get_num_free_kvblocks() {
        size_t free_kvblocks, overcommitted_kvblocks;

        if (!use_uvm_backend)
            free_kvblocks = PAGES_TO_KVBLOCKS(page_handles.size());
        else
            free_kvblocks = PAGES_TO_KVBLOCKS(uvm_page_handles.size());

        overcommitted_kvblocks = get_num_overcommitted_kvblocks();
        return free_kvblocks + overcommitted_kvblocks;
    }

    /* check pages that are free as well as overcommitted */
    int can_allocate_new_sequence() {
        /* TODO(ashish): quick and dirty fix to let the system make forward progress in profiling pass */
        // if (!is_configured)
        //     // return true;
        //     return 1024288;

        /* add space for the first decode token */
        // size_t nr_blocks_required = PAGES_TO_KVBLOCKS(tokens_to_pages(seq_len + 1));
        size_t num_blocks_available = get_num_free_kvblocks();

        /* keep buffer for at least 1 block to avoid running into weird cases */
        // if (num_blocks_available > nr_blocks_required)
        //     return true;

        // return false;
        return num_blocks_available;
    }

    int nr_blocks_required(size_t seq_len) {
        if (!is_configured)
            return 0;
        return PAGES_TO_KVBLOCKS(tokens_to_pages(seq_len + 1));
    }

    inline bool kvblocks_available(size_t num_kvblocks) {
        size_t num_pages = KVBLOCKS_TO_PAGES(num_kvblocks);

        if (!use_uvm_backend)
            return page_handles.size() >= num_pages;
        else
            return uvm_page_handles.size() >= num_pages;
    }

    inline CUHandle pop_page_handle() {
        if (page_handles.empty())
            throw std::runtime_error("***** page pool is empty *****");

        CUHandle handle = page_handles.back();
        page_handles.pop_back();
        return handle;
    }

    inline NvU64 pop_uvm_page_handle() {
        if (uvm_page_handles.empty())
            throw std::runtime_error("***** uvm_page pool is empty *****");
        NvU64 handle = uvm_page_handles.back();
        uvm_page_handles.pop_back();
        return handle;
    }

    /*
    * Release blocks when a request exits
    * TODO: incorporate deferred freeing
    */
    inline void unmap_req_page_one(int reqId) {
        size_t req_offset;

        req_offset = get_req_current_unmap_offset(reqId);
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            CUdeviceptr k_data_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
            CUdeviceptr v_data_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
            if (!use_uvm_backend) {
                CHECK_CUDA(cuMemUnmap(k_data_ptr + req_offset, page_size_bytes));
                CHECK_CUDA(cuMemUnmap(v_data_ptr + req_offset, page_size_bytes));
                std::pair p_handles = page_handles_map[std::make_tuple(reqId, req_offset, layer_idx)];
                page_handles.push_back(p_handles.first);
                page_handles.push_back(p_handles.second);
                page_handles_map.erase(std::make_tuple(reqId, req_offset, layer_idx));
            } else {
                // just give back the handles --- no need to unmap
                std::pair p_handles = uvm_page_handles_map[std::make_tuple(reqId, req_offset, layer_idx)];
                uvm_page_handles.push_back(p_handles.first);
                uvm_page_handles.push_back(p_handles.second);
                uvm_page_handles_map.erase(std::make_tuple(reqId, req_offset, layer_idx));
            }
        }
        dec_req_page_count(reqId);
    }

    inline void release_kvcache_pages_some(int reqId, size_t retain_blocks) {
        while(get_req_pages(reqId) > retain_blocks)
            unmap_req_page_one(reqId);
    }

    inline void release_kvcache_pages_all(int reqId) {
        release_kvcache_pages_some(reqId, 0);
    }

    inline void map_page_handles(int reqId,
                                    int layer_idx,
                                    size_t req_offset,
                                    CUdeviceptr k_data_ptr,
                                    CUdeviceptr v_data_ptr,
                                    CUHandle k_handle,
                                    CUHandle v_handle) {
        /*
        * TODO(ashish): Since we are doing eager allocation, we shoud simply return if we are
        * trying to map beyond the allocated memory of a request. In other cases, we should simply
        * raise an exception.
        */
        if (req_offset >= (reqId + 1) * virt_buff_size_per_req)
            std::runtime_error("***** [Unexpected] request has already received max number of pages *****");

        CHECK_CUDA(cuMemMap(k_data_ptr + req_offset, page_size_bytes, 0, k_handle, 0));
        CHECK_CUDA(cuMemMap(v_data_ptr + req_offset, page_size_bytes, 0, v_handle, 0));
        CHECK_CUDA(cuMemSetAccess(k_data_ptr + req_offset, page_size_bytes, &accessDesc, 1));
        CHECK_CUDA(cuMemSetAccess(v_data_ptr + req_offset, page_size_bytes, &accessDesc, 1));
        page_handles_map[std::make_tuple(reqId, req_offset, layer_idx)] = std::make_pair(k_handle, v_handle);
    }

    inline int map_uvm_page_handles(int reqId,
                                      int layer_idx,
                                      size_t req_offset,
                                      CUdeviceptr k_data_ptr,
                                      CUdeviceptr v_data_ptr,
                                      NvU64 k_handle,
                                      NvU64 v_handle) {
        /*
        * TODO(ashish): Since we are doing eager allocation, we shoud simply return if we are
        * trying to map beyond the allocated memory of a request. In other cases, we should simply
        * raise an exception.
        */
        if (req_offset >= (reqId + 1) * virt_buff_size_per_req)
            return -1;

        CHECK_VATTN(vattn_mem_map((void*)(k_data_ptr + req_offset), k_handle));
        CHECK_VATTN(vattn_mem_map((void*)(v_data_ptr + req_offset), v_handle));
        uvm_page_handles_map[std::make_tuple(reqId, req_offset, layer_idx)] = std::make_pair(k_handle, v_handle);
        return 0;
    }

    /*
    * Grow KV Cache phys allocation by num_blocks
    */
    void grow_kvcache_phys(int reqId, int num_blocks, bool sync) {
        size_t req_offset;
        at::Storage k_storage, v_storage;

        if (num_blocks <= 0)
            return;

        if(!kvblocks_available(num_blocks)) {
            /* we do not need to raise an exception if memory allocation is
             * being done by the background thread.
             */
            if (!sync)
                return;

            /* there is no other option but to abort */
            verbose = true;
            log.log("free handles: " + std::to_string(PAGES_TO_KVBLOCKS(page_handles.size())));
            log.log("required: " + std::to_string(num_blocks));
            show_allocator_state();
            throw std::runtime_error("***** OOM on demand: not enough free pages to continue *****");
            return;
        }

        for (int count = 0; count < num_blocks; count++) {
            /*
            * Go through all the layers for a single block first so that offset computation
            * can be reused.
            */
            req_offset = get_req_current_map_offset(reqId);
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
                CUdeviceptr k_data_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
                CUdeviceptr v_data_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
                if (!use_uvm_backend) {
                    CUHandle k_handle = pop_page_handle();
                    CUHandle v_handle = pop_page_handle();
                    /*
                    TODO(ashish): in some cases when we are proactively allocating, we might end up
                    attempting to overallocate. We should catch such cases and exit this loop on the first
                    layer's attempt itself.
                    */
                    map_page_handles(reqId, layer_idx, req_offset, k_data_ptr, v_data_ptr, k_handle, v_handle);
                } else {
                    NvU64 k_handle = pop_uvm_page_handle();
                    NvU64 v_handle = pop_uvm_page_handle();
                    map_uvm_page_handles(reqId, layer_idx, req_offset, k_data_ptr, v_data_ptr, k_handle, v_handle);
                }
            }
            inc_req_page_count(reqId);
        }
    }

    /*
    * Test function to map the same physical handles in all active requests
    */
    void map_common_handles_in_batch(size_t num_tokens) {
        size_t req_offset;
        size_t num_blocks;
        at::Storage k_storage, v_storage;

        num_blocks = tokens_to_pages(num_tokens);
        if (num_blocks <= 0)
            return;

        if(!kvblocks_available(num_blocks)) {
            log.log("free handles: " + std::to_string(PAGES_TO_KVBLOCKS(page_handles.size())));
            log.log("required: " + std::to_string(num_blocks));
            throw std::runtime_error("***** OOM on demand: not enough free pages to continue *****");
            return;
        }

        for (int count = 0; count < num_blocks; count++) {
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
                CUdeviceptr k_data_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
                CUdeviceptr v_data_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
                if (!use_uvm_backend) {
                    CUHandle k_handle = pop_page_handle();
                    CUHandle v_handle = pop_page_handle();
                    for (int reqId = 0; reqId < max_batch_size; reqId++) {
                        req_offset = get_req_current_map_offset(reqId);
                        map_page_handles(reqId, layer_idx, req_offset, k_data_ptr, v_data_ptr, k_handle, v_handle);
                    }
                } else {
                    NvU64 k_handle = pop_uvm_page_handle();
                    NvU64 v_handle = pop_uvm_page_handle();
                    for (int reqId = 0; reqId < max_batch_size; reqId++) {
                        req_offset = get_req_current_map_offset(reqId);
                        map_uvm_page_handles(reqId, layer_idx, req_offset, k_data_ptr, v_data_ptr, k_handle, v_handle);
                    }
                }
            }
            for (int reqId = 0; reqId < max_batch_size; reqId++)
                inc_req_page_count(reqId);
        }
    }

    /*
    * This would reclaim pages if not enough are already available
    */
    void alloc_prefill_kvcache(int reqId, size_t seq_len) {
        int nr_required = tokens_to_pages(seq_len);
        int nr_mapped = get_req_pages(reqId);

        if (nr_required <= nr_mapped)
            return;

        nr_required -= nr_mapped;
        if (!kvblocks_available(nr_required))
            reclaim_kvblocks_on_demand(nr_required);

        /* this should not get triggered frequently with the deferred reclaim optimization */
        log.log("[DEBUG] allocating " + std::to_string(nr_required) + " pages for reqId: " + std::to_string(reqId));
        grow_kvcache_phys(reqId, nr_required, true);
        set_req_seq_length(reqId, seq_len);
    }

    /*
    * This is not meant for final use but to test simple cases with eager allocation and free
    */
    void single_step_sync(std::vector<size_t> seq_lens, bool eager_reclaim) {
        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            /* set the sequence length based on latest values coming from the model */
            set_req_seq_length(reqId, seq_lens[reqId]);
            /* req not active but physical memory blocks are allocated */
            if (eager_reclaim == true and (seq_lens[reqId] == 0 && mapped_pages[reqId] != 0)) {
                release_kvcache_pages_all(reqId);
                continue;
            }
            alloc_prefill_kvcache(reqId, seq_lens[reqId]);
        }
    }

    /*
    * Ensure that we have enough pages for each new sequence
    */
    void prepare_prefill_kvcache() {
        for(int reqId = 0; reqId < max_batch_size; reqId++) {
            alloc_prefill_kvcache(reqId, get_req_seq_length(reqId));
        }
    }

    int need_new_page_async(int reqId, int eager_step_count) {
        if (!is_active_req(reqId))
            return 0;

        /* do not attempt allocation if a request already has been allocated max num pages */
        int nr_mapped = get_req_pages(reqId);
        if (nr_mapped == max_pages_per_req)
            return 0;

        /* add eager_step_count to sequence length to allocate memory proactively */
        int nr_required = tokens_to_pages(get_req_seq_length(reqId) + eager_step_count);

        if (nr_required <= nr_mapped)
            return 0;

        return nr_required - nr_mapped;
    }

    void reclaim_kvblocks_on_demand(int num_kvblocks) {
        int nr_mapped, nr_required, nr_extra;

        /* Now we get into relaim mode */
        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--) {
            /* demand fulfilled */
            if (kvblocks_available(num_kvblocks))
                break;

            nr_mapped = get_req_pages(reqId);
            nr_required = tokens_to_pages(get_req_seq_length(reqId));
            /* reclaim only as much as needed */
            nr_extra = MIN(nr_mapped - nr_required, num_kvblocks);
            if (nr_extra <= 0)
                continue;

            release_kvcache_pages_some(reqId, nr_mapped - nr_extra);
        }
    }

    /*
    * Release one page at a time (in the order opposite to how we allocated a new request id)
    * We do not reclaim from the req id that we know is going to be allocated soon
    */
    void do_deferred_reclaim() {
        if (!deferred_reclaim)
            return;

        int next_prefill_reqId = -1;

        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            if (!is_active_req(reqId)) {
                    next_prefill_reqId = reqId;
                    break;
            }
        }

        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--) {
            if (is_active_req(reqId) || reqId == next_prefill_reqId)
                continue;
            if (get_req_pages(reqId) == 0)
                continue;
            unmap_req_page_one(reqId);
            break;
        }
    }

   /*
   * 1. Check how many new pages will be required in the next step
   * 2. Ensure that the free pool is large enough, reclaim memory if required
   * 3. Allocate memory for the next step.
   *
   * If new blocks are not required, we free one block in an iteration (in the decreasing order of reqIds)
   * If new blocks are required, free as many blocks as required + some more so that next prefill can also be handled
   * TODO: check corner-cases e.g., what if only one inactive reqId has pages mapped? we could probably not relaim in
   * this case so that next prefill can re-use already allocated memory.
   */

   /* TODO(ashish): fix/update these heuristics, if required */
   #define EAGER_NUM_STEPS      (20)
   #define EAGER_NUM_KVBLOCKS   (1)
    void do_kvcache_memory_management() {
        size_t nr_required = 0;
        int nr_mapped_curr = 0;
        bool done = false;

        for (int reqId = 0; reqId < max_batch_size; reqId++)
            nr_required += need_new_page_async(reqId, 1);

        /* reclaim if required */
        if (!kvblocks_available(nr_required)) {
            log.log("[DEBUG] reclaiming " + std::to_string(nr_required) + " KV blocks in background thread...");
            reclaim_kvblocks_on_demand(nr_required);
        }

        /*
         * check if we have enough free pages to continue.
         * if not, we return without doing anything, hoping that memory will become
         * available when a request exits. raising an OOM exception here would be too aggressive.
         */

        if (!kvblocks_available(nr_required))
            return;

        for (int eager_step_count = 1; eager_step_count < EAGER_NUM_STEPS && !done; eager_step_count++) {
            for (int reqId = 0; reqId < max_batch_size; reqId++) {
                int nr_required_curr = need_new_page_async(reqId, eager_step_count);
                grow_kvcache_phys(reqId, nr_required_curr, false);
                nr_mapped_curr += nr_required_curr;
                if (eager_step_count == 1)
                    continue;
                if (nr_mapped_curr >= EAGER_NUM_KVBLOCKS) {
                    done = true;
                    break;
                }
            }
        }

        /* return if we just allocated one or more pages */
        if (nr_required)
            return;

        /* deferred reclaim */
        do_deferred_reclaim();
    }

    void spawn_kvcache_manager() {
        std::thread([this]() {
            mem_manager_running = true;
            do_kvcache_memory_management();
            mem_manager_running = false;
        }).detach();
    }

    /* single step for asynchronous allocation */
    void step_async(std::vector<size_t> seq_lens) {
        set_curr_seq_lengths(seq_lens);
        /* synchronize with the background thread first */
        wait_kvcache_manager_sync();
        /* allocate prefill memory synchronously, if required */
        prepare_prefill_kvcache();
        /* allocate decode memory for next step asynchronously */
        spawn_kvcache_manager();
    }

    /*
    * Return one of the inactive ids using best fit.
    * NOTE: the caller is supposed to check if the returned reqId is valid or not
    */
    int alloc_new_batch_idx(unsigned long seqlen) {
        int new_id = -1;
        int nr_required = tokens_to_pages(seqlen);

        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            if (is_active_req(reqId))
                continue;

            if (new_id == -1) {
                new_id = reqId;
                continue;
            }

            if (get_req_pages(reqId) >= nr_required &&
                get_req_pages(reqId) < get_req_pages(new_id))
                new_id = reqId;
        }

        if (new_id != -1)
            set_req_seq_length(new_id, seqlen);

        return new_id;
    }

    void free_batch_idx(int reqId) {
        set_req_seq_length(reqId, 0);
    }

    void set_deferred_reclamation(bool val) {
        deferred_reclaim = val;
    }

    /* cleanup code */
    void do_uvm_kvcache_cleanup() {
        // NOTE: this function must be called after wait_kvcache_manager_sync
        // remove the physical blocks from requests if they are holding any
        for (int reqId = 0; reqId < max_batch_size; reqId++) {
            release_kvcache_pages_all(reqId);
        }
        // releasing physical handles
        for(int i = 0; i < uvm_page_handles.size(); i++) {
            CHECK_VATTN(vattn_release_mem_handle(uvm_page_handles[i]));
        }
        // cleaning up k_tensor and v_tensors (they are equal in count)
        size_t nelements = (max_batch_size * max_context_length * num_kv_heads * head_size);
        for(int j = 0; j < k_tensors.size(); j++) {
            CHECK_VATTN(vattn_free_reserved_address((void*)(k_tensors[j].data_ptr()), k_tensors[j].element_size() * nelements));
            CHECK_VATTN(vattn_free_reserved_address((void*)(v_tensors[j].data_ptr()), v_tensors[j].element_size() * nelements));
        }
    }

    void do_cuda_kvcache_cleanup() {
        for (int reqId = 0; reqId < max_batch_size; reqId++)
            release_kvcache_pages_all(reqId);

        for (int i = 0; i < k_tensors.size(); i++) {
            CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr()), virt_buff_size));
            CHECK_CUDA(cuMemUnmap(reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr()), virt_buff_size));
            CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(k_tensors[i].data_ptr()), virt_buff_size));
            CHECK_CUDA(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(v_tensors[i].data_ptr()), virt_buff_size));
        }

        for(int i = 0; i < page_handles.size(); i++)
            CHECK_CUDA(cuMemRelease(page_handles[i]));
    }

    /* TODO(ashish): check if this is compatible with PyTorch destructor */
    void cleanup() {
        wait_kvcache_manager_sync();
        if (use_uvm_backend) {
            do_uvm_kvcache_cleanup();
            return;
        }
        do_cuda_kvcache_cleanup();
        k_tensors.clear();
        v_tensors.clear();
        page_handles.clear();
        uvm_page_handles.clear();
        log.log("released memory and cleaned up vattention ...");
    }
};

static vAttentionCachingAllocator vattn;

std::vector<at::Tensor> init_kvcache(unsigned long num_layers, unsigned long num_kv_heads,
                        unsigned long head_size, unsigned long max_batch_size,
                        unsigned long max_context_length, int device,
                        py::object dtype, bool use_uvm_backend=false) {
    std::vector<at::Tensor> tensors;
    vattn.init_kvcache(num_layers, num_kv_heads, head_size,
                                    max_batch_size, max_context_length,
                                    device, dtype, use_uvm_backend);
    tensors = vattn.init_kvcache_virtual();
    return tensors;
}

void show_kvcache_config() {
    vattn.show_kvcache_config();
}

void show_allocator_state() {
    vattn.show_allocator_state();
}

int reserve_physical_pages(size_t free_memory) {
    return vattn.reserve_physical_pages(free_memory);
}

void step(std::vector<size_t> seq_lens, bool eager_reclaim) {
    vattn.single_step_sync(seq_lens, eager_reclaim);
}

void step_async(std::vector<size_t> seq_lens)  {
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

void map_common_handles(size_t num_tokens) {
    vattn.map_common_handles_in_batch(num_tokens);
}

int alloc_new_batch_idx(unsigned long seqlen) {
    return vattn.alloc_new_batch_idx(seqlen);
}

void free_batch_idx(int reqId) {
    vattn.free_batch_idx(reqId);
}

int can_allocate_new_sequence() {
    return vattn.can_allocate_new_sequence();
}

int nr_blocks_required(size_t seq_len) {
    return vattn.nr_blocks_required(seq_len);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /*
    * These must be invoked during initialization/termination
    * TODO(ashish): Merge these into one API?
    */
    m.def("reserve_physical_pages", &reserve_physical_pages, "reserve physical memory blocks...");
    m.def("init_kvcache", &init_kvcache, "initialize KV cache...");
    m.def("cleanup", &cleanup, "cleanup and release allocator resources context...");
    /* Tunables and other helper APIs */
    m.def("set_verbose", &set_verbose, "to enable/disable printing logs...");
    m.def("set_deferred_reclamation", &set_deferred_reclamation, "enable/disable deferred freeing...");
    /* Testing APIs */
    m.def("show_kvcache_config", &show_kvcache_config, "show kv cache configuration...");
    m.def("show_allocator_state", &show_allocator_state, "show free pool of physical memory blocks...");
    m.def("step", &step, "step function...");
    m.def("map_common_handles", &map_common_handles, "map common handles in batch...");
    /* API for actual physical memory allocation - one call per iteration */
    m.def("step_async", &step_async, "single step function for the async version...");
    /* Request-level APIs */
    m.def("alloc_new_batch_idx", &alloc_new_batch_idx, "allocate a request id...");
    m.def("free_batch_idx", &free_batch_idx, "free a request id...");
    m.def("can_allocate_new_sequence", &can_allocate_new_sequence, "check if we can allocate pages for a new request...");
    m.def("nr_blocks_required", &nr_blocks_required, "number of blocks required for a sequence...");
}
