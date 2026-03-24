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
#include <string>

#include <thread>
#include <atomic>

#include "utils.h"
#include "uvmInternal.h"
#include "cudaInternal.h"
#include "mux.h"
#include "vtensor.h"

class vAttentionCachingAllocator
{
private:
    /* whether kv cache has been configured yet */
    bool is_configured = false;
    /* custom virtual tensor allocator */
    VirtualTensorAllocator *allocator;
    Log log;

public:
    bool megacache_enabled;
    void init_kv_block_size()
    {
        page_size = do_cuda_init(device, page_size);
        if (component_spec_enabled)
        {
            if (tokens_per_page == 0)
                tokens_per_page = page_size / page_buffer_token_bytes;
        }
        else if (megacache_enabled)
            tokens_per_page = page_size / (num_kv_heads * head_size * bytes_per_elem * num_layers);
        else
            tokens_per_page = page_size / (num_kv_heads * head_size * bytes_per_elem);
        log.log("Initialized CUDA context and memory config etc...");
        log.log("num_tokens_per_kvblock: " + std::to_string(tokens_per_page));
    }

    void init_buffer_sizes()
    {
        u64 remainder;

        if (component_spec_enabled)
            virt_buff_size_per_token = page_buffer_token_bytes;
        else if (megacache_enabled)
            virt_buff_size_per_token = num_kv_heads * head_size * bytes_per_elem * num_layers;
        else
            virt_buff_size_per_token = num_kv_heads * head_size * bytes_per_elem;
        virt_buff_size_per_req = virt_buff_size_per_token * max_context_length;
        virt_buff_size_per_req = ROUND_UP(virt_buff_size_per_req, page_size);

        max_pages_per_req = virt_buff_size_per_req / page_size;

        /* align to ensure that each request begins at a mappable offset */
        remainder = virt_buff_size_per_req % page_size;
        if (remainder != 0)
            virt_buff_size_per_req += page_size - remainder;

        virt_buff_size = virt_buff_size_per_req * max_batch_size;
        virt_buff_num_kv_tokens = max_batch_size * max_context_length;
        if (megacache_enabled)
            virt_buff_num_kv_tokens *= num_layers;
        log.log("virt_buff_num_kv_tokens: " + std::to_string(virt_buff_num_kv_tokens));
        log.log("virt_buff_size_per_req: " + std::to_string(virt_buff_size_per_req / MB) + " MB");
        log.log("virt_buff_size: " + std::to_string(virt_buff_size / MB) + " MB");
    }

    inline void show_allocator_state()
    {
        std::stringstream ss;
        u64 nr_pages = !is_uvm_backend(page_size) ? cuda_pages.size() : uvm_pages.size();
        log.log("Free pool: " + std::to_string(pages_to_kvblocks(nr_pages)) + " KV blocks");

        log.log("reqId : seqlen: mapped: required");
        for (int i = 0; i < max_batch_size; i++)
        {
            ss.str(std::string());
            ss << std::setw(8) << i << ": "
               << std::setw(8) << get_req_seq_length(i) << " : "
               << std::setw(8) << mapped_pages[i] << " : "
               << std::setw(8) << tokens_to_pages(get_req_seq_length(i));
            log.log(ss.str());
        }
    }

    void init_kvcache(int num_layers_,
                      int num_kv_heads_,
                      int head_size_,
                      int max_batch_size_,
                      long max_context_length_,
                      int device_,
                      py::object dtype_,
                      u64 page_size_,
                      bool megacache)
    {
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
        scalar_type = torch::python::detail::py_object_to_dtype(dtype_);
        page_size = page_size_;
        megacache_enabled = megacache;
        megacache_enabled_global = megacache;
        bytes_per_elem = dtype.attr("itemsize").cast<int>();
        page_buffer_token_bytes = megacache_enabled
            ? (u64)num_kv_heads * head_size * bytes_per_elem * num_layers
            : (u64)num_kv_heads * head_size * bytes_per_elem;
        cached_token_bytes_local = (u64)num_layers * 2 * num_kv_heads * head_size * bytes_per_elem;
        component_spec_enabled = false;
        cache_component_token_dims = {
            num_kv_heads * head_size,
            num_kv_heads * head_size,
        };
        init_kv_block_size();
        init_buffer_sizes();
        init_kvcache_batch_metadata();
        k_ptr.resize(num_layers);
        v_ptr.resize(num_layers);
        allocator = new VirtualTensorAllocator(device, page_size);
        is_configured = true;
    }

    static at::ScalarType parse_scalar_type(const std::string &dtype_name)
    {
        if (dtype_name == "float16" || dtype_name == "half")
            return at::ScalarType::Half;
        if (dtype_name == "bfloat16")
            return at::ScalarType::BFloat16;
        if (dtype_name == "float32" || dtype_name == "float")
            return at::ScalarType::Float;
        throw std::runtime_error("Unsupported dtype for component-spec initialization: " + dtype_name);
    }

    void init_kvcache_component_spec(py::dict payload)
    {
        py::dict cache_spec = payload["cache_spec"].cast<py::dict>();
        py::list cache_components = cache_spec["cache_components"].cast<py::list>();

        max_batch_size = payload["max_batch_size"].cast<int>();
        max_context_length = payload["max_context_length"].cast<long>();
        device = payload["device_idx"].cast<int>();
        page_size = cache_spec["page_size"].cast<u64>();
        megacache_enabled = cache_spec["megacache"].cast<bool>();
        megacache_enabled_global = megacache_enabled;
        num_layers = cache_spec["num_layers"].cast<int>();
        num_kv_heads = cache_spec["num_kv_heads"].cast<int>();
        head_size = cache_spec["head_size"].cast<int>();
        tokens_per_page = cache_spec["tokens_per_page"].cast<u64>();
        page_buffer_token_bytes = cache_spec["page_buffer_token_bytes"].cast<u64>();
        cached_token_bytes_local = cache_spec["cached_token_bytes_local"].cast<u64>();
        bytes_per_elem = cache_spec["dtype_size"].cast<int>();
        scalar_type = parse_scalar_type(payload["dtype"].cast<std::string>());
        component_spec_enabled = true;
        cache_component_token_dims.clear();
        for (auto component_obj : cache_components)
        {
            py::dict component = component_obj.cast<py::dict>();
            cache_component_token_dims.push_back(component["token_dim"].cast<int>());
        }
        if (cache_component_token_dims.size() != 2)
            throw std::runtime_error("component-spec initialization currently expects exactly 2 cache components");

        init_kv_block_size();
        init_buffer_sizes();
        init_kvcache_batch_metadata();
        k_ptr.resize(num_layers);
        v_ptr.resize(num_layers);
        allocator = new VirtualTensorAllocator(device, page_size);
        is_configured = true;
    }

    void show_kvcache_config()
    {
        log.log("Num layers: " + std::to_string(num_layers));
        log.log("Num kv_heads: " + std::to_string(num_kv_heads));
        log.log("Head size: " + std::to_string(head_size));
        log.log("Max batch size: " + std::to_string(max_batch_size));
        log.log("Max context length: " + std::to_string(max_context_length));
        log.log("Bytes per elem: " + std::to_string(bytes_per_elem));
        log.log("virt_buff_size_per_req: " + std::to_string(virt_buff_size_per_req));
        log.log("virt_buff_size: " + std::to_string(virt_buff_size));
    }

    at::Tensor alloc_virtual_tensor()
    {
        at::ScalarType type_ = scalar_type;
        at::IntArrayRef shape;
        if (megacache_enabled)
            shape = {max_batch_size, max_context_length, num_layers, num_kv_heads, head_size};
        else
            shape = {max_batch_size, max_context_length, num_kv_heads, head_size};

        at::Tensor t = alloc_vtensor(shape, page_size, type_, allocator, device);
        return t;
    }

    at::Tensor alloc_component_virtual_tensor(int component_token_dim)
    {
        at::IntArrayRef shape;
        if (megacache_enabled)
            shape = {max_batch_size, max_context_length, num_layers, component_token_dim};
        else
            shape = {max_batch_size, max_context_length, component_token_dim};

        at::Tensor t = alloc_vtensor(shape, page_size, scalar_type, allocator, device);
        return t;
    }

    std::vector<at::Tensor> init_kvcache_virtual()
    {
        if (!check_kvcache_config())
        {
            log.log("Invalid kv cache configuration...");
            return std::vector<at::Tensor>();
        }

        if (megacache_enabled)
        {
            at::Tensor t0 = alloc_virtual_tensor();
            at::Tensor t1 = alloc_virtual_tensor();
            k_tensors.push_back(t0);
            v_tensors.push_back(t1);
        }
        else
        {
            for (int i = 0; i < 2 * num_layers; i++)
            {
                at::Tensor t = alloc_virtual_tensor();
                if (i < num_layers)
                {
                    k_tensors.push_back(t);
                    continue;
                }
                v_tensors.push_back(t);
            }
        }
        std::vector<at::Tensor> tensors;
        tensors.insert(tensors.end(), k_tensors.begin(), k_tensors.end());
        tensors.insert(tensors.end(), v_tensors.begin(), v_tensors.end());
        return tensors;
    }

    std::vector<at::Tensor> init_kvcache_component_spec_virtual()
    {
        if (!check_kvcache_config())
        {
            log.log("Invalid component-spec kv cache configuration...");
            return std::vector<at::Tensor>();
        }

        if (megacache_enabled)
        {
            at::Tensor t0 = alloc_component_virtual_tensor(cache_component_token_dims[0]);
            at::Tensor t1 = alloc_component_virtual_tensor(cache_component_token_dims[1]);
            k_tensors.push_back(t0);
            v_tensors.push_back(t1);
        }
        else
        {
            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
            {
                k_tensors.push_back(alloc_component_virtual_tensor(cache_component_token_dims[0]));
                v_tensors.push_back(alloc_component_virtual_tensor(cache_component_token_dims[1]));
            }
        }
        std::vector<at::Tensor> tensors;
        tensors.insert(tensors.end(), k_tensors.begin(), k_tensors.end());
        tensors.insert(tensors.end(), v_tensors.begin(), v_tensors.end());
        return tensors;
    }

    u64 reserve_physical_pages(u64 free_memory)
    {
        return reserve_gpu_pages(free_memory, page_size);
    }

    inline u64 get_num_free_kvblocks()
    {
        u64 free_kvblocks;
        free_kvblocks = pages_to_kvblocks(get_num_free_pages(page_size));

        return free_kvblocks + get_num_overcommitted_kvblocks();
    }

    /* check pages that are free as well as overcommitted */
    u64 num_free_kvblocks()
    {
        return get_num_free_kvblocks();
    }

    inline bool kvblocks_available(u64 num_kvblocks)
    {
        return pages_to_kvblocks(get_num_free_pages(page_size)) >= num_kvblocks ? true : false;
    }

    inline void unmap_req_page_one(int reqId)
    {
        u64 req_offset;
        req_offset = get_req_current_unmap_offset(reqId);
        if (megacache_enabled)
        {
            CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[0].data_ptr());
            CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[0].data_ptr());
            UNMAP_PAGES(reqId, 0, req_offset, kcache_ptr, vcache_ptr, page_size);
            dec_req_page_count(reqId);
        }
        else
        {

            for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
            {
                CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
                CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
                UNMAP_PAGES(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, page_size);
            }
            dec_req_page_count(reqId);
        }
    }

    inline void release_kvcache_pages_some(int reqId, u64 retain_blocks)
    {
        while (get_req_pages(reqId) > retain_blocks)
            unmap_req_page_one(reqId);
    }

    inline void release_kvcache_pages_all(int reqId)
    {
        release_kvcache_pages_some(reqId, 0);
    }

    inline bool is_valid_offset(int reqId, u64 req_offset, bool sync)
    {
        if (req_offset < (reqId + 1) * virt_buff_size_per_req)
            return true;

        /* for async allocation attempts, it is enough to simply return */
        if (!sync)
            return false;

        std::runtime_error("***** [Unexpected] request has already received max number of pages *****");
        return false;
    }

    /* Grow KV cache physical memory allocation by num_blocks */
    void grow_kvcache_phys(int reqId, u64 num_blocks, bool sync)
    {
        u64 req_offset;
        at::Storage k_storage, v_storage;

        if (num_blocks > 0) {
            printf("[vAttention ALLOC] reqId: %d | blocks: %llu | sync: %s\n", 
               reqId, (unsigned long long)num_blocks, sync ? "true" : "false");
        }

        if (num_blocks <= 0)
            return;

        if (!kvblocks_available(num_blocks))
        {
            /* no-op if this is being called by the background thread. */
            if (!sync)
                return;

            /* there is no other option but to abort */
            verbose = true;
            if (megacache_enabled)
            {
                log.log("free pages: " + std::to_string(pages_to_kvblocks(cuda_pages.size())));
                log.log("required: " + std::to_string(num_blocks));
            }
            else
            {
                log.log("free pages: " + std::to_string(pages_to_kvblocks(cuda_pages.size())));
                log.log("required: " + std::to_string(num_blocks));
            }
            show_allocator_state();
            throw std::runtime_error("***** OOM on demand: not enough free pages to continue *****");
            return;
        }

        for (int count = 0; count < num_blocks; count++)
        {
            req_offset = get_req_current_map_offset(reqId);
            if (!is_valid_offset(reqId, req_offset, sync))
                return;

            if (megacache_enabled)
            {
                CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[0].data_ptr());
                CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[0].data_ptr());
                MAP_PAGES(reqId, 0, req_offset, kcache_ptr, vcache_ptr, page_size);
                inc_req_page_count(reqId);
            }
            else
            {
                for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
                {
                    CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
                    CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
                    MAP_PAGES(reqId, layer_idx, req_offset, kcache_ptr, vcache_ptr, page_size);
                }
                inc_req_page_count(reqId);
            }
        }
    }

    /* This is only to validate that same physical page can be given to multiple requests */
    void map_common_pages_in_batch(u64 num_tokens)
    {
        u64 req_offset, num_blocks;
        at::Storage k_storage, v_storage;

        num_blocks = tokens_to_pages(num_tokens);
        if (num_blocks <= 0)
            return;

        if (!kvblocks_available(num_blocks))
        {
            if (megacache_enabled)
            {
                log.log("free pages: " + std::to_string(pages_to_kvblocks(cuda_pages.size())));
                log.log("required: " + std::to_string(num_blocks));
            }
            else
            {
                log.log("free pages: " + std::to_string(pages_to_kvblocks(cuda_pages.size())));
                log.log("required: " + std::to_string(num_blocks));
            }
            throw std::runtime_error("***** OOM on demand: not enough free pages to continue *****");
            return;
        }

        for (int count = 0; count < num_blocks; count++)
        {
            if (megacache_enabled)
            {
                CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[0].data_ptr());
                CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[0].data_ptr());
                MAP_COMMON_PAGES(0, kcache_ptr, vcache_ptr, page_size);
                for (int reqId = 0; reqId < max_batch_size; reqId++)
                    inc_req_page_count(reqId);
            }
            else
            {
                for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
                {
                    CUdeviceptr kcache_ptr = reinterpret_cast<CUdeviceptr>(k_tensors[layer_idx].data_ptr());
                    CUdeviceptr vcache_ptr = reinterpret_cast<CUdeviceptr>(v_tensors[layer_idx].data_ptr());
                    MAP_COMMON_PAGES(layer_idx, kcache_ptr, vcache_ptr, page_size);
                }
                for (int reqId = 0; reqId < max_batch_size; reqId++)
                    inc_req_page_count(reqId);
            }
        }
    }

    /* Map physical memory for the current iteration before returning control */
    void map_pages_for_curr_step(int reqId, u64 seq_len)
    {
        if (seq_len == 0) return;

        u64 nr_required = tokens_to_pages(seq_len);
        u64 nr_mapped = get_req_pages(reqId);
        
        // --- START FRAGMENTATION LOGIC ---
        
        // 1. Calculate physical token capacity in currently mapped logical kvblocks.
        u64 physical_token_capacity = nr_mapped * tokens_per_page;

        // 2. Calculate "Resident" Useful Tokens
        // We only count tokens actually residing in the currently mapped physical blocks.
        u64 resident_useful_tokens = (seq_len < physical_token_capacity) ? seq_len : physical_token_capacity;

        // 3. Convert to bytes for the full resident payload tracked by the allocator.
        u64 bytes_useful = resident_useful_tokens * cached_token_bytes_local;

        // 4. Calculate total physical memory mapped for the full logical kvblocks.
        u64 bytes_allocated = nr_mapped * get_pages_per_kvblock() * page_size;

        if (nr_mapped > 0) {
            double frag_percent = (bytes_allocated > 0) 
                ? (1.0 - ((double)bytes_useful / bytes_allocated)) * 100.0 
                : 0.0;

            printf("[vAttention FRAG] Req: %d | TotalSeq: %llu | Mapped Blocks: %llu | Resident: %llu | Useful: %.2f MB | Physical: %.2f MB | Internal Frag: %.2f%%\n",
                reqId, 
                (unsigned long long)seq_len,
                (unsigned long long)nr_mapped,
                (unsigned long long)resident_useful_tokens,
                (double)bytes_useful / (1024.0 * 1024.0),
                (double)bytes_allocated / (1024.0 * 1024.0),
                frag_percent);
        }
        // --- END FRAGMENTATION LOGIC ---

        if (nr_required <= nr_mapped)
            return;

        u64 nr_to_grow = nr_required - nr_mapped;
        printf("[vAttention DEBUG] reqId: %d | Growing by %llu blocks\n", reqId, (unsigned long long)nr_to_grow);

        if (!kvblocks_available(nr_to_grow))
            reclaim_kvblocks_on_demand(nr_to_grow);

        log.log("[DEBUG] allocating " + std::to_string(nr_to_grow) + " pages for reqId: " + std::to_string(reqId));
        grow_kvcache_phys(reqId, nr_to_grow, true);
        set_req_seq_length(reqId, seq_len);
    }
    /* This is not meant for final use but to test vattention with sync allocation */
    void step_sync(std::vector<u64> seq_lens, bool eager_reclaim)
    {
        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            /* set the sequence length based on latest values coming from the framework */
            set_req_seq_length(reqId, seq_lens[reqId]);
            /* request not active but physical memory blocks are allocated */
            if (eager_reclaim == true and (seq_lens[reqId] == 0 && mapped_pages[reqId] != 0))
            {
                release_kvcache_pages_all(reqId);
                continue;
            }
            map_pages_for_curr_step(reqId, seq_lens[reqId]);
        }
    }

    /* Ensure that we have enough pages for each new sequence */
    void prepare_prefill_kvcache()
    {
        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            map_pages_for_curr_step(reqId, get_req_seq_length(reqId));
        }
    }

    void reclaim_kvblocks_on_demand(u64 num_kvblocks)
    {
        u64 nr_mapped, nr_required;

        /* now we get into relaim mode */
        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
            /* demand fulfilled */
            if (kvblocks_available(num_kvblocks))
                break;

            nr_mapped = get_req_pages(reqId);
            nr_required = tokens_to_pages(get_req_seq_length(reqId));
            if (nr_mapped <= nr_required)
                continue;

            release_kvcache_pages_some(reqId, nr_required);
        }
    }

    /*
     * Release one page at a time (in the order opposite to how we allocated a new request id)
     * We do not reclaim from the req id that we know is going to be allocated soon
     */
    void do_reclaim_pages()
    {
        if (deferred_reclaim)
            return;

        int next_prefill_reqId = -1;

        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            if (!is_active_req(reqId))
            {
                next_prefill_reqId = reqId;
                break;
            }
        }

        for (int reqId = max_batch_size - 1; reqId >= 0; reqId--)
        {
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

/*
 * These are based on heuristics and should be fine for most cases.
 * Configure if needed.
 */
#define EAGER_NUM_STEPS (10)
#define EAGER_NUM_KVBLOCKS (2)
    void do_kvcache_memory_management()
    {
        u64 nr_required = 0;
        u64 nr_mapped_curr = 0;
        bool done = false;

        for (int reqId = 0; reqId < max_batch_size; reqId++)
            nr_required += need_new_page_async(reqId, 1);

        if (!kvblocks_available(nr_required))
        {
            log.log("[DEBUG] reclaiming " + std::to_string(nr_required) + " KV blocks in background thread...");
            reclaim_kvblocks_on_demand(nr_required);
        }

        /*
         * Check if we have enough free pages to continue. If not, we return without doing
         * anything, hoping that memory will become available when some request exits.
         */

        if (!kvblocks_available(nr_required))
            return;

        for (int eager_step_count = 1; eager_step_count < EAGER_NUM_STEPS && !done; eager_step_count++)
        {
            for (int reqId = 0; reqId < max_batch_size; reqId++)
            {
                u64 nr_required_curr = need_new_page_async(reqId, eager_step_count);
                grow_kvcache_phys(reqId, nr_required_curr, false);
                nr_mapped_curr += nr_required_curr;
                if (eager_step_count == 1)
                    continue;
                if (nr_mapped_curr >= EAGER_NUM_KVBLOCKS)
                {
                    done = true;
                    break;
                }
            }
        }

        /*
         * Doing too much work in one iteration can impact latency, so we
         * return without attemtping reclamation if we just allocated one or more pages.
         */
        if (nr_required)
            return;

        do_reclaim_pages();
    }

    void spawn_kvcache_manager()
    {
        std::thread([this]()
                    {
            mem_manager_running = true;
            do_kvcache_memory_management();
            mem_manager_running = false; })
            .detach();
    }

    /* single step for asynchronous allocation */
    void step_async(std::vector<u64> seq_lens)
    {
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
    int alloc_new_batch_idx(u64 seqlen)
    {
        int new_id = -1;
        u64 nr_required = tokens_to_pages(seqlen);

        for (int reqId = 0; reqId < max_batch_size; reqId++)
        {
            if (is_active_req(reqId))
                continue;

            if (new_id == -1)
            {
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

    void free_batch_idx(int reqId)
    {
        set_req_seq_length(reqId, 0);
    }

    void set_deferred_reclamation(bool val)
    {
        deferred_reclaim = val;
    }

    py::dict get_allocator_debug_info()
    {
        py::dict info;
        info["tokens_per_page"] = py::int_(tokens_per_page);
        info["page_buffer_token_bytes"] = py::int_(page_buffer_token_bytes);
        info["cached_token_bytes_local"] = py::int_(cached_token_bytes_local);
        info["pages_per_kvblock"] = py::int_(get_pages_per_kvblock());
        info["component_spec_enabled"] = py::bool_(component_spec_enabled);
        info["megacache_enabled"] = py::bool_(megacache_enabled);
        info["num_layers"] = py::int_(num_layers);
        info["num_cache_components"] = py::int_(get_num_cache_components());
        return info;
    }

    u64 debug_tokens_to_pages(u64 num_tokens)
    {
        return tokens_to_pages(num_tokens);
    }

    /* TODO(ashish): check if this is compatible with PyTorch destructor */
    void cleanup()
    {
        wait_kvcache_manager_sync();
        DO_KVCACHE_CLEANUP(page_size);
        k_tensors.clear();
        v_tensors.clear();
        cache_component_token_dims.clear();
        component_spec_enabled = false;
        megacache_enabled_global = false;
        tokens_per_page = 0;
        page_buffer_token_bytes = 0;
        cached_token_bytes_local = 0;
        log.log("released memory and cleaned up vattention ...");
    }
};

#include "apis.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    /*
     * These must be invoked during initialization/termination
     * TODO(ashish): Merge these into one API?
     */
    m.def("reserve_physical_pages", &reserve_physical_pages, "reserve physical memory blocks...");
    m.def("init_kvcache", &init_kvcache, "initialize KV cache...");
    m.def("init_kvcache_component_spec", &init_kvcache_component_spec, "initialize KV cache from a structured component spec...");
    m.def("cleanup", &cleanup, "cleanup and release allocator resources context...");
    /* Tunables and other helper APIs */
    m.def("set_verbose", &set_verbose, "to enable/disable printing logs...");
    m.def("set_deferred_reclamation", &set_deferred_reclamation, "enable/disable deferred freeing...");
    m.def("get_allocator_debug_info", &get_allocator_debug_info, "return allocator sizing metadata...");
    m.def("debug_tokens_to_pages", &debug_tokens_to_pages, "convert token count to logical kvblocks...");
    /* Testing APIs */
    m.def("show_kvcache_config", &show_kvcache_config, "show kv cache configuration...");
    m.def("show_allocator_state", &show_allocator_state, "show free pool of physical memory blocks...");
    m.def("step", &step, "step function...");
    m.def("map_common_pages", &map_common_pages, "map common pages in batch...");
    /* API for actual physical memory allocation - one call per iteration */
    m.def("step_async", &step_async, "single step function for the async version...");
    /* Request-level APIs */
    m.def("alloc_new_batch_idx", &alloc_new_batch_idx, "allocate a request id...");
    m.def("free_batch_idx", &free_batch_idx, "free a request id...");
    m.def("num_free_kvblocks", &num_free_kvblocks, "number of free kv blocks...");
}
