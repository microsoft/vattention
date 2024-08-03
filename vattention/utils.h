#define KB (1024UL)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define KVBLOCKS_TO_PAGES(kvblocks) ((kvblocks) * (2) * (num_layers))
#define PAGES_TO_KVBLOCKS(pages) ((pages) / (2 * (num_layers)))
#define ROUND_UP(x, y) ((((x) + (y) - 1) / (y)) * (y))
bool verbose = false;

typedef long long unsigned int NvU64;
typedef unsigned long u64;

CUcontext ctx;
CUmemAllocationProp prop = {};
CUmemAccessDesc accessDesc = {};
typedef CUmemGenericAllocationHandle CUPage;
// physical memory pages for non-uvm backend (default)
std::vector<CUmemGenericAllocationHandle> cuda_pages;
// physical memory pages for uvm backend (custom driver)
std::vector<NvU64> uvm_pages;

using cudaPhysPageMap = std::map<std::tuple<u64, u64, u64>, std::pair<CUPage, CUPage>>;
using uvmPhysPageMap = std::map<std::tuple<u64, u64, u64>, std::pair<NvU64, NvU64>>;

// track physical memory pages (required while unmapping)
cudaPhysPageMap cuda_pagemap;
uvmPhysPageMap uvm_pagemap;

std::vector<CUdeviceptr> k_ptr;
std::vector<CUdeviceptr> v_ptr;

// kv cache for the full model, one elem per layer
std::vector<at::Tensor> k_tensors;
std::vector<at::Tensor> v_tensors;

// this synchronizes background daemons with API calls
std::atomic<bool> mem_manager_running(false);

/* model specifc params */
int num_kv_heads;
int head_size;
int num_layers;
int bytes_per_elem;
int device;
py::object dtype;

/* framework specific params */
int max_batch_size;
long max_context_length;

/* virtual memory metadata */
u64 virt_buff_size;
u64 virt_buff_size_per_req;
u64 virt_buff_size_per_token;
u64 virt_buff_num_kv_tokens;

/* allocator and GPU specific data */
u64 tokens_per_page; // number of elements per physical block

/* a request can be allocated only this many pages per vtensor */
u64 max_pages_per_req;

/* physical memory metadata */
std::vector<u64> mapped_pages; // per request allocated block count
std::vector<u64> curr_seq_lengths; // per request allocated block count

/* garbage collection thread to utilize slack */
std::thread gc_thread;

/*
* This controls whether we want to reclaim memory in the background
* thread or not. If set to false, reclaim memory only when required.
*/
bool deferred_reclaim = true;

/* memory allocator specific. page_size is in bytes */
u64 page_size = 2 * MB;

static inline bool is_uvm_backend(u64 page_size) {
    return page_size != 2 * MB;
}

void init_kvcache_batch_metadata() {
    mapped_pages.resize(max_batch_size);
    curr_seq_lengths.resize(max_batch_size);
    for(int i = 0; i < max_batch_size; i++) {
        curr_seq_lengths[i] = 0;
        mapped_pages[i] = 0;
    }
}

bool check_kvcache_config() {
    return !(num_layers == 0 || num_kv_heads == 0 || head_size == 0 ||
                max_batch_size == 0 || max_context_length == 0);
}

inline u64 tokens_to_pages(u64 num_tokens) {
    return (num_tokens + tokens_per_page - 1) / tokens_per_page;
}

inline u64 pages_to_tokens(u64 num_blocks) {
    return num_blocks * tokens_per_page;
}

inline u64 get_req_mapped_tokens(int reqId) {
    return pages_to_tokens(mapped_pages[reqId]);
}

inline u64 get_req_pages(int reqId) {
    return mapped_pages[reqId];
}

inline u64 inc_req_page_count(int reqId) {
    return ++mapped_pages[reqId];
}

inline u64 dec_req_page_count(int reqId) {
    return --mapped_pages[reqId];
}

inline u64 get_req_begin_offset_virt(int reqId) {
    return reqId * virt_buff_size_per_req;
}

inline bool is_active_req(int reqId) {
    return curr_seq_lengths[reqId] != 0;
}

inline u64 get_req_seq_length(int reqId) {
    return curr_seq_lengths[reqId];
}

inline void set_req_seq_length(int reqId, u64 seq_len) {
    curr_seq_lengths[reqId] = seq_len;
}

inline void set_curr_seq_lengths(std::vector<u64> seq_lens) {
    curr_seq_lengths = seq_lens;
}

inline void wait_kvcache_manager_sync() {
    while (mem_manager_running);
}

inline void set_seq_lengths_for_next_step(std::vector<u64> seq_lens) {
    for (int reqId = 0; reqId < max_batch_size; reqId++) {
        if (!is_active_req(reqId))
            continue;

        set_req_seq_length(reqId, seq_lens[reqId] + 1);
    }
}

inline u64 get_num_overcommitted_kvblocks() {
    u64 overcommitted_kvblocks = 0;
    for (int reqId = 0; reqId < max_batch_size; reqId++)
        overcommitted_kvblocks += mapped_pages[reqId] - tokens_to_pages(get_req_seq_length(reqId));
    return overcommitted_kvblocks;
}

inline u64 get_req_current_map_offset(int reqId) {
    u64 num_mapped_blocks = get_req_pages(reqId);
    u64 block_offset_within_req = num_mapped_blocks * page_size;

    return get_req_begin_offset_virt(reqId) + block_offset_within_req;
}

inline u64 get_req_current_unmap_offset(int reqId) {
    u64 num_mapped_blocks;
    u64 block_offset_within_req;

    num_mapped_blocks = get_req_pages(reqId);
    /* unmap should not be called if nothing is mapped */
    assert(num_mapped_blocks > 0);
    /* return the offset to unmap, which is the beginning of the last mapped block */
    block_offset_within_req = (num_mapped_blocks-1) * page_size;
    return get_req_begin_offset_virt(reqId) + block_offset_within_req;
}

u64 need_new_page_async(int reqId, int eager_step_count) {
    if (!is_active_req(reqId))
        return 0;

    /* do not attempt allocation if a request already has been allocated max num pages */
    u64 nr_mapped = get_req_pages(reqId);
    if (nr_mapped == max_pages_per_req)
        return 0;

    /* add eager_step_count to sequence length to allocate memory proactively */
    u64 nr_required = tokens_to_pages(get_req_seq_length(reqId) + eager_step_count);
    return nr_required <= nr_mapped ? 0 : nr_required - nr_mapped;
}

u64 get_num_phys_blocks(u64 num_layers, u64 free_memory, u64 page_size)
{
    u64 num_phys_blocks;
    /* do not allocate if we can't use a page. we need multiples of 2 * num_layers */
    num_phys_blocks = free_memory / page_size;
    num_phys_blocks -= num_phys_blocks % (2 * num_layers);
    return num_phys_blocks;
}

class Log {
  public:
    void log(const std::string& msg) {
      if (verbose)
        std::cout << msg << std::endl;
    }
};
