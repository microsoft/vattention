#include "vattn.h"

#define KB (1024UL)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define KVBLOCKS_TO_PAGES(slots) ((slots) * (2) * (num_layers))
#define PAGES_TO_KVBLOCKS(pages) ((pages) / (2 * (num_layers)))

typedef CUmemGenericAllocationHandle CUHandle;

bool verbose = false;

CUcontext ctx;
CUmemAllocationProp prop = {};
CUmemAccessDesc accessDesc = {};
// physical memory handles for non-uvm backend (default)
std::vector<CUmemGenericAllocationHandle> page_handles;
// physical memory handles for uvm backend (custom driver)
std::vector<NvU64> uvm_page_handles;

static inline size_t pad_size(size_t orig, size_t align) {
    return (orig + align - 1) & ~(align - 1);
}

static inline bool use_uvm_backend(size_t page_size) {
    return page_size != 2 * MB;
}