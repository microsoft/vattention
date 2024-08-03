void raise_warning_for_complex_half(at::ScalarType scalar_type)
{
    if (scalar_type == at::ScalarType::ComplexHalf)
    {
        std::cerr << "Warning: ComplexHalf type is used." << std::endl;
    }
}

class VirtualTensorAllocator : public at::Allocator
{
public:
    int device_idx = 0;
    size_t page_size = 0;

    VirtualTensorAllocator(int device_idx_, size_t page_size_)
    {
        this->device_idx = device_idx_;
        this->page_size = page_size_;
    }

    c10::DataPtr allocate(size_t size) override
    {
        c10::DeviceIndex device = this->device_idx;
        constexpr size_t one_exa_bytes = 1152921504606846976ULL;
        TORCH_CHECK_WITH(
            OutOfMemoryError,
            size < one_exa_bytes,
            "CUDA out of memory. Tried to allocate more than 1EB memory.");

        if (size == 0)
            throw std::runtime_error("can't allocate 0 sized tensor...");

        C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
        CUdeviceptr ptr_gpu;
        if (!is_uvm_backend(this->page_size))
        {
            CHECK_CUDA(cuMemAddressReserve(&ptr_gpu, size, 0, 0, 0));
        }
        else
        {
            void *ptr;
            C10_CUDA_CHECK(cudaMallocManaged(&ptr, size));
            ptr_gpu = (CUdeviceptr)ptr;
        }
        return {reinterpret_cast<void *>(ptr_gpu), reinterpret_cast<void *>(ptr_gpu), &release, c10::Device(c10::DeviceType::CUDA, device)};
    }

    /*
    TODO (ashish): check when this gets triggered
    */
    void copy_data(void *dest, const void *src, std::size_t count) const override
    {
        /* no-op */
    }

    /*
    TODO(ashish): add logic to release virtual memory
    */
    static void release(void *ptr)
    {
        /* no-op */
    }
};

template <typename T>
at::Tensor _alloc_vtensor(
    at::ArrayRef<T> shape,
    size_t page_size,
    c10::Allocator *allocator,
    c10::DispatchKeySet ks,
    at::ScalarType scalar_type,
    int device_idx,
    c10::optional<c10::MemoryFormat> memory_format_opt)
{
    at::detail::check_size_nonnegative(shape);
    raise_warning_for_complex_half(scalar_type);
    caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    auto size_bytes = at::detail::computeStorageNbytesContiguous(shape, dtype.itemsize());
    size_bytes = ROUND_UP(size_bytes, page_size);
    /*
     * ensure that each request's buffer is at least as big as the page size
     * first element of shape should always be batch size
     */
    if (size_bytes < page_size * shape[0])
        size_bytes = page_size * shape[0];

    if (size_bytes % (page_size * shape[0]) != 0)
        throw std::runtime_error("size_bytes is not a multiple of page_size * shape[0]");

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator,
        true);

    auto tensor = at::detail::make_tensor_base<c10::TensorImpl>(
        std::move(storage_impl), ks, dtype);

    if (ks.has(c10::DispatchKey::Meta) || shape.size() != 1 || shape[0] != 0)
        tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(shape);

    if (memory_format_opt.has_value()) {
        // Restriding a just-created empty contiguous tensor does nothing.
        if (*memory_format_opt != c10::MemoryFormat::Contiguous)
            tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
    return tensor;
}

TORCH_CUDA_CPP_API at::Tensor alloc_vtensor(
    at::IntArrayRef shape,
    size_t page_size,
    at::ScalarType dtype,
    VirtualTensorAllocator *allocator,
    int device_idx)
{
    c10::optional<c10::MemoryFormat> memory_format_opt;
    at::globalContext().lazyInitCUDA();
    c10::Device device = c10::Device(c10::kCUDA, device_idx);
    TORCH_INTERNAL_ASSERT(device.is_cuda());
    const c10::DeviceGuard device_guard(device);
    // VirtualTensorAllocator *allocator = new VirtualTensorAllocator();
    constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
    return _alloc_vtensor(shape, page_size, allocator, cuda_dks, dtype, device_idx, memory_format_opt);
}