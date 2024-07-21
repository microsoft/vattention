# Introduction

vAttention is a memory manager for KV-cache in LLM serving systems. It adds support for dynamic memory allocation to unmodified attention kernels, by storing KV-cache in contiguous virtual memory and leveraging system support ([CUDA virtual memory APIs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)) for on-demand allocation of physical memory. This way of memory management is different from the popular [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) approach; PagedAttention implements demand paging in user space and requires rewriting custom kernels to support dynamic memory allocation. vAttention also improves performance over PagedAttention in many cases, especially for prefill-bound workloads. Please checkout our [paper](https://arxiv.org/abs/2405.04437) for more details.

# Getting Started

This repository contains an implementation of vAttention, intergrated with an LLM serving system Sarathi-Serve that was published in OSDI'24 ([paper](https://www.usenix.org/conference/osdi24/presentation/agrawal), [code](https://github.com/microsoft/sarathi-serve)).

# Installation

Using this repo requires **PyTorch 2.3.0** and **CUDA 12.1** (or later but other CUDA versions may or may not work). We have tested vAttention on **A100 GPUs** using **python 3.10** but expect it to work on other systems as long as they are running the specified CUDA and PyTorch versions.

### Installing vAttention and Sarathi-Serve

Create a conda environment as follows:

```sh
conda create -n vattn python=3.10
conda activate vattn
```

Now, download and extract libtorch first (this is required to build the vattention memory allocator), and then build sarathi-serve and vattention as follows:

```sh
# the libtorch version has to match with the torch version, and we have tested only v2.3.0
wget https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-shared-with-deps-2.3.0+cu121.zip

# build sarathi-serve
cd sarathi-lean/
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
cd ../

# build vattention
cd vattention/
LIBTORCH_PATH=<path to libtorch dir> python setup.py install
cd ../
```
# Running benchmarks

The repo provides a benchmark-runner which can be used to run different workloads (dynamic/static, datasets/synthetic) with various attention-backends and schedulers. The benchmark-runner provides a comprehensive list of configuration knobs listed in [default.yml](sarathi-lean/sarathi/benchmark/config/default.yml). Please check [Sarathi-Serve](sarathi-lean/sarathi/benchmark/README.md) for a detailed explanation of the knobs.

For experiments related to vAttention, the following are the most important knobs (these are case-insensitive):
- `attention_backend` : Specify the attention kernel to be used. Currently supports `fa_paged_[block_size]`, `fi_paged_[block_size]`, `fa_vattn_[page_size]`, `fi_vattn_[page_size]`, `fa_vattn_[page_size]_sync`, `fi_vattn_[page_size]_sync` where `fa` denotes FlashAttention (we tested v2.5.9) and `fi` denotes FlashInfer (we tested v0.0.6). We recommend using block size 256 for FlashAttention and 16 for FlashInfer (set the knobs as `fa_paged_256`, `fi_paged_16`) because we have observed them performing best with these block sizes. vAttention supports 64KB, 128KB, 256KB and 2MB page sizes (example knobs: `fa_vattn_256kb`, `fi_vattn_2mb_sync`). Using suffix `_sync` in vAttention knob disables our optimization of overlapping memory allocation with compute (i.e., GPU physical memory is allocated synchronously). More details on the backends can be found here: [Attention Backends](#attention-backends).

This repository includes **two** template benchmark scripts:

1. [benchmark_e2e_dynamic_trace.py](scripts/benchmark_e2e_dynamic_trace.py): This script is to run expriments on a dynamic trace. It runs **256** requests from the **_arxive dataset_** for qps of 0.4, 0.8, 1, 2, 4 and 6 where requests arrive in an interval of **_poisson_** distribution.

1. [benchmark_e2e_static_trace.py](scripts/benchmark_e2e_static_trace.py): This script is for static end-to-end benchmarking experiments to reproduce the results in the paper. The script runs 50 requests for context length 32k, 64k and 128k and prefill to decode ratio of 500, 100 and 50.

```sh
# testing the setup:
python scripts/benchmark_e2e_static_trace.py --test
or 
python scripts/benchmark_e2e_dynamic_trace.py --test

# run benchmarks for performance evaluation as follows:
python scripts/benchmark_e2e_static_trace.py
or
python scripts/benchmark_e2e_dynamic_trace.py
```

Benchmark results are redirected to `experiments/e2e_static_eval` or `experiments/e2e_dynamic_eval`. Model configurations can be found in `scripts/utils.py` (all Yi and Llama family of models are expected to work). Parse benchmark results as follows:

```sh
python scripts/process_e2e_static.py
or
python scripts/process_e2e_dynamic.py
```


<br/>

# Implementation Details

vAttention delegates the responsibility of memory management to CUDA drivers that run in the OS kernel space. By decoupling the allocation of virtual and physical memory, vAttention enables allocating physical memory on demand while retaining the virtual memory contiguity of KV-cache.

Integrating vAttention into an LLM serving system is simple. We choose Sarathi-Serve to exemplify this because Sarathi-Serve is a state-of-the-art LLM inference scheduler, has an elaborate metric store and a versatile benchmark_runner that makes running traces and performing experiments easy. Furthermore, its modular setup for the attention backends makes it easy to add more attention backends.

The core of our code changes are as follows:

- [vATTN_cache_engine.py](sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py):

    A `vATTNCacheEngine` class initializes and manages some aspects of the KV cache in python land e.g., mapping the sequence id of a request to its batch index in the KV-cache, and the current context length of the each request [like vLLMCacheEngine]. A serving system initializes the vAttention memory allocator as follows:

    ```sh
    vattention.init_kvcache(
        self.num_layers,
        self.num_heads,
        self.head_size,
        self.max_batch_size,
        self.max_model_seq_len,
        self.device_idx,
        self.dtype,
        USE_UVM)
    ```

    which returns **virtual** PyTorch tensors without any physical memory mapped underneath. The serving system can also reserve physical memory for KV-cache ahead-of-time as follows:

    ```sh
    vattention.reserve_physical_pages(cache_config.memory_for_gpu)
    ```
    
    which pre-allocates physical memory pages on the GPU. These pages are then attached to the virtual tensors at runtime. 

- [base_worker.py](sarathi-lean/sarathi/worker/base_worker.py)

    - before the model forward pass, the worker calls
        ```sh
        # asynchronous memory allocation
        vattention.step_async(self.curr_seq_lens)
        or
        # synchronous memory allocation
        vattention.step(self.curr_seq_lens)
        ```
        which allocates physical memory pages for the active requests, based on their requirement.

    - after the forward pass, the worker calls
        ```sh
        vattention.free_batch_idx(batch_idx)
        ```
        on request ids which have been completed. vAttention can reclaim these pages to allocate them to different requests on need basis.


And that is it.

# Attention Backends

We have modified Sarathi-Serve to support the following backends:

1. [vattention_flashattention_wrapper.py](sarathi-lean/sarathi/model_executor/attention/vattention_flashattention_wrapper.py):

    This backend implements non-paged prefill and decode using [flash_attention](https://github.com/Dao-AILab/flash-attention)'s `flash_attn_with_kvcache` API [for both prefill and decode computation]

    Can be accessed by setting the values of `--attention_backend` to `fa_vattn_[page_size]` or `fa_vattn_[page_size]_sync`

2. [vattention_flashinfer_wrapper.py](sarathi-lean/sarathi/model_executor/attention/vattention_flashinfer_wrapper.py):

    This backend implements non-paged prefill using [flashinfer](https://github.com/flashinfer-ai/flashinfer)'s `flashinfer.prefill.single_prefill_with_kv_cache` and non-paged decode using flash_attention's `flash_attn_with_kvcache`

    Can be accessed by setting the values of `--attention_backend` to `fi_vattn_[page_size]` or `fi_vattn_[page_size]_sync`.

## Using smaller page sizes

NVIDIA CUDA drivers allocate memory only at the granularity of large pages (2MB or above). If you want to use smaller page sizes of 64KB, 128KB or 256KB with vAttention, please follow the [README.md](./nvidia-vattn-uvm-driver/README.md) to replace the default CUDA UVM driver with our custom driver (nvidia-vattn-uvm-driver).

**NOTE:** Replacing CUDA drivers is not required if you only want to use vAttention with 2MB pages.

## Citation

If you use our work, please consider citing our paper:

```
@misc{prabhu2024vattention,
      title={vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention},
      author={Ramya Prabhu and Ajay Nayak and Jayashree Mohan and Ramachandran Ramjee and Ashish Panwar},
      year={2024},
      eprint={2405.04437},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.04437},
}
```

## Acknowledgment

This repository originally started as a fork of [Sarathi-Serve](https://github.com/microsoft/sarathi-serve) which in turn is a fork of the [vLLM project](https://vllm-project.github.io/). vAttention and Sarathi-Serve are research prototypes and do not have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.
