# Introduction

vAttention is a memory manager for KV cache in LLM serving systems. The key feature of vAttention is that it adds support for dynamic memory allocation to unmodified attention kernels, enabling more efficient use of GPU memory without re-writing GPU kernels. We achieve this by storing KV cache in contiguous virtual memory while leverating system (CUDA) support for dynamic allocation of physical memory. vAttention also improves performance in many cases, over the state-of-the-art PagedAttention approach, especially for prefill-bound workloads. Please checkout our [paper](https://arxiv.org/abs/2405.04437) for more details.

# Getting Started

This repository contains an implementation of vAttention, intergrated with an LLM serving system Sarathi-Serve ([paper](https://www.usenix.org/conference/osdi24/presentation/agrawal), [code](https://github.com/microsoft/sarathi-serve)) published in OSDI'24.

# Installation

Using this repo requires **PyTorch 2.3.0** and **CUDA 12.1** (other CUDA versions may or may not work). We have tested vAttention on **A100 GPUs** using **python 3.10.13** but expect it to work on other systems as long as they are running the specified CUDA and PyTorch versions.

### Installing vAttention and Sarathi-Serve

Please download and extract libtorch first (this is required to build the vattention memory allocator), and then build sarathi-serve and vattention as follows:

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

The repo provides a benchmark-runner which can be used to run different workloads (dynamic/static, datasets/synthetic) with various attention-backends and schedulers. The benchmark-runner provides a comprehensive list of configuration knobs listed in [default.yml](sarathi-lean/sarathi/benchmark/config/default.yml). A thorough explaination of the knobs can be found here: [Sarathi-Serve](sarathi-lean/sarathi/benchmark/README.md).

For experiments related to vAttention, the following are the most important knobs (note that the knobs are case-insensitive):
- `attention_backend` : Specify the attention kernel to be used for attention. Currently supports `fa_paged`, `fi_paged`, `fa_vattn`, `fi_vattn`, `fa_vattn_sync`, `fi_vattn_sync` where *'fa'* denotes FLASHATTENTION and *'fi'* denotes FLASHINFER. We tested v2.5.9 for FLASHATTENTION and v0.0.6 for FLASHINFER. More details on the backends can be found here: [Attention Backends](#attention-backends).

This repository includes **two** template benchmark scripts:

1. [benchmark_e2e_dynamic_trace.py](scripts/benchmark_e2e_dynamic_trace.py): This script is to run expriments on a dynamic trace. It runs **256** requests from the **_arxive dataset_** for qps of 0.4, 0.8, 1, 2, 4 and 6 where requests arrive in an interval of **_poisson_** distribution.

1. [benchmark_e2e_static_trace.py](scripts/benchmark_e2e_static_trace.py): This script is for static end-to-end benchmarking experiments to reproduce the results in the paper. The script runs 50 requests for context length 32k, 64k and 128k and prefill to decode ratio of 500, 100 and 50. Results should be generated for `fa_paged`, `fi_paged`, `fa_vattn`.

```sh
# run benchmark scripts as follows:
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

vAttention delegates the responsibility of memory management to the kernel space (CUDA drivers). By separating the allocation of virtual and physical memory, and allocating physical memory on demand, it provides a contiguous interface to the per-request KV cache without compromising on efficiency of GPU kernels. 

Integrating vAttention into an LLM serving system is simple. We choose Sarathi-Serve to exemplify this because Sarathi-Serve is a state-of-the-art LLM inference scheduler, has an elaborate metric store and a versatile benchmark_runner that makes running traces and performing experiments easy. Furthermore, its modular setup for the attention backends makes it easy to add more attention backends.

The core of our code changes are as follows:

- [vATTN_cache_engine.py](sarathi-lean/sarathi/worker/cache_engine/vATTN_cache_engine.py):

    A `vATTNCacheEngine` class initializes and manages some aspects of the KV cache in python land e.g., mapping the sequence id of a request to its batch index in the KV cache, and the current context length of the each request [like vLLMCacheEngine].

    - But unlike its vLLM counterpart, when an object of this class is instantiated, it calls:

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

        which returns **virtual** PyTorch tensors without any physical memory mapped underneath.
        <br/>

    - vATTNCacheEngine also calls the following api

        ```sh
        vattention.reserve_physical_pages(cache_config.memory_for_gpu)
        ```
        which pre-allocates physical memory pages on the GPU for KV cache. These pages are then attached to the virtual memory tensors at runtime. 

- [base_worker.py](sarathi-lean/sarathi/worker/base_worker.py)

    - before the model forward pass, the worker calls
        ```sh
        vattention.step_async(self.curr_seq_lens)
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

    Can be accessed by setting the values of *'--attention_backend'* to *'fa_vattn'*

2. [vattention_flashinfer_wrapper.py](sarathi-lean/sarathi/model_executor/attention/vattention_flashinfer_wrapper.py):

    This backend implements non-paged prefill using [flashinfer](https://github.com/flashinfer-ai/flashinfer)'s `flashinfer.prefill.single_prefill_with_kv_cache` and non-paged decode using flash_attention's `flash_attn_with_kvcache`

    Can be accessed by setting the values of *'--attention_backend'* to *'fi_vattn'*

Note: Using *'fa_vattn'* or *'fi_vattn'* automatically runs vAttention with asynchronous memory allocation. To access the synchronous version, add suffix *'_sync'* to the *'--attention_backend'* knob. For example, if you want to run *'fa_vattn'* or *'fi_vattn'* with synchronous memory allocation, supply *'fa_vattn_sync'* or *'fi_vattn_sync'* to *'--attention_backend'*.

Our current release supports only 2MB pages. Support for smaller pages (e.g., 64KB) is work-in-progress.


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
