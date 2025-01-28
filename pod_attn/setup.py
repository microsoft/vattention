# Copyright (c) 2023, Tri Dao.

import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform

from setuptools import setup, find_packages
import subprocess

import urllib.request
import urllib.error
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "pod_attn"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FLASH_ATTENTION_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return f'linux_{platform.uname().machine}'
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    check_if_cuda_home_none("fused_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                "POD-Attention is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
    # cc_flag.append("-gencode")
    # cc_flag.append("arch=compute_75,code=sm_75")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    ext_modules.append(
        CUDAExtension(
            name="fused_attn",
            sources=[
                "pod_attn/fused_api.cpp",
                #"pod_attn/truefused_fwd_hdim128_fp16_sm80.cu",
                #"pod_attn/truefused_fwd_hdim128_fp16_split_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_fo8_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_split_fo8_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_fo9_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_split_fo9_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_fo10_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_split_fo10_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_fo11_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_split_fo11_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_fo64_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_split_fo64_sm80.cu",
                #"pod_attn/truefused_fwd_hdim128_fp16_causal_sm80.cu",
                #"pod_attn/truefused_fwd_hdim128_fp16_causal_split_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_fo8_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_split_fo8_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_fo9_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_split_fo9_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_fo10_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_split_fo10_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_fo11_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_split_fo11_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_fo64_sm80.cu",
                "pod_attn/truefused_fwd_hdim128_fp16_causal_split_fo64_sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
                        "-DNDEBUG",  # Important, otherwise performance is severely impacted
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        # "-DFLASHATTENTION_DISABLE_DROPOUT",
                        # "-DFLASHATTENTION_DISABLE_ALIBI",
                        # "-DFLASHATTENTION_DISABLE_SOFTCAP",
                        # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                        # "-DFLASHATTENTION_DISABLE_LOCAL",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="flash_attn_og",
            sources=[
                "pod_attn/flash_api.cpp",
                "pod_attn/flash_fwd_hdim128_fp16_sm80.cu",
                "pod_attn/flash_fwd_hdim128_fp16_causal_sm80.cu",
                "pod_attn/flash_fwd_split_hdim128_fp16_sm80.cu",
                "pod_attn/flash_fwd_split_hdim128_fp16_causal_sm80.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                        # "--ptxas-options=-v",
                        # "--ptxas-options=-O2",
                        # "-lineinfo",
                        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
                        "-DNDEBUG",  # Important, otherwise performance is severely impacted
                        # "-DFLASHATTENTION_DISABLE_BACKWARD",
                        # "-DFLASHATTENTION_DISABLE_DROPOUT",
                        # "-DFLASHATTENTION_DISABLE_ALIBI",
                        # "-DFLASHATTENTION_DISABLE_SOFTCAP",
                        # "-DFLASHATTENTION_DISABLE_UNEVEN_K",
                        # "-DFLASHATTENTION_DISABLE_LOCAL",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    )


def get_package_version():
    with open(Path(this_dir) / "pod_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "output",
            "scripts",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "microbenchmarks",
            "pod_attn.egg-info",
        )
    ),
    description="POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/vattention/tree/main/pod_attn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
    ],
    setup_requires=[
        "packaging",
        "psutil",
        "ninja",
    ],
)
