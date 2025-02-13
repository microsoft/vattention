# POD-Attention
This repository contains the source code and profiling scripts for POD-Attention. POD-Attention fuses prefill and decode attention kernels into a single optimized kernel that aims to saturate both GPU compute and memory simultaneously, critical for hybrid-batching-based LLM inference.
POD-Attention is built on top of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main) kernels (v2.6.1) and is integrated with Sarathi-Serve - a state-of-the-art hybrid-batching-based LLM inference scheduler. This artifact contains the source code of POD-Attention, benchmarks used for evaluation, and all scripts needed to replicate results reported in the paper.

Full details of our implementation can be found in our paper:
<pre>
<b>POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference</b>
[To appear in] <i>ACM 30th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2025</i>
https://arxiv.org/abs/2410.18038
</pre>

## Performance
![POD_attention_sweep](https://github.com/user-attachments/assets/f5d90c6f-4b73-435c-8be5-23dc3fbed7f1)
To examine POD-Attention's broad applicability in LLM inference, we examined over a thousand different hybrid batch configurations, sweeping different context lengths, decode batch sizes, and LLM model configurations.
The above graph shows POD-Attention's performance on these, normalized to the current approach of serially executing FlashAttention-2's prefill and decode kernels. 

POD-Attention outperforms this approach by up to <b>61% (average 33%)</b>.

FA_Stream: Executes [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) with prefill and decode in separate streams.   
FI_Serial: Executes [FlashInfer](https://github.com/flashinfer-ai/flashinfer) prefill and decode in serial.   
FI_Batched: Combines prefill and decode inputs and execute using a single FlashInfer kernel.   
FA_HFuse: Employs [HFuse](https://github.com/aoli-al/HFuse) to create a merged kernel for prefill and decode.   
POD (FI): Our POD-Attention implementation fusing FlashInfer's prefill and decode kernels ([available here](https://github.com/AKKamath/flashinfer/)).   
POD (FA): Our POD-Attention implementation fusing FlashAttention-2's prefill and decode kernels (this repository).   

# Installation and dependencies
Minimum NVIDIA Ampere GPU is needed to run this code. We tested using an A100 GPU on an x86 machine running Ubuntu 22.04.

## Docker installation 
We provide a docker image for POD-Attention with all its dependencies pre-installed. To access the docker image, you need to have [Docker](https://docs.docker.com/engine/installation/) and [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker/) installed on your system. You can then launch the docker container and navigate to the folder containing the POD-Attention artifact, as follows:
```sh
$ docker run --gpus all -it \
  -p 8181:8181 --rm --ipc=host --cap-add=SYS_ADMIN \
  rnp1910/pod_attention:asplos_25_pytorch_run
$ cd /workspace/vattention/pod_attn  
```

## Manual installation
For manual installation, we can download POD-Attention to the home directory to install it. 
We use Anaconda to obtain the appropriate versions of CUDA, Python, and PyTorch. 
This can take up to 2 hours.
```sh
$ git clone \
  https://github.com/microsoft/vattention.git
$ cd vattention/pod_attn/
# Install miniconda; skip if already installed
$ make install_miniconda
$ bash # Refresh shell and activate
$ conda activate pod_attn
# Install CUDA Toolkit
(pod_attn)$ conda install -y -c \
  conda-forge cuda-toolkit=12.4.0
# Install dependencies
(pod_attn)$ pip install -r requirements.txt
# Install FlashInfer
(pod_attn)$ git clone https://github.com/AKKamath/flashinfer.git --recursive
(pod_attn)$ pushd flashinfer; pip install -e . -v; popd
# Install POD-Attention and vAttention
(pod_attn)$ make install_all
```

## Test configuration
The code has been tested with CUDA 12.4 and Python 3.9 on an A100 GPU.

# Running POD-Attention

POD-Attention's API has been integrated into vAttention's fork of Sarathi-serve. You can run the associated backend by running sarathi-serve's benchmark runner with the attention backend `'FA_POD'` or `'FA_POD_MEGACACHE'`.

Our evaluation primarily contains two kinds of experiments, attention performance (Figures 1, 6, 10, 11, 13, 14) and end-to-end LLM performance (Figure 12 and Table 5). 
Figure 7 evaluates various kernel fusion strategies with a micro-benchmark. Most of these require only one GPU except for Table 5 and Figure 12 that require two GPUs. Use the Makefile present in the vattention/pod_attn/ folder to run experiments as follows:

```sh
make figure1  # 2 minutes; sudo used by script
make figure6  # 2 minutes
make figure7  # 2 minutes
make figure10 # 1 minute; sudo used by script
make figure11 # 2 hours
make figure12 # 9 hours
make figure13 # 1 minute
make figure14 # 1 minute
make table6 # 4 hours
```

## Expected results
The artifact scripts redirect the raw output numbers and logs to the output/ folder, while the plotted graphs can be found in the graphs/ folder. Tables are saved as CSVs in the same folder. Results may have minor runtime variations from those reported in in the paper, but general trends should hold.

# Folder contents
The various folders are as follows:
* csrc/cutlass/ contains required CUTLASS files.
* pod_attn/ contains the source code. The important files are:
	* pod_attn/fused_fwd_kernel.h --- Contains the GPU code. Specifically, "compute_fused_tb_attn" contains POD-Attention's SM-aware CTA scheduling code.
	* pod_attn/fused_fwd_launch_template.h --- Contains the code which launches the POD-Attention kernel. In this file, "run_true_fused_fwd" contains the actual kernel launch, while "run_true_fused_mha_fwd_hdim128" performs various parameter selection (prefill/decode tile size) for the launch.
	* pod_attn/fused_api.cpp --- Contains some preprocessing and parameter selection. Here, "mha_true_fused_fwd_kvcache" contains the code for limiting prefill splitting.
* tests/ contains tests used during evaluation of POD-Attention.

