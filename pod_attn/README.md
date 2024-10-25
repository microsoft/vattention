# POD-Attention
This repository contains the source code and profiling scripts for POD-Attention. POD-Attention fuses prefill and decode attention kernels into a single optimized kernel that aims to saturate both GPU compute and memory simultaneously.

Full details of our implementation can be found in our paper:
<pre>
<b>POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference</b>
https://arxiv.org/abs/2410.18038
</pre>

This repository started as a fork of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main).
<br></br>

# Folder contents
The various folders are as follows:
* csrc/cutlass/ contains required CUTLASS files.
* pod_attn/ contains the source code. The important files are:
	* pod_attn/fused_fwd_kernel.h --- Contains the GPU code. Specifically, "compute_fused_tb_attn" contains POD-Attention's SM-aware CTA scheduling code.
	* pod_attn/fused_fwd_launch_template.h --- Contains the code which launches the POD-Attention kernel. In this file, "run_true_fused_fwd" contains the actual kernel launch, while "run_true_fused_mha_fwd_hdim128" performs various parameter selection (prefill/decode tile size) for the launch.
	* pod_attn/fused_api.cpp --- Contains some preprocessing and parameter selection. Here, "mha_true_fused_fwd_kvcache" contains the code for limiting prefill splitting.
* tests/ contains tests used during evaluation of POD-Attention.


# Installation and dependencies
Minimum NVIDIA Ampere GPU is needed to run this code.

Python dependencies are listed in requirements.txt, they can be installed with:
```sh
pip install -r requirements.txt
```
To compile and build POD-Attention, run the following command:
```sh
python setup.py install
```
This may take around half an hour to compile for the first time.

## Test configuration
The code has been tested with CUDA 12.4 and Python 3.9 on an A100 GPU.

# Running POD-Attention

POD-Attention's API has been integrated into vAttention's fork of Sarathi-serve. You can run the associated backend by running sarathi-serve's benchmark runner with the attention backend `'FA_POD'` or `'FA_POD_MEGACACHE'`.

### Motivational results (Figure 1)
To replicate the results of figure 1, [NCU](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) must be installed on your machine and your account should have sudo priviledges.

Run:
```sh
bash tests/banner_fig.sh fa
bash tests/banner_fig.sh pod
bash tests/banner_fig.sh perf
```
The first command profiles FlashAttention kernels using NCU, the second profiles POD-Attention, while the third compares the performance of FlashAttention, FlashInfer and POD-Attention.
Output is generated in the output/ folder.

### Attention performance sweep (Figure 11)
We include a [test file](pod_attn/tests/attn_sweep.py) to compare POD-Attention, [FlashAttention (FA)](https://github.com/Dao-AILab/flash-attention), [FlashInfer (FI)](https://github.com/flashinfer-ai/flashinfer/), and [HFuse with FlashAttention (FA_HFuse)](https://github.com/aoli-al/HFuse) across a variety of context lengths and models.

Before running this test, make sure FlashInfer is installed and available.

FA v2.6.1 and FA_HFuse are already included and installed by this repository and do not need to be separately installed.

To run the test, run:

```sh
python  tests/attn_sweep.py
```
Output is in semicolon-separated format.