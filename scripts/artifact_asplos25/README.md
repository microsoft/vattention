# Introduction

vAttention is a simpler, portable and performant alternative to PagedAttention for dynamic memory management in LLM serving systems. This directory contains the artifact scripts to reproduce key results of our paper. For more detailed documentation, refer to [README.md](../../README.md).

# Requirements

The artifact requires **PyTorch 2.3.0** and **CUDA 12.1** (or later but other CUDA versions may or may not work). We have tested vAttention with the Linux kernel, **A100 GPUs** and **python 3.10** but expect it to work on other Linux-based systems as long as they are running the specified CUDA and PyTorch versions. The scripts are configured to run three models `Yi-6B`, `Llama-3-8B` and `Yi-34B`. Running `Yi-6B` requires one 80GB A100 GPU while the other two models require two 80GB GPUs connected via NVLink. If you have access to a single GPU, please update the run scripts to avoid running `Llama-3-8B` and `Yi-34B`.

# Installation

First, create and activate a conda environment as follows:

```sh
conda create -n vattn python=3.10
conda activate vattn
```

Now, install the artifact as follows:

```sh
./install.sh
```

You can launch all experiments at once or individually as:

```sh
./run_all.sh
or
./run_figure_2.sh
./run_figure_3.sh
```

The raw output logs and the final plots will be redirected to `./logs` and `./plots` subdirectories respectively. 