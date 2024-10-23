# Introduction
This repository contains the source code and profiling scripts for POD-Attention. POD-Attention fuses prefill and decode attention kernels into a single optimized kernel that aims to saturate both GPU compute and memory simultaneously.

This repository originally started as a fork of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main).
<br></br>

# Installation and Dependencies

To build POD-Attention, run the following command:
```sh
python setup.py install 
```

# Running POD-Attention

POD-Attention's API has been integrated into vAttention's fork of Sarathi-serve. You can run the associated backend by running sarathi-serve's benchmark runner with the attention backend `'FA_POD'` or `'FA_POD_MEGACACHE'`.
