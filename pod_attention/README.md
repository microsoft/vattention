# Introduction
This repository contains the source code and profiling scripts for POD Attention. POD-Attention fuses prefill and decode attention kernels into a single optimized kernel that aims to saturate both GPU compute and memory simultaneously.

This repository originally started as a fork of [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/main).