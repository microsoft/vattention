#!/bin/bash

git clone https://github.com/Dao-AILab/flash-attention.git
git clone --recursive --single-branch --branch v2.1.0 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule
sync git submodule update --init --recursive
