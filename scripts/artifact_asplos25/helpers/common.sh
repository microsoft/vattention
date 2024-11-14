#!/bin/bash

scripts=$(dirname $src)
root=$(dirname $scripts)
benchmark=$root/sarathi-lean/sarathi/benchmark/main.py

model_arg=""
set_model_arg() {
    if [ $1 == "yi-6b" ]; then
        model_arg="--model_name 01-ai/Yi-6B-200k --model_tensor_parallel_degree 1"
    elif [ $1 == "llama-3-8b" ]; then
        model_arg="--model_name meta-llama/Meta-Llama-3-8B --model_tensor_parallel_degree 2"
    elif [ $1 == "yi-34b" ]; then
        model_arg="--model_name 01-ai/Yi-34B-200k --model_tensor_parallel_degree 2"
    fi
}