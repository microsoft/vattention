#!/bin/bash

scripts=$(dirname $src)
root=$(dirname $scripts)
benchmark=$root/sarathi-lean/sarathi/benchmark/main.py

model_arg=""
set_model_arg() {
    if [[ "$1" == "yi-6b" ]]; then
        model_arg="--model_name 01-ai/Yi-6B-200k --model_tensor_parallel_degree 1 --uniform_request_length_generator_prefill_to_decode_ratio 8 --sarathi_scheduler_chunk_size 512" 
    elif [[ "$1" == "llama-3-8b" ]]; then
        model_arg="--model_name meta-llama/Meta-Llama-3-8B --model_tensor_parallel_degree 2 --uniform_request_length_generator_prefill_to_decode_ratio 17 --sarathi_scheduler_chunk_size 1024"
    elif [[ "$1" == "yi-34b" ]]; then
        model_arg="--model_name 01-ai/Yi-34B-200k --model_tensor_parallel_degree 2"
    elif [[ "$1" == "llama-2-7b" ]]; then
        model_arg="--model_name meta-llama/Llama-2-7b-hf --model_tensor_parallel_degree 2 --uniform_request_length_generator_prefill_to_decode_ratio 64 --sarathi_scheduler_chunk_size 1024"
    fi
}

file_name=""
set_file_name() {
    if [[ "$1" == "fa_pod" ]]; then
        file_name="fa_pod"
    elif [[ "$1" == "fa_vattn" ]]; then
        if [[ "$2" == "sarathi" ]]; then
            file_name="fa_serial"
        else
            file_name="fa_vllm"
        fi
    fi
}