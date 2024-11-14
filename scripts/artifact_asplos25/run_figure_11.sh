#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

trace_file=$src/traces/sync_vs_async_trace.csv
num_reqs=300

models="llama-3-8b"
attn_backends="fa_vattn fa_vattn_sync"

run_experiments() {
    for model in $models; do
        set_model_arg $model
        for attn_backend in $attn_backends; do
            block_size=2097152 # corresponds to 2MB
            output_dir="$src/logs/figure_11/model_${model}_attn_${attn_backend}/"
            echo -e "\n========================================================================"
            echo "[Figure-11] Running Config ==> Model: $model Attention Backend: $attn_backend"
            echo -e "==========================================================================\n"
            #: '
            python $benchmark $model_arg \
                --request_generator_provider synthetic \
                --synthetic_request_generator_interval_provider static \
                --replica_scheduler_provider vllm \
                --vllm_scheduler_max_tokens_in_batch 32768 \
                --replica_scheduler_max_batch_size 64 \
                --model_max_model_len 32768 \
                --metrics_store_enable_op_level_metrics false \
                --metrics_store_keep_individual_batch_metrics true \
                --output_dir $output_dir \
                --synthetic_request_generator_num_requests $num_reqs \
                --trace_request_length_generator_prefill_scale_factor 1 \
                --trace_request_length_generator_trace_file $trace_file \
                --trace_request_length_generator_min_tokens 128 \
                --trace_request_length_generator_max_tokens 32768 \
                --model_block_size $block_size \
                --model_attention_backend $attn_backend \
                --gpu_memory_utilization 0.9
            #'
        done
    done
}

run_experiments
python $src/helpers/plot_figure_11.py