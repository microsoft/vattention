#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

models="yi-6b llama-3-8b yi-34b"
attn_backends="fa_paged fi_paged fa_vattn"
batch_sizes="1 2 4 8 12 16 32"

context_length=16384
run_experiments() {
    for model in $models; do
        set_model_arg $model
        for attn_backend in $attn_backends; do
            [[ $model == "yi-34b" ]] && batch_sizes="1 2 4 8 12 16"
            for bs in $batch_sizes; do
                # adjust arguments
                num_reqs=$(( bs*2 ))
                block_size=16
                # set kvcache block size for different attention backends
                [[ $attn_backend == "fa_paged" ]] && block_size=256
                [[ $attn_backend == "fa_vattn" || $attn_backend == "fi_vattn" ]] && block_size=2097152 # 2MB
                output_dir="$src/logs/figure_7/model_${model}_bs_${bs}_attn_${attn_backend}/"
                echo -e "\n====================================================================================="
                echo "[Figure-7] Running Config ==> Model: $model Batch Size: $bs Attention Backend: $attn_backend"
                echo -e "======================================================================================\n"
                #: '
                python $benchmark $model_arg \
                    --request_generator_provider synthetic \
                    --synthetic_request_generator_length_provider uniform \
                    --synthetic_request_generator_interval_provider static \
                    --uniform_request_length_generator_max_tokens $context_length \
                    --uniform_request_length_generator_min_tokens $context_length \
                    --uniform_request_length_generator_prefill_to_decode_ratio 500 \
                    --replica_scheduler_provider vllm \
                    --trace_request_length_generator_prefill_scale_factor 1 \
                    --trace_request_length_generator_decode_scale_factor 1 \
                    --replica_scheduler_max_batch_size $bs \
                    --vllm_scheduler_max_tokens_in_batch $context_length \
                    --model_max_model_len $context_length \
                    --metrics_store_enable_op_level_metrics false \
                    --metrics_store_keep_individual_batch_metrics true \
                    --output_dir $output_dir \
                    --synthetic_request_generator_num_requests $num_reqs \
                    --trace_request_generator_max_tokens $context_length \
                    --model_block_size $block_size \
                    --model_attention_backend $attn_backend \
                    --gpu_memory_utilization 0.9
                #'
            done
        done
    done
}

run_experiments
python $src/helpers/plot_figure_7.py