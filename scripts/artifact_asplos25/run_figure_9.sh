#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

online_trace_file=$src/traces/arxiv_long_online.csv
num_reqs=100

if [[ "$1" == "--full" ]]; then
    num_reqs=512
fi

models="yi-6b llama-3-8b yi-34b"
attn_backends="fa_paged fi_paged fa_vattn"

run_experiments() {
    for model in $models; do
        set_model_arg $model
        for attn_backend in $attn_backends; do
            block_size=16
            # set kvcache block size for different attention backends
            [[ $attn_backend == "fa_paged" ]] && block_size=256
            [[ $attn_backend == "fa_vattn" ]] && block_size=2097152
            model_qps="0.2 0.25"
            [[ $model == "llama-3-8b" ]] && model_qps="0.25 0.3"
            [[ $model == "yi-34b" ]] && model_qps="0.1 0.125"
            for qps in $model_qps; do
                output_dir="$src/logs/figure_9/model_${model}_attn_${attn_backend}_qps_${qps}/"
                echo -e "\n================================================================================="
                echo "[Figure-9] Running Config ==> Model: $model Attention Backend: $attn_backend QPS: $qps"
                echo -e "==================================================================================\n"
                #: '
                python $benchmark $model_arg \
                    --request_generator_provider synthetic \
                    --synthetic_request_generator_interval_provider poisson \
                    --poisson_request_interval_generator_qps 0.2 \
                    --replica_scheduler_provider vllm \
                    --vllm_scheduler_max_tokens_in_batch 65536 \
                    --model_max_model_len 65536 \
                    --metrics_store_enable_op_level_metrics false \
                    --metrics_store_keep_individual_batch_metrics true \
                    --output_dir $output_dir \
                    --synthetic_request_generator_num_requests $num_reqs \
                    --trace_request_length_generator_min_tokens  16384 \
                    --trace_request_length_generator_max_tokens  65536 \
                    --trace_request_length_generator_prefill_scale_factor 1 \
                    --trace_request_length_generator_trace_file $online_trace_file \
                    --model_block_size $block_size \
                    --model_attention_backend $attn_backend \
                    --gpu_memory_utilization 0.9
                #'
            done
        done
    done
}

run_experiments
python $src/helpers/plot_figure_9.py