#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

offline_trace_file=$src/traces/arxiv_long_offline.csv
num_reqs=100

# Check for optional argument to see if the full trace should be used
if [[ "$1" == "--full" ]]; then
    num_reqs=$(( $(wc -l < $offline_trace_file) ))
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
            output_dir="$src/logs/figure_8/model_${model}_attn_${attn_backend}/"
            echo -e "\n========================================================================"
            echo "[Figure-8] Running Config ==> Model: $model Attention Backend: $attn_backend"
            echo -e "=========================================================================\n"
            #: '
            python $benchmark $model_arg \
                --request_generator_provider synthetic \
                --synthetic_request_generator_interval_provider static \
                --replica_scheduler_provider sarathi \
                --sarathi_scheduler_chunk_size 16384 \
                --vllm_scheduler_max_tokens_in_batch 196608 \
                --model_max_model_len 262144 \
                --metrics_store_enable_op_level_metrics false \
                --metrics_store_keep_individual_batch_metrics true \
                --output_dir $output_dir \
                --synthetic_request_generator_num_requests $num_reqs \
                --trace_request_length_generator_min_tokens  65536 \
                --trace_request_length_generator_max_tokens  196608 \
                --trace_request_length_generator_prefill_scale_factor 1 \
                --trace_request_length_generator_trace_file $offline_trace_file \
                --model_block_size $block_size \
                --model_attention_backend $attn_backend \
                --gpu_memory_utilization 0.9
                #'
        done
    done
}

run_experiments
python $src/helpers/plot_figure_8.py