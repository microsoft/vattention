#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

models="yi-6b llama-3-8b yi-34b"
attn_backends="fa_paged fi_paged fa_vattn fi_vattn"
context_lengths="2048 4096 8192 16384 32768 65536 131072 196608"

run_experiments() {
    for model in $models; do
        set_model_arg $model
        for cl in $context_lengths; do
            for attn_backend in $attn_backends; do
                # adjust arguments
                num_reqs=10
                block_size=16
                max_model_len=$cl
                max_batch_tokens=$cl
                scheduler="vllm"
                pd_ratio=500
                # limit the number of requests and decode tokens for quick experiments
                [[ $cl -gt 16384 ]] && num_reqs=2
                [[ $cl -gt 16384 ]] && pd_ratio=1000
                # set kvcache block size for different attention backends
                [[ $attn_backend == "fa_paged" ]] && block_size=256
                [[ $attn_backend == "fa_vattn" || $attn_backend == "fi_vattn" ]] && block_size=2097152
                # set max model lengh to a power of 2
                [[ $cl == 196144 ]] && max_model_len=262144 # corresponds to 2MB
                [[ $cl == 196144 ]] && max_batch_tokens=262144
                # enable prefill chunking for yi-34b as runs into OOM for long contexts without prefill chunking
                [[ $model == "yi-34b" ]] && scheduler="sarathi"
                [[ $model == "yi-34b" ]] && max_batch_tokens=16384
                output_dir="$src/logs/figure_6/model_${model}_cl_${cl}_attn_$attn_backend/"
                echo -e "\n===================================================================================="
                echo "[Figure-6] Running Config ==> Model: $model Context Length: $cl Attention Backend: $attn_backend"
                echo -e "=====================================================================================\n"
                #: '
                python $benchmark $model_arg \
                    --request_generator_provider synthetic \
                    --synthetic_request_generator_length_provider uniform \
                    --synthetic_request_generator_interval_provider static \
                    --uniform_request_length_generator_max_tokens $cl \
                    --uniform_request_length_generator_min_tokens $cl \
                    --uniform_request_length_generator_prefill_to_decode_ratio $pd_ratio \
                    --replica_scheduler_provider $scheduler \
                    --trace_request_length_generator_prefill_scale_factor 1 \
                    --trace_request_length_generator_decode_scale_factor 1 \
                    --replica_scheduler_max_batch_size 1 \
                    --vllm_scheduler_max_tokens_in_batch $max_batch_tokens \
                    --model_max_model_len $max_model_len \
                    --metrics_store_enable_op_level_metrics false \
                    --metrics_store_keep_individual_batch_metrics true \
                    --output_dir $output_dir \
                    --synthetic_request_generator_num_requests $num_reqs \
                    --trace_request_generator_max_tokens $cl \
                    --model_block_size $block_size \
                    --model_attention_backend $attn_backend \
                    --gpu_memory_utilization 0.9
                #'
            done
        done
    done
}

run_experiments
python $src/helpers/plot_figure_6.py
attn_backends="fa_vattn_megacache fa_vattn_megacache_sync"
