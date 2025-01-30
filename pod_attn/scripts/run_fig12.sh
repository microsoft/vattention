#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

TEST_CHECK=false
if [[ "$1" == "--test" ]]; then
    TEST_CHECK=true
    num_reqs=10
fi

models="yi-6b llama-3-8b llama-2-7b"
attn_backends="fa_pod fa_vattn" 
schedulars="sarathi vllm"

context_length=16384
bs=999
run_experiments() {
    for model in $models; do
        set_model_arg $model
        for attn_backend in $attn_backends; do
            for schedular in $schedulars; do
                if [[ $schedular == "vllm" && $attn_backend == "fa_pod" ]]; then
                    continue
                fi
                set_file_name $attn_backend $schedular 
                # adjust arguments
                num_reqs=2000
                if [[ $model == "yi-6b" ]]; then
                    num_reqs=1000
                fi
		if [[ "$TEST_CHECK" == true ]]; then
    			num_reqs=10  # Set to 10 if test mode is enabled
	 	fi	
                block_size=2097152 # 2MB
                output_dir="$src/logs/figure_12/model_${model}_attn_${file_name}"
                echo -e "\n====================================================================================="
                echo "[Figure-12] Running Config ==> Model: $model Batch Size: $bs Attention Backend: $attn_backend"
                echo "\n                     Scheduler: $schedular Context Length: $context_length"
                echo -e "======================================================================================\n"
                #: '
                echo $model_arg
                python $benchmark $model_arg \
                    --request_generator_provider synthetic \
                    --synthetic_request_generator_length_provider uniform \
                    --synthetic_request_generator_interval_provider static \
                    --uniform_request_length_generator_max_tokens $context_length \
                    --uniform_request_length_generator_min_tokens $context_length \
                    --replica_scheduler_provider $schedular \
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
                #
            done
        done
    done
}

run_experiments
python $src/helpers/pod_attn_plot_figure_12.py
mv $src/plots/figure_12.pdf $src/../graphs/.
