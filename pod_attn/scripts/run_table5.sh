#!/bin/bash

src=$(dirname "$(realpath "$0")")
source $src/helpers/common.sh

online_trace_file=$src/traces/arxiv_sample.csv
num_reqs=2048

if [[ "$1" == "--test" ]]; then
    num_reqs=10
fi

model="llama-3-8b"
attn_backends="fa_pod fa_vattn"
schedulars="vllm sarathi"

run_experiments() {
    set_model_arg $model
    for schedular in $schedulars; do
        for attn_backend in $attn_backends; do
            if [[ "$attn_backend" == "fa_pod" && "$schedular" == "vllm" ]]; then
                continue
            fi
	    file_name_=$attn_backend
	    
	    if [[ "$schedular" == "vllm" ]]; then
		    file_name_="fa_vllm"  
	    fi
	
	    block_size=16
            # set kvcache block size for different attention backends
            block_size=2097152
            model_qps="0.85 0.95"
            for qps in $model_qps; do
                output_dir="$src/logs/table_5/model_${model}_attn_${file_name_}_qps_${qps}/"
                echo -e "\n========================================================================================================"
                echo "[Table-5] Running Config ==> Model: $model Attention Backend: $attn_backend QPS: $qps Schedular: $schedular"
                echo -e "==========================================================================================================\n"
                #: '
                python $benchmark $model_arg \
                    --request_generator_provider synthetic \
                    --synthetic_request_generator_interval_provider poisson \
                    --poisson_request_interval_generator_qps $qps \
                    --replica_scheduler_provider $schedular \
                    --vllm_scheduler_max_tokens_in_batch 65536 \
                    --model_max_model_len 65536 \
                    --metrics_store_enable_op_level_metrics false \
                    --metrics_store_keep_individual_batch_metrics true \
                    --output_dir $output_dir \
                    --synthetic_request_generator_num_requests $num_reqs \
                    --sarathi_scheduler_chunk_size 1024 \
                    --trace_request_length_generator_max_tokens 32768 \
                    --trace_request_length_generator_min_tokens 2048 \
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

#run_experiments
python $src/helpers/construct_table_5.py
mv $src/Table-5.csv $src/../graphs/.
