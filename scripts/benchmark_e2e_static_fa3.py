import subprocess
import sys
import os
import utils
import json


# for quick testing
COMPUTE_UNITS = [2,3]
mapping = []
for gpu in COMPUTE_UNITS:
    mapping.append(["node:10.3.32.206", gpu])

compute_mapping= json.dumps({"0": mapping})

gpu_mem_util = 0.9
exp_title = 'static_e2e'
models, attention_backends = {'01-ai/Yi-34B-200k', 'meta-llama/Meta-Llama-3-8B'},  ['fa_vattn_2mb', 'fa_vattn_v3_2mb', 'fa_paged_256', 'fi_page_256']
num_requests, context_lengths, pd_ratios = 50, [ 32768, 65536, 131072 ], [500,100, 50]
experiment_dir = 'experiments'
for model in models:
    for backend in attention_backends:
       for p_d in pd_ratios:
         for context_len in context_lengths:
            model_file_name = model.split('/')[-1]
            tp_dim = 2
            max_batch_size = 300
            max_tokens = context_len
            kv_block_size = utils.get_block_or_page_size(backend)
            attn_backend_arg = utils.get_backend(backend)
            command = [
                    'python', '../sarathi-lean/sarathi/benchmark/main.py',
                    '--model_name', model,
                    '--model_tensor_parallel_degree', f'{tp_dim}',
                    '--request_generator_provider', 'synthetic',
                    '--synthetic_request_generator_length_provider', 'uniform',
                    '--synthetic_request_generator_interval_provider', 'static',
                    '--uniform_request_length_generator_max_tokens', str(context_len),
                    '--uniform_request_length_generator_min_tokens', str(context_len),
                    '--uniform_request_length_generator_prefill_to_decode_ratio', str(p_d),
                    '--replica_scheduler_provider', 'vllm',
                    '--trace_request_length_generator_prefill_scale_factor', '1',
                    '--trace_request_length_generator_decode_scale_factor', '1',
                    '--replica_scheduler_max_batch_size', str(max_batch_size),
                    '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),
                    '--model_max_model_len', str(max_tokens),
                    '--metrics_store_enable_op_level_metrics', 'false',
                    '--metrics_store_keep_individual_batch_metrics', 'true',
                    '--output_dir', f'{experiment_dir}/{exp_title}/model_{model_file_name}_tp_{tp_dim}_attn_{backend}_cl_{context_len}_pd_{p_d}_reqs_{num_requests}/',
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_generator_max_tokens', str(max_tokens),
                    '--model_block_size', str(kv_block_size),
                    '--model_attention_backend', f'{attn_backend_arg}',
                    '--gpu_memory_utilization', f'{gpu_mem_util}',
                    '--replica_resource_mapping', compute_mapping,
                ]
            print("Running command:", " ".join(command))
            try:
                subprocess.run(command, check=True)
            except:
                continue
