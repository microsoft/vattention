import subprocess
import sys
import os
import utils

# configurable
num_requests = 256
gpu_mem_util = 0.9
max_batch_size = 256

attention_backends = ['fa_paged', 'fi_paged', 'fa_vattn', 'fi_vattn']
qps_values = [0.4, 0.8, 1, 2, 4, 6]
chunk_size = 4096

# fixed
src, root, main = utils.get_paths()
experiment_dir = os.path.join(root, 'experiments', 'e2e_dynamic_eval')
dataset_path = os.path.join(root, 'sarathi-lean', utils.dataset_subpath)

# for quick testing
models, attention_backends = {'01-ai/Yi-6B-200k'}, ['fa_paged', 'fa_vattn']
num_requests, qps_values = 16, [1]

for model in models:
    for qps in qps_values:
        for backend in attention_backends:
            model_file_name = utils.models[model]['logentry']
            tp_dim = utils.models[model]['tp']
            kv_block_size = 2048 if 'fa' in backend.lower() else 16
            max_tokens = utils.get_max_context_length(backend, utils.MAX_CONTEXT_LENGTH_DYNAMIC_TRACES)
            command = [
                'python', main,
                    '--model_name', model,
                    '--model_tensor_parallel_degree', f'{tp_dim}',
                    '--request_generator_provider', 'synthetic',
                    '--synthetic_request_generator_length_provider', 'trace',
                    '--synthetic_request_generator_interval_provider', 'poisson', #'static',
                    '--poisson_request_interval_generator_qps', f'{qps}',
                    '--trace_request_length_generator_trace_file', f'{dataset_path}',
                    '--replica_scheduler_provider', 'vllm',
                    # '--replica_scheduler_provider', 'sarathi',
                    # '--sarathi_scheduler_chunk_size', str(chunk_size),
                    '--trace_request_length_generator_prefill_scale_factor', '1',
                    '--trace_request_length_generator_decode_scale_factor', '1',
                    '--replica_scheduler_max_batch_size', str(max_batch_size),
                    '--vllm_scheduler_max_tokens_in_batch', str(max_tokens),
                    '--model_max_model_len', str(max_tokens),
                    '--metrics_store_enable_op_level_metrics', 'false',
                    '--metrics_store_keep_individual_batch_metrics', 'false',
                     '--output_dir', f'{experiment_dir}/{utils.dataset_name}/{model_file_name}_attn_{backend}_qps_{qps}/',
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_length_generator_max_tokens', str(max_tokens),
                    '--trace_request_length_generator_min_tokens', str(0),
                    '--model_block_size', f'{kv_block_size}',
                    '--model_attention_backend', f'{backend}',
                    '--gpu_memory_utilization', f'{gpu_mem_util}',
                ]
            # assert dataset_name in dataset_path
            print("Running command:", " ".join(command))

            try:
                subprocess.run(command, check=True)
            except:
                continue
