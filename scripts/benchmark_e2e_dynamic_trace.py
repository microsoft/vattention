import subprocess
import sys
import os
import utils


# configurable
num_requests = 256
gpu_mem_util = 0.9
max_batch_size = 256

models = utils.models
attention_backends = ['fa_paged_256', 'fi_paged_16', 'fa_vattn_2mb', 'fa_vattn_256kb', 'fi_vattn_2mb', 'fi_vattn_256kb']
qps_values = [0.4, 0.8, 1, 2, 4, 6]
chunk_size = 4096

# fixed
src, root, main = utils.get_paths()
experiment_dir = utils.dynamic_experiment_dir
dataset_path = os.path.join(root, 'sarathi-lean', utils.dataset_subpath)

# for quick testing
if utils.args.test == True:
    models, attention_backends = {'yi-6b-1'}, ['fa_vattn_2mb']
    num_requests, qps_values = 8, [1]

for model in models:
    for qps in qps_values:
        for backend in attention_backends:
            model_logentry = utils.models[model]['logentry']
            tp_dim = utils.models[model]['tp']
            max_tokens = utils.get_max_context_length(backend, utils.MAX_CONTEXT_LENGTH_DYNAMIC_TRACES)
            kv_block_size = utils.get_block_or_page_size(backend)
            attn_backend_arg = utils.get_backend(backend)
            command = [
                'python', main,
                    '--model_name', utils.models[model]['hfrecord'],
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
                     '--output_dir', f'{experiment_dir}/dataset_{utils.dataset_name}_model_{model_logentry}_tp_{tp_dim}_attn_{backend}_qps_{qps}_reqs_{num_requests}/',
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_length_generator_max_tokens', str(max_tokens),
                    '--trace_request_length_generator_min_tokens', str(0),
                    '--model_block_size', f'{kv_block_size}',
                    '--model_attention_backend', f'{attn_backend_arg}',
                    '--gpu_memory_utilization', f'{gpu_mem_util}',
                ]
            # assert dataset_name in dataset_path
            print("Running command:", " ".join(command))

            try:
                subprocess.run(command, check=True)
            except:
                continue
