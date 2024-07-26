import subprocess
import sys
import os
import utils

# configurable
num_requests = 50
gpu_mem_util = 0.9

models = utils.models
attention_backends = ['fa_paged_256', 'fi_paged_16', 'fa_vattn_2mb', 'fa_vattn_256kb', 'fi_vattn_2mb', 'fi_vattn_256kb']
context_lengths = [32768, 65536, 131072]
pd_ratios = [500, 100, 50]

# fixed
src, root, main = utils.get_paths()
experiment_dir = utils.static_experiment_dir

# for quick testing
if utils.args.test == True:
    models, attention_backends = {'yi-6b-1'}, ['fa_vattn_2mb']
    num_requests, context_lengths, pd_ratios = 5, [32768], [500]

for model in models:
    for backend in attention_backends:
       for p_d in pd_ratios:
         for context_len in context_lengths:
            model_logentry = utils.models[model]['logentry']
            tp_dim = utils.models[model]['tp']
            max_batch_size = min(256, num_requests) if 'yi-34b' in model.lower() else 16
            max_tokens = utils.get_max_context_length(backend, context_len)
            kv_block_size = utils.get_block_or_page_size(backend)
            attn_backend_arg = utils.get_backend(backend)
            command = [
                    'python', main,
                    '--model_name', utils.models[model]['hfrecord'],
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
                    '--output_dir', f'{experiment_dir}/model_{model_logentry}_tp_{tp_dim}_attn_{backend}_cl_{context_len}_pd_{p_d}_reqs_{num_requests}/',
                    '--synthetic_request_generator_num_requests', str(num_requests),
                    '--trace_request_generator_max_tokens', str(max_tokens),
                    '--model_block_size', str(kv_block_size),
                    '--model_attention_backend', f'{attn_backend_arg}',
                    '--gpu_memory_utilization', f'{gpu_mem_util}',
                ]
            print("Running command:", " ".join(command))
            try:
                subprocess.run(command, check=True)
            except:
                continue
