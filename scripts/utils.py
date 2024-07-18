import os

src = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(src)
main = os.path.join(root, 'sarathi-lean', 'sarathi', 'benchmark', 'main.py')

# dataset_subpath = 'data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv'
# dataset_name = 'sharegpt'
dataset_subpath = 'data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv'
dataset_name = 'arxiv'

# upper limit on maximum context length for dynamic traces
MAX_CONTEXT_LENGTH_DYNAMIC_TRACES = 32768

models = {
    '01-ai/Yi-6B-200k': {'tp': 1, 'logentry': 'yi-6b'},
    'meta-llama/Meta-Llama-3-8B': {'tp': 2, 'logentry': 'llama-3-8b'},
    '01-ai/Yi-34B-200k': {'tp': 2, 'logentry': 'yi-34b'}
}

# vattention allocates memory in power of two while fa_paged/fi_paged
# sometimes run into illegal memory access if context length is more than
# a certain limit (e.g., 200K) and rope scaling is applied (observed for yi-6B).
# Hence, we set max context length as per the experiment to get around this problem.
# TODO(ashish) Fix it.
def get_max_context_length(attn_backend, context_len):
    if context_len & (context_len - 1) == 0:
        return context_len
    if '_paged' in attn_backend.lower():
        return min(200000, 2 ** (context_len.bit_length() + 1))
    return 2 ** (context_len.bit_length() + 1)

def get_paths():
    return src, root, main