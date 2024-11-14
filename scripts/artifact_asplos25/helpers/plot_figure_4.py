import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/figure_4/')

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})


colors = {
    'Yi-6B': 'chocolate',
    'Llama-3-8B': 'green',
    'Yi-34B': 'blue',
}

linestyles = {
    'Yi-6B': '-',
    'Llama-3-8B': '--',
    'Yi-34B': '-.',
}

markers = {
    'Yi-6B': 'o',
    'Llama-3-8B': 's',
    'Yi-34B': 'd',
}

def plot_figure(dfs, key, ylabel, y_gap):
    plt.figure(figsize=(16, 6))
    y_max = -1
    models = list(dfs.keys())
    if len(models) == 0:
        return
    batch_sizes = dfs[models[0]]['bs']
    x_ticks = np.arange(len(batch_sizes))
    for i, model in enumerate(models):
        plt.plot(x_ticks, dfs[model][key], label=model, color=colors[model], linestyle = linestyles[model], linewidth=2, markersize=10, marker=markers[model])
        y_max = max(y_max, dfs[model][key].max())

    plt.xlabel('Batch Size', fontsize=28, fontweight='bold')
    plt.ylabel(ylabel, fontsize=28, fontweight='bold')
    x_labels = [str(bs) for bs in batch_sizes]
    plt.xticks(x_ticks, x_labels, fontsize=24)
    plt.yticks(np.arange(0, y_max*1.1, y_gap), fontsize=24)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right', ncols=3, fontsize=28)
    os.makedirs(os.path.join(root, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(root, f'plots/figure_4_{key}.pdf'))

record = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if model == 'yi-6b' else \
            'Yi-34B' if model == 'yi-34b' else \
            'Llama-3-8B' if model == 'llama-3-8b' else model

def read_perf_record(path):
    model = prettify_model_name(get_substring(path, 'figure_4/model_', '_bs_'))
    bs = int(get_substring(path, '_bs_', '/replica_0'))
    df = pd.read_csv(path)
    df = df[(df['batch_num_prefill_tokens'] == 0) & (df['batch_num_decode_tokens'] == bs)]
    if len(df) == 0:
        return
    latency = df['batch_execution_time'].min()
    if model not in record:
        record[model] = {}
    if bs not in record[model]:
        record[model][bs] = int(bs / latency)

def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file == 'batch_metrics.csv':
                path = os.path.join(root, file)
                read_perf_record(path)

kv_size = {
    'Yi-6B': 64,
    'Llama-3-8B': 128,
    'Yi-34B': 240,
}

read_logs()
dfs = {}
for model in record:
    dfs[model] = pd.DataFrame(record[model].items(), columns=['bs', 'throughput'])
    dfs[model] = dfs[model].sort_values(by='bs')
    dfs[model]['alloc_rate'] = (dfs[model]['throughput'] * kv_size[model]) / (1024)

plot_figure(dfs, 'throughput', 'Tokens/second', 1000)
plot_figure(dfs, 'alloc_rate', 'Memory Allocation \n (MB/second)', 100)