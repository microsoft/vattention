import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/figure_9/')

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ["FA_Paged", "FI_Paged", "FA_vAttention"]

colors = {
    'FA_Paged': 'red',
    'FI_Paged': 'blue',
    'FA_vAttention': 'green',
}

linestyles = {
    'FA_Paged': '-',
    'FI_Paged': '-',
    'FA_vAttention': '--',
}

def plot_figure(model, qps, dfs):
    plt.figure(figsize=(18, 10))
    for attn in configs:
        df = dfs[attn]
        if df is not None:
            df = df.sort_values(by='request_e2e_time')
            df['cdf'] = df['request_e2e_time'].rank(method='first') / len(df)
            plt.plot(df['request_e2e_time'], df['cdf'], label=attn, color=colors[attn], linestyle=linestyles[attn], linewidth=3)
    plt.xlabel('Request Execution Latency (seconds)', fontsize=40, fontweight='bold')
    plt.ylabel('CDF', fontsize=40, fontweight='bold')
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.ylim(0, 1)
    plt.grid(axis='x')
    plt.grid(axis='y')
    plt.title(f'{model} (QPS={qps})', fontsize=36, fontweight='bold')
    plt.legend(loc='lower right', fontsize=36, ncol=1)
    os.makedirs(os.path.join(root, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(root, f'plots/figure_9_{model}_{qps}.pdf'))
    plt.close()

def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if model == 'yi-6b' else \
            'Yi-34B' if model == 'yi-34b' else \
            'Llama-3-8B' if model == 'llama-3-8b' else model

def prettify_attn_name(attn):
    return 'FA_Paged' if attn == 'fa_paged' else \
            'FI_Paged' if attn == 'fi_paged' else \
            'FA_vAttention' if attn == 'fa_vattn' else \
            'FI_vAttention' if attn == 'fi_vattn' else attn

record = {}
def read_perf_record(path):
    model = prettify_model_name(get_substring(path, 'figure_9/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '_qps_'))
    qps = get_substring(path, '_qps_', '/replica_0')
    if model not in record:
        record[model] = {}
    if qps not in record[model]:
        record[model][qps] = {}
    if attn not in record[model][qps]:
        record[model][qps][attn] = pd.read_csv(path)

def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(root, file)
                read_perf_record(path)

read_logs()
for model in record:
    dfs = {}
    for qps in record[model]:
        dfs["FA_Paged"] = record[model][qps]['FA_Paged'] if 'FA_Paged' in record[model][qps] else None
        dfs["FI_Paged"] = record[model][qps]['FI_Paged'] if 'FI_Paged' in record[model][qps] else None
        dfs["FA_vAttention"] = record[model][qps]['FA_vAttention'] if 'FA_vAttention' in record[model][qps] else None
        plot_figure(model, qps, dfs)