import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/figure_6/')

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ["FA_Paged", "FI_Paged", "FA_vAttention", "FI_vAttention"]

config_labels = {
    'FA_Paged': 'FA_Paged',
    'FI_Paged': 'FI_Paged',
    'FA_vAttention': 'FA_vAttention',
    'FI_vAttention': 'FI_vAttention',
}

colors = {
    'FA_Paged': 'chocolate',
    'FI_Paged': 'green',
    'FA_vAttention': 'chocolate',
    'FI_vAttention': 'green',
}

linestyles = {
    'FA_Paged': '-',
    'FI_Paged': '-',
    'FA_vAttention': '--',
    'FI_vAttention': '--',
}

markerstyles = {
    'FA_Paged': 'o',
    'FI_Paged': 's',
    'FA_vAttention': 'D',
    'FI_vAttention': '^'
}

def plot_figure(df, figname, title):
    plt.figure(figsize=(10, 5))
    y_max = -1
    context_lengths = df['cl'].unique()
    x_ticks = np.arange(len(context_lengths))
    for i, cfg in enumerate(configs):
        plt.plot(x_ticks, df[cfg], label=config_labels[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=2, marker=markerstyles[cfg], markersize=7)
        y_max = max(y_max, df[cfg].max())

    plt.xlabel('Context Length', fontsize=24, fontweight='bold')
    plt.ylabel('Tokens/second', fontsize=24, fontweight='bold')
    x_labels = [str(cl//1024) + "K" for cl in context_lengths]
    plt.xticks(x_ticks, x_labels, fontsize=18)
    plt.yticks(np.arange(0, y_max*1.1, 3000), fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', ncols=1, fontsize=22)
    plt.savefig(figname)

record = {}
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

def read_perf_record(path):
    model = get_substring(path, 'figure_6/model_', '_cl_')
    cl = int(get_substring(path, '_cl_', '_attn_'))
    attn = get_substring(path, '_attn_', '/replica_0')
    attn = prettify_attn_name(attn)
    df = pd.read_csv(path)
    df = df[df['cdf'] >= 0.5].sort_values(by='cdf')
    latency_sorted = df['prefill_time_execution_plus_preemption'].sort_values()
    # take the lowest values of latency
    latency = latency_sorted.iloc[0]
    if len(latency_sorted) >= 5:
        latency = latency_sorted.iloc[0:5].mean()
    if model not in record:
        record[model] = {}
    if cl not in record[model]:
        record[model][cl] = {}
    if attn not in record[model][cl]:
        record[model][cl][attn] = int(cl / latency)

def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file == 'prefill_time_execution_plus_preemption.csv':
                path = os.path.join(root, file)
                read_perf_record(path)

read_logs()
df = pd.DataFrame.from_dict(record, orient='index').stack().apply(pd.Series).reset_index()
df.columns.values[0] = 'model'
df.columns.values[1] = 'cl'
df.fillna(0, inplace=True)
models = df['model'].unique()
for model in models:
    df_model = df[df['model'] == model].sort_values(by='cl')
    title = prettify_model_name(model)
    figname = os.path.join(root, f'plots/figure_6_{title}.pdf')
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plot_figure(df_model, figname, title)