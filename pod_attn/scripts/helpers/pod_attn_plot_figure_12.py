import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/figure_12/')
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ["FA_Serial", "fa_vllm", "fa_pod"]

colors = {
    'FA_Serial': 'chocolate',
    'fa_vllm': 'cadetblue',
    'fa_pod': 'lightgreen',
}

hatches = {
    'fa_vllm': '-',
    'FA_Serial': '\\',
    'fa_pod': '',
}

opacity=0.65

def plot_figure(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    models = df.index.tolist()
    width = 0.20
    x_pos = np.arange(len(models))
    fontsize = 36
    ax.bar(x_pos - 1*width, df["fa_vllm"], width, label="fa_vllm", color=colors["fa_vllm"], hatch=hatches["fa_vllm"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["fa_vllm"]):
        ax.text(i - 1*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos - 0*width, df["FA_Serial"], width, label="FA_Serial", color=colors["FA_Serial"], hatch=hatches["FA_Serial"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA_Serial"]):
        ax.text(i - 0*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos + 1*width, df["fa_pod"], width, label="fa_pod", color=colors["fa_pod"], hatch=hatches["fa_pod"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["fa_pod"]):
        ax.text(i + 1*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    plt.xticks(x_pos, models, fontsize=fontsize)
    plt.yticks(np.arange(0, 10, 1), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Requests per minute", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper right', ncol=3, frameon=True, fontsize=fontsize-4)
    plt.tight_layout()
    os.makedirs(os.path.join(root, "plots"), exist_ok=True)
    plt.savefig(os.path.join(root, "plots/figure_12.pdf"))

record = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if model == 'yi-6b' else \
            'Llama-2-7B' if model == 'llama-2-7b' else \
            'Llama-3-8B' if model == 'llama-3-8b' else model

def prettify_attn_name(attn):
    return 'fa_vllm' if attn == 'fa_vllm' else \
            'FA_Serial' if attn == 'fa_serial' else \
            'fa_pod' if attn == 'fa_vattn' else attn

def read_perf_record(path):
    model = prettify_model_name(get_substring(path, 'figure_12/model_', '_attn_'))
    attn = prettify_attn_name(get_substring(path, '_attn_', '/replica_0'))
    df = pd.read_csv(path)

    latency = df['request_e2e_time'].max()
    num_requests = len(df)
    if model not in record:
        record[model] = {}
    if attn not in record[model]:
        record[model][attn] = (num_requests * 60) / latency

def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(root, file)
                read_perf_record(path)

read_logs()
df = pd.DataFrame.from_dict(record).transpose()
#models = df.index.tolist()
for model in ["Yi-6B", "Llama-3-8B", "Llama-2-7B"]:
    if model not in df.index:
        df.loc[model] = 0
    for attn in ['fa_vllm', 'FA_Serial', 'fa_pod']:
        if attn not in df.loc[model]:
            df.loc[model, attn] = 0
df.fillna(0, inplace=True)
df = df.reindex(["Yi-6B", "Llama-3-8B", "Llama-2-7B"])
plot_figure(df)