import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/figure_8/')

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ["FA_Paged", "FI_Paged", "FA_vAttention"]

colors = {
    'FI_Paged': 'chocolate',
    'FA_Paged': 'cadetblue',
    'FA_vAttention': 'lightgreen',
}

hatches = {
    'FI_Paged': '-',
    'FA_Paged': '\\',
    'FA_vAttention': '',
}

opacity=0.65

def plot_figure(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    models = df.index.tolist()
    width = 0.20
    x_pos = np.arange(len(models))
    fontsize = 36
    ax.bar(x_pos - 1*width, df["FA_Paged"], width, label="FA_Paged", color=colors["FA_Paged"], hatch=hatches["FA_Paged"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA_Paged"]):
        ax.text(i - 1*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos - 0*width, df["FI_Paged"], width, label="FI_Paged", color=colors["FI_Paged"], hatch=hatches["FI_Paged"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FI_Paged"]):
        ax.text(i - 0*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos + 1*width, df["FA_vAttention"], width, label="FA_vAttention", color=colors["FA_vAttention"], hatch=hatches["FA_vAttention"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA_vAttention"]):
        ax.text(i + 1*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    plt.xticks(x_pos, models, fontsize=fontsize)
    plt.yticks(np.arange(0, 10, 1), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Requests per minute", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper right', ncol=3, frameon=True, fontsize=fontsize-4)
    plt.tight_layout()
    os.makedirs(os.path.join(root, "plots"), exist_ok=True)
    plt.savefig(os.path.join(root, "plots/figure_8.pdf"))

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
    model = prettify_model_name(get_substring(path, 'figure_8/model_', '_attn_'))
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
for model in ["Yi-6B", "Llama-3-8B", "Yi-34B"]:
    if model not in df.index:
        df.loc[model] = 0
    for attn in ['FA_Paged', 'FI_Paged', 'FA_vAttention']:
        if attn not in df.loc[model]:
            df.loc[model, attn] = 0
df.fillna(0, inplace=True)
df = df.reindex(["Yi-6B", "Llama-3-8B", "Yi-34B"])
print(df)
plot_figure(df)