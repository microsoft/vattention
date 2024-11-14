import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs_async = os.path.join(root, 'logs/figure_11/model_llama-3-8b_attn_fa_vattn/replica_0/batch_metrics.csv')
logs_sync = os.path.join(root, 'logs/figure_11/model_llama-3-8b_attn_fa_vattn_sync/replica_0/batch_metrics.csv')

def check_logs():
    global logs_async, logs_sync
    if os.path.exists(logs_async) and os.path.exists(logs_sync):
        return
    if not os.path.exists(logs_async) and not os.path.exists(logs_sync):
        logs_async = os.path.join(root, 'logs/figure_11/model_llama-3-8b_attn_fa_vattn_megacache/replica_0/batch_metrics.csv')
        logs_sync = os.path.join(root, 'logs/figure_11/model_llama-3-8b_attn_fa_vattn_sync_megacache/replica_0/batch_metrics.csv')
        if os.path.exists(logs_async) and os.path.exists(logs_sync):
            return
    raise FileNotFoundError("Could not find logs")

check_logs()

df_async = pd.read_csv(logs_async)
df_sync = pd.read_csv(logs_sync)

# decode only iterations
df_async = df_async[(df_async['batch_num_prefill_tokens'] == 0)].iloc[:500]
df_sync = df_sync[(df_sync['batch_num_prefill_tokens'] == 0)].iloc[:500]

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})
fig, ax = plt.subplots(figsize=(16, 6))
async_times = df_async['batch_execution_time'] * 1000
sync_times = df_sync['batch_execution_time'] * 1000
y_max = max(async_times.max(), sync_times.max())
ax.plot(sync_times, label='Without overlapping', color='red', linestyle='--')
ax.plot(async_times, label='With Overlapping', color='green')
ax.set_xlabel('Decode Iteration', fontweight='bold', fontsize=28)
ax.set_ylabel('Latency (ms)', fontweight='bold', fontsize=28)
ax.set_ylim(0, y_max + 10)
ax.grid(axis='y', linestyle='--')
ax.grid(axis='x', linestyle='--')
ax.legend(loc='lower right', fontsize=28)
plt.tight_layout()
os.makedirs(os.path.join(root, "plots"), exist_ok=True)
plt.savefig(os.path.join(root, "plots/figure_11.pdf"))