import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

helpers = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(helpers)
logs = os.path.join(root, 'logs/table_5/')

num_requests = [2048]
record = {}
tbt_record = {}
tbt_stall_record = {}
def get_substring(string, start, end):
    return string[string.find(start)+len(start):string.find(end)]

def prettify_model_name(model):
    return 'Yi-6B' if model == 'yi-6b' else \
            'Llama-2-7B' if model == 'llama-2-7b' else \
            'Llama-3-8B' if model == 'llama-3-8b' else model

def prettify_attn_name(attn):
    return 'fa_vllm' if attn == 'fa_vllm' else \
            'FA_Serial' if attn == 'fa_vattn' else \
            'fa_pod' if attn == 'fa_pod' else attn

def read_perf_record(path):
    attn = prettify_attn_name(get_substring(path, '_attn_', '_qps_'))
    qps = get_substring(path, '_qps_', '/replica_0')
    df = pd.read_csv(path)
    latency_50 = df['request_e2e_time'].quantile(0.5)
    latency_99 = df['request_e2e_time'].quantile(0.99)
    ttft_50 = df['prefill_e2e_time'].quantile(0.5)
    ttft_99 = df['prefill_e2e_time'].quantile(0.99)
    if num_requests[0]!= len(df):
        print(f"Number of requests mismatch: {num_requests} != {len(df)}")
        num_requests[0] = len(df)
    if qps not in record:
        record[qps] = {}
    if attn not in record[qps]:
        record[qps][attn] = {'latency_p50': latency_50, 'latency_p99': latency_99, 'ttft_p50': ttft_50, 'ttft_p99': ttft_99}

def read_tbt_record(path):
    attn = prettify_attn_name(get_substring(path, '_attn_', '_qps_'))
    qps = get_substring(path, '_qps_', '/replica_0')
    df = pd.read_csv(path)
    tbt_50 = df['decode_token_execution_plus_preemption_time'].quantile(0.5)
    tbt_99 = df['decode_token_execution_plus_preemption_time'].quantile(0.99)
    if qps not in tbt_record:
        tbt_record[qps] = {}
    if attn not in tbt_record[qps]:
        tbt_record[qps][attn] = {'tbt_p50': tbt_50, 'tbt_p99': tbt_99}
    
def get_tbt_violations(path, deadline):
    df = pd.read_csv(path)
    filtered_df = df[df['decode_token_execution_plus_preemption_time_list'] > deadline/1000]
    filtered_df.loc[:, 'reqid'] = filtered_df['Decode Token Id'].apply(lambda x: x.split('_')[1])
    unique_reqids = filtered_df['reqid'].value_counts()
    num_unique_reqids = len(unique_reqids)
    return (num_unique_reqids/num_requests[0])*100

def read_tbt_stalls(path):
    attn = prettify_attn_name(get_substring(path, '_attn_', '_qps_'))
    qps = get_substring(path, '_qps_', '/replica_0')
    for deadline in [200, 500]:
        pct_stalls = get_tbt_violations(path, deadline)
        if qps not in tbt_stall_record:
            tbt_stall_record[qps] = {}
        if attn not in tbt_stall_record[qps]:
            tbt_stall_record[qps][attn] = {f'pct_stalls_{deadline}_ms': round(pct_stalls, 2)}
        

def read_logs():
    for root, dirs, files in os.walk(logs):
        for file in files:
            if file == 'sequence_metrics.csv':
                path = os.path.join(root, file)
                read_perf_record(path)
            if file == 'decode_token_execution_plus_preemption_time.csv':
                path = os.path.join(root, file)
                read_tbt_record(path)
            if file == 'decode_token_execution_plus_preemption_time_list.csv':
                path = os.path.join(root, file)
                read_tbt_stalls(path)

read_logs()
df = pd.DataFrame.from_dict(record).transpose()
df_tbt = pd.DataFrame.from_dict(tbt_record).transpose()
df_tbt_stalls = pd.DataFrame.from_dict(tbt_stall_record).transpose()

# Replace None / NaN with empty dictionaries to prevent errors
df = df.applymap(lambda x: x if isinstance(x, dict) else {})
df_tbt = df_tbt.applymap(lambda x: x if isinstance(x, dict) else {})
df_tbt_stalls = df_tbt_stalls.applymap(lambda x: x if isinstance(x, dict) else {})

# Expand each dictionary into separate columns
df = pd.concat(
    [df[col].apply(pd.Series).add_prefix(col + "_") for col in df.columns], axis=1
)
df_tbt = pd.concat(
    [df_tbt[col].apply(pd.Series).add_prefix(col + "_") for col in df_tbt.columns], axis=1
)
df_tbt_stalls = pd.concat(
    [df_tbt_stalls[col].apply(pd.Series).add_prefix(col + "_") for col in df_tbt_stalls.columns], axis=1
)


# Reset index and rename the column
df.reset_index(inplace=True)
df.rename(columns={"index": "QPS"}, inplace=True)

df_tbt.reset_index(inplace=True)
df_tbt.rename(columns={"index": "QPS"}, inplace=True)

df_tbt_stalls.reset_index(inplace=True)
df_tbt_stalls.rename(columns={"index": "QPS"}, inplace=True)

# Merge the DataFrames
df = pd.merge(df, df_tbt, on="QPS")
df = pd.merge(df, df_tbt_stalls, on="QPS")

# Save as CSV
df.to_csv("Table-5.csv", index=False)

# Print the formatted DataFrame
print(df)