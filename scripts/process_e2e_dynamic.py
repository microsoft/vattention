import os
import sys
import pandas as pd
import utils

src, root, main = utils.get_paths()
experiment_dir = utils.dynamic_experiment_dir

logs = utils.get_output_files(experiment_dir, 'sequence_metrics.csv')
def get_config_info(log):
    dataset = utils.extract_substr(log, 'dataset_', '_model_')
    model = utils.extract_substr(log, '_model_', '_tp_')
    tp = utils.extract_substr(log, '_tp_', '_attn_')
    attn = utils.extract_substr(log, '_attn_', '_qps_')
    qps = utils.extract_substr(log, '_qps_', '_reqs_')
    num_reqs = utils.extract_substr(log, '_reqs_', '/')
    return dataset, model, tp, attn, qps, num_reqs

print('dataset;model;tp;attn;qps;num_requests;p50;p90;p99')
for log in logs:
    dataset, model, tp, attn, qps, num_reqs = get_config_info(log)
    df = pd.read_csv(log)
    p50 = round(df['request_e2e_time_normalized'].quantile(0.5), 3)
    p90 = round(df['request_e2e_time_normalized'].quantile(0.9), 3)
    p99 = round(df['request_e2e_time_normalized'].quantile(0.99), 3)
    print(f'{dataset};{model};{tp};{attn};{qps};{num_reqs};{p50};{p90};{p99}')