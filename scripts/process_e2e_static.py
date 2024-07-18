import os
import pandas as pd
import utils

src, root, main = utils.get_paths()
experiment_dir = utils.static_experiment_dir

logs = utils.get_output_files(experiment_dir, 'sequence_metrics.csv')

def get_config_info(log):
    model = utils.extract_substr(log, 'model_', '_tp_')
    tp = utils.extract_substr(log, '_tp_', '_attn_')
    attn = utils.extract_substr(log, '_attn_', '_cl_')
    context_len = utils.extract_substr(log, '_cl_', '_pd_')
    p_d = utils.extract_substr(log, '_pd_', '_reqs_')
    num_reqs = utils.extract_substr(log, '_reqs_', '/')
    return model, tp, attn, context_len, p_d, num_reqs

print('model;tp;attn;context_len;p_d;num_requests;makespan')
for log in logs:
    model, tp, attn, context_len, p_d, num_reqs = get_config_info(log)
    df = pd.read_csv(log)
    # in a static trace, all requests arrive at t=0, hence the longest
    # request is also the makespan of the trace
    makespan = round(df['request_e2e_time'].max(), 3)
    print(f'{model};{tp};{attn};{context_len};{p_d};{num_reqs};{makespan}')