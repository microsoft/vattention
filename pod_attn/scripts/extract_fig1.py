import re
import os
import sys

sm_regex = re.compile('"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","(.*?)"')
bw_regex = re.compile('"dram__throughput.avg.pct_of_peak_sustained_elapsed","%","(.*?)"')
type_regex = re.compile('.*_(.*?).csv')

def extract(files, dict):
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            type_name = type_regex.match(file).group(1)
            type_name = int(type_name) if type_name.isdigit() else type_name
            sm = sm_regex.findall(content)
            bw = bw_regex.findall(content)
            dict[type_name] = (sm, bw)

rt_regex = re.compile(r';([0-9]+\.[0-9]+);([0-9]+\.[0-9]+);([0-9]+\.[0-9]+);([0-9]+\.[0-9]+);([0-9]+\.[0-9]+);([0-9]+\.[0-9]+)')

def runtime(files, dict):
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            type_name = type_regex.match(file).group(1)
            rt = rt_regex.findall(content)[0]
            dict[type_name] = rt

directory = sys.argv[1]

decode_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("decode_")]
decode_dict = {}
extract(decode_files, decode_dict)
with open(directory + "decode.csv", "w") as f:
    f.write("model;bs;sm-util;dram-util\n")
    for t in sorted(decode_dict.keys()):
        f.write(f";{t};{decode_dict[t][0][0]};{decode_dict[t][1][0]}\n")

prefill_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("prefill_")]
prefill_dict = {}
extract(prefill_files, prefill_dict)
with open(directory + "prefill.csv", "w") as f:
    f.write("model;cl;sm-util;dram-util\n")
    for t in sorted(prefill_dict.keys()):
        f.write(f";{t};{prefill_dict[t][0][0]};{prefill_dict[t][1][0]}\n")

pod_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("fused_")]
pod_dict = {}
extract(pod_files, pod_dict)
with open(directory + "fused.csv", "w") as f:
    f.write("model;bs;sm-util;dram-util\n")
    for t in sorted(pod_dict.keys()):
        f.write(f";{t};{pod_dict[t][0][0]};{pod_dict[t][1][0]}\n")

rt_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("full_")]
rt = {}
runtime(rt_files, rt)
with open(directory + "runtime.csv", "w") as f:
    f.write("config;fa_p;fa_d;fi_p;fi_d;fi_batched;pod\n")
    for t in sorted(rt.keys()):
        f.write(f"{t};{rt[t][0]};{rt[t][1]};{rt[t][2]};{rt[t][3]};{rt[t][4]};{rt[t][5]}\n")