import re
import os
import sys

sm_regex = re.compile('"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","(.*?)"')
bw_regex = re.compile('"dram__throughput.avg.pct_of_peak_sustained_elapsed","%","(.*?)"')
type_regex = re.compile('.*_(.*?).csv')
batch_sizes = ['8', '16', '32']
tile_dict = {1: "(128, 64)", 0 : "(64, 128)", 2: "(32, 64)", 3: "(16, 32)"}
tiles = [1, 0, 2, 3]

def extract(files, dict_sm, dict_bw):
    for file in files:
        with open(file, 'r') as f:
            content = f.read()
            type_name = type_regex.match(file).group(1)
            type_name = int(type_name) if type_name.isdigit() else type_name
            dict_sm[type_name] = {}
            dict_bw[type_name] = {}
            sm = sm_regex.findall(content)
            bw = bw_regex.findall(content)
            for i in range(len(sm)):
                sm[i] = float(sm[i])
                bw[i] = float(bw[i])
                dict_sm[type_name][batch_sizes[i]] = sm[i]
                dict_bw[type_name][batch_sizes[i]] = bw[i]

directory = sys.argv[1]

decode_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("decode_")]
sm_dict = {}
bw_dict = {}
extract(decode_files, sm_dict, bw_dict)

with open(directory + "compute.csv", "w") as f:
    f.write(f"tile_dim;")
    for t in batch_sizes:
        f.write(f"{t};")
    f.write("\n")
    for t in tiles:
        f.write(f"{tile_dict[t]};")
        for bs in batch_sizes:
            f.write(f"{sm_dict[t][bs]};")
        f.write("\n")

with open(directory + "dram.csv", "w") as f:
    f.write(f"tile_dim;")
    for t in batch_sizes:
        f.write(f"{t};")
    f.write("\n")
    for t in tiles:
        f.write(f"{tile_dict[t]};")
        for bs in batch_sizes:
            f.write(f"{bw_dict[t][bs]};")
        f.write("\n")