#!/bin/bash

NCU=$(which ncu)
PYTHON=$(which python)
NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

TEST_CMD="${PYTHON} tests/fa_tile_sizes.py"
OUTPUT="output/fig10/"

mkdir -p ${OUTPUT}

d_batch_sizes="8 16 32"
for tile in 0 1 2 3; do
    echo "" > ${OUTPUT}/decode_${tile}.csv
    for d_bs in $d_batch_sizes; do
        sudo $NCU --metrics $NCU_METRICS -k "regex:.*fwd.*kernel" --csv ${TEST_CMD} --tile ${tile} --d_bs $d_bs >> ${OUTPUT}/decode_${tile}.csv
    done
done