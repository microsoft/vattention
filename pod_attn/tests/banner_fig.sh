#!/bin/bash

NCU=$(which ncu)
PYTHON=$(which python)
NCU_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed"

TEST_CMD="${PYTHON} tests/banner_fig_profile.py"
FULL_TEST_CMD="${PYTHON} tests/banner_fig_comparison.py"
OUTPUT="output/fig1/"

profile_prefills() {
    p_cls="1024 2048 4096 8192 16384"
    for p_cl in $p_cls; do
        sudo $NCU --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv ${TEST_CMD} --stage prefill --p_bs 1 --p_cl $p_cl > ${OUTPUT}/prefill_$p_cl.csv
    done
}

profile_decodes() {
    d_batch_sizes="16 32 64 128 256"
    for d_bs in $d_batch_sizes; do
        sudo $NCU --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv ${TEST_CMD} --stage decode --d_bs $d_bs > ${OUTPUT}/decode_$d_bs.csv
    done
}

profile_fa() {
    profile_prefills
    profile_decodes
}

c0="--p_bs 1 --p_cs 1024 --p_cl 12288 --d_bs 80 --d_cl 12288"
c1="--p_bs 1 --p_cs 12288 --p_cl 12288 --d_bs 220 --d_cl 12288"
c2="--p_bs 1 --p_cs 16384 --p_cl 16384 --d_bs 250 --d_cl 12288"
#c3="--p_bs 1 --p_cs 12288 --p_cl 12288 --d_bs 220 --d_cl 12288"
profile_fused() {
    sudo $NCU --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv ${TEST_CMD} $c0 --stage fused > ${OUTPUT}/fused_c0.csv
    sudo $NCU --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv ${TEST_CMD} $c1 --stage fused > ${OUTPUT}/fused_c1.csv
    sudo $NCU --metrics $NCU_METRICS -k "regex:.*fwd.*" --csv ${TEST_CMD} $c2 --stage fused > ${OUTPUT}/fused_c2.csv
}

run_full() {
    ${FULL_TEST_CMD} $c0 > ${OUTPUT}/full_C0.csv
    ${FULL_TEST_CMD} $c1 > ${OUTPUT}/full_C1.csv
    ${FULL_TEST_CMD} $c2 > ${OUTPUT}/full_C2.csv
}

mkdir -p ${OUTPUT}

#profile_prefills
#profile_decodes
if [ "$1" == "fa" ]; then
    profile_fa
elif [ "$1" == "pod" ]; then
    profile_fused
elif [ "$1" == "perf" ]; then
    run_full
else
    echo "Usage: $0 [fa|pod|perf]"
fi