#!/bin/bash

FULL_TEST_CMD="python tests/banner_fig_comparison.py"

c0="--p_bs 1 --p_cs 1024 --p_cl 12288 --d_bs 80 --d_cl 12288"
c1="--p_bs 1 --p_cs 12288 --p_cl 12288 --d_bs 220 --d_cl 12288"
c2="--p_bs 1 --p_cs 16384 --p_cl 16384 --d_bs 250 --d_cl 12288"

run_full() {
    ${FULL_TEST_CMD} $c0
    ${FULL_TEST_CMD} $c1
    ${FULL_TEST_CMD} $c2
}

#profile_prefills
#profile_decodes
run_full