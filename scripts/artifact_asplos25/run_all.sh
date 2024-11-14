#!/bin/bash

src=$(dirname "$(realpath "$0")")
mkdir -p $src/plots

run_figure_2() {
    echo "========================================"
    echo "Running Figure-2"
    echo "========================================"
    python $src/run_figure_2.py
}

run_figure_3() {
    echo "========================================"
    echo "Running Figure-3"
    echo "========================================"
    $src/run_figure_3.sh
}

run_figure_6() {
    echo "========================================"
    echo "Running Figure-6"
    echo "========================================"
    $src/run_figure_6.sh
}

run_figure_7() {
    echo "========================================"
    echo "Running Figure-7"
    echo "========================================"
    $src/run_figure_7.sh
}

run_figure_8() {
    echo "========================================"
    echo "Running Figure-8"
    echo "========================================"
    $src/run_figure_8.sh
}

run_figure_9() {
    echo "========================================"
    echo "Running Figure-9"
    echo "========================================"
    $src/run_figure_9.sh
}

run_figure_11() {
    echo "========================================"
    echo "Running Figure-11"
    echo "========================================"
    $src/run_figure_11.sh
}

run_figure_2
run_figure_3
run_figure_6
run_figure_7
run_figure_11
run_figure_8
run_figure_9