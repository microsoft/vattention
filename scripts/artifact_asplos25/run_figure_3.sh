#!/bin/bash

src=$(dirname "$(realpath "$0")")
python $src/helpers/run_figure_3.py
python $src/helpers/plot_figure_3.py

