#!/bin/bash

src=$(dirname "$(realpath "$0")")
python $src/helpers/run_figure_2.py
python $src/helpers/plot_figure_2.py
