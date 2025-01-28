OUTPUT ?= output/
GRAPHS ?= graphs/

start:
	mkdir -p ${OUTPUT}
	mkdir -p ${GRAPHS}

run_figure1: start
	bash tests/banner_fig.sh fa
	bash tests/banner_fig.sh pod
	bash tests/banner_fig.sh perf

gen_figure1: start
	python scripts/extract_fig1.py ${OUTPUT}/fig1/
	python scripts/plot_fig1.py ${OUTPUT}/fig1/ ${GRAPHS}

figure1: start
	$(MAKE) run_figure1
	$(MAKE) gen_figure1