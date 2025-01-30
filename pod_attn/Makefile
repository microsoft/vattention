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

run_figure6: start
	python tests/chunked_quantization.py > ${OUTPUT}/fig6.txt

gen_figure6: start
	python scripts/plot_fig6.py ${OUTPUT}/fig6.txt ${GRAPHS}/fig6.png

figure6: start
	$(MAKE) run_figure6
	$(MAKE) gen_figure6

build/micro_compute_mem: tests/micro_compute_mem.cu
	mkdir -p build
	nvcc -o build/micro_compute_mem tests/micro_compute_mem.cu -arch=sm_80

run_figure7: start build/micro_compute_mem
	./build/micro_compute_mem > ${OUTPUT}/fig7.txt

gen_figure7: start
	python scripts/extract_fig7.py ${OUTPUT}/fig7.txt > ${OUTPUT}/fig7_parsed.txt
	python scripts/plot_fig7.py ${OUTPUT}/fig7_parsed.txt ${GRAPHS}/fig7.png

figure7: start
	$(MAKE) run_figure7
	$(MAKE) gen_figure7

run_figure10: start
	./tests/tile_sizes.sh

gen_figure10: start
	python scripts/extract_fig10.py ${OUTPUT}/fig10/
	python scripts/plot_fig10.py ${OUTPUT}/fig10/ ${GRAPHS}/fig10

figure10: start
	$(MAKE) run_figure10
	$(MAKE) gen_figure10

run_figure11: start
	python tests/attn_sweep.py > ${OUTPUT}/fig11.txt

gen_figure11: start
	python scripts/plot_fig11.py ${OUTPUT}/fig11.txt ${GRAPHS}/fig11.png

figure11: start
	$(MAKE) run_figure11
	$(MAKE) gen_figure11

run_figure12: start
	bash scripts/run_fig12.sh # --test # Added target for figure12

figure12: start
	$(MAKE) run_figure12  # Makefile command for running figure12

run_table5: start
	bash scripts/run_table5.sh # --test

table5: start
	$(MAKE) run_table5

run_figure13: start
	python tests/sens_cta.py > ${OUTPUT}/fig13.txt

gen_figure13: start
	python scripts/plot_fig13.py ${OUTPUT}/fig13.txt ${GRAPHS}/fig13.png

figure13: start
	$(MAKE) run_figure13
	$(MAKE) gen_figure13

run_figure14: start
	python tests/sens_scheduling.py > ${OUTPUT}/fig14.txt

gen_figure14: start
	python scripts/plot_fig14.py ${OUTPUT}/fig14.txt ${GRAPHS}/fig14.png

figure14: start
	$(MAKE) run_figure14
	$(MAKE) gen_figure14

install_all: start
	python setup.py install
	bash /workspace/vattention/scripts/artifact_asplos25/install.sh
