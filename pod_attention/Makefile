SRC     = ./tests
BUILD   = ./build
SCRIPTS = ./scripts
OUTPUT  = ./output

# Setup the environment
setup_dirs: 
	mkdir -p ${BUILD}
	mkdir -p ${OUTPUT}

setup_ampere: setup_dirs
	python setup_ampere.py install

.PHONY: run_fused extract_fused run_fused_copy run_attention setup_ampere setup_dirs

# Compile microbenchmarks
${BUILD}/fused_micro.exe: ${SRC}/fused_compute_mem.cu setup_dirs
	nvcc $< -arch=sm_80 -o $@

${BUILD}/fused_micro_copy.exe: ${SRC}/fused_compute_mem_copy.cu setup_dirs
	nvcc $< -arch=sm_80 -o $@

# Run/extract microbenchmarks
run_fused: ${BUILD}/fused_micro.exe
	${BUILD}/fused_micro.exe > ${OUTPUT}/fused_micro.out

extract_fused:
	python ${SCRIPTS}/extract_fused.py ${OUTPUT}/fused_micro.out

run_fused_copy: ${BUILD}/fused_micro_copy.exe
	${BUILD}/fused_micro_copy.exe > ${OUTPUT}/fused_micro_copy.out

# Run benchmark
run_attention: setup_ampere
	python ${SRC}/fused_fa2_test.py > ${OUTPUT}/attention_bench.out
	$(MAKE) extract_atten > benchmark_results.out

extract_atten:
	python ${SCRIPTS}/extract_benchmark.py ${OUTPUT}/attention_bench.out

NCU = /home/t-adikamath/anaconda3/bin/ncu
PYTHON = /home/t-adikamath/anaconda3/bin/python
run_util_bs:
	sudo ${NCU} -k regex:".*fwd.*" ${PYTHON} ${SRC}/attention_util.py > ${OUTPUT}/ncu_attention_util.out
	$(MAKE) extract_util_bs > benchmark_util_bs.out

extract_util_bs:
	python ${SCRIPTS}/extract_util.py ${OUTPUT}/ncu_attention_util.out

run_util_cl:
	sudo ${NCU} -k regex:".*fwd.*" ${PYTHON} ${SRC}/attention_util_cl.py > ${OUTPUT}/ncu_attention_util_cl.out
	$(MAKE) extract_util_cl > benchmark_util_cl.out

extract_util_cl:
	python ${SCRIPTS}/extract_util.py ${OUTPUT}/ncu_attention_util_cl.out "1024 2048 4096 8192 16384"
