#all: vattn multi_attn reassign_vattn profile_vattn layers_vattn uvm_measure cu_measure
all: vattn

vattn:
	nvcc vattn.cu -o vattn

multi_attn:
	nvcc multi_vattn.cu -o multi_vattn

reassign_vattn:
	nvcc reassign_vattn.cu -o reassign_vattn

profile_vattn:
	nvcc profile_vattn.cu -o profile_vattn

remap_vattn:
	nvcc remap_vattn.cu -o remap_vattn

layers_vattn:
	nvcc layers_vattn.cu -o layers_vattn

uvm_measure:
	nvcc uvm_measure.cu -o uvm_measure

cu_measure:
	nvcc cu_measure.cu -lcuda -o cu_measure

clean:
	rm -f vattn

#rm -f vattn multi_vattn reassign_vattn profile_vattn remap_vattn layers_vattn uvm_measure cu_measure
