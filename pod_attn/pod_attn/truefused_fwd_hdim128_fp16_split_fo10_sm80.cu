// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "fused_fwd_launch_template.h"
template<>
void run_true_fused_mha_fwd_<cutlass::half_t, 128, false, 10, true>(Flash_fwd_params &params_prefill, Flash_fwd_params &params_decode, cudaStream_t stream) {
    run_true_fused_mha_fwd_hdim128<cutlass::half_t, false, 10, true>(params_prefill, params_decode, stream);
}
