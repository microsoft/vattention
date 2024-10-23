# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [128]
IS_CAUSAL = ["false", "true"]
FUSED_OPS = [0, 9, 11, 13]
DECODE_SPLIT = ["false", "true"]
KERNEL_IMPL_TEMPLATE_FWD = """#include "fused_fwd_launch_template.h"
template<>
void run_fused_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_fused_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}
"""
FUSED_IMPL_TEMPLATE_FWD ="""#include "fused_fwd_launch_template.h"
template<>
void run_true_fused_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}, {FUSED_OP}, {DECODE_SPLIT}>(Flash_fwd_params &params_prefill, Flash_fwd_params &params_decode, cudaStream_t stream) {{
    run_true_fused_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}, {FUSED_OP}, {DECODE_SPLIT}>(params_prefill, params_decode, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLIT = """#include "fused_fwd_launch_template.h"

template void run_fused_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream);
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    is_causal: bool
    direction: str
    fused_op: Optional[int] = None
    decode_split: Optional[bool] = False

    @property
    def template(self) -> str:
        if self.direction == "fused_fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal
            )
        elif self.direction == "truefused_fwd":
            return FUSED_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal, FUSED_OP=self.fused_op, DECODE_SPLIT=self.decode_split
            )
        else:
            return KERNEL_IMPL_TEMPLATE_FWD_SPLIT.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal
            )

    @property
    def filename(self) -> str:
        return f"{self.direction}_hdim{self.head_dim}_{self.dtype}_{'causal_' if self.is_causal == 'true' else ''}{'split_' if self.decode_split == 'true' else ''}{f'fo{self.fused_op}_' if self.fused_op else ''}sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for direction in ["fused_fwd", "fused_fwd_split"]:
        for dtype, head_dim, is_causal, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, IS_CAUSAL, SM):
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction)
    for direction in ["truefused_fwd"]:
        for dtype, head_dim, is_causal, sm, fused_op, decode_split in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, IS_CAUSAL, SM, FUSED_OPS, DECODE_SPLIT):
            if decode_split == "true" and fused_op != 0:
                continue
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction, fused_op=fused_op, decode_split=decode_split)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)
    print('"fused_ampere/' + kernel.filename + '",')


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
