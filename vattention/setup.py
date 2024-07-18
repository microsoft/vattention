import os
from setuptools import setup, Extension
from torch.utils import cpp_extension

libtorch_path = os.getenv('LIBTORCH_PATH', '/ramyapra/sarathi-env/libtorch')
assert os.path.exists(libtorch_path), f'libtorch not found at {libtorch_path}'

setup(name='vattention',
      version='0.0.1',
      ext_modules=[cpp_extension.CUDAExtension('vattention', ['vattention.cu'],
      include_dirs=[libtorch_path, os.path.join(libtorch_path, 'torch/csrc/api/include')],
      extra_link_args=['-lc10', '-lcuda', '-ltorch'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )

