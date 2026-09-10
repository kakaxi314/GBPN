from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GbpnOps',
    ext_modules=[
        CUDAExtension('GbpnOps',
                      [
                          'gbpn_cuda.cpp',
                          'gbpn_cuda_kernel.cu',
                      ],
                      extra_compile_args={'cxx': ['-g','-O3'],
                                          'nvcc': ['-O3']}
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
