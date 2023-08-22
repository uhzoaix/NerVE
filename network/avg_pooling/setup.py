from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os, glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
include_dirs = [os.path.join(ROOT_DIR, 'include')]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='grid_pooling',
    version='1.0',
    description='points avg pooling in grid',
    long_description='average pooling on grid, pointwise feature to cellwise feature',
    ext_modules=[
        CUDAExtension(
            name='grid_pooling',
            sources=sources,
            include_dirs=include_dirs
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)