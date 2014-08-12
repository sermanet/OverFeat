from distutils.core import setup, Extension
import numpy
import os.path as path

module1 = Extension("overfeat",
                    include_dirs = ['../../src', numpy.get_include()],
                    library_dirs = ['../../src'],
                    libraries = ['TH', 'overfeat', 'openblas'],
                    sources = ['overfeatmodule.cpp'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp', '-lTH', '-L%s'%(path.abspath('../../src/libTH.a'))])

setup(name = 'overfeat',
      version = '1.0',
      description = 'Python bindings for overfeat',
      ext_modules = [module1],
      install_requires = ['numpy'])

