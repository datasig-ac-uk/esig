import os
from esig import install_helpers as helpers
from setuptools import setup, find_packages, Extension


__author__ = 'David Maxwell <maxwelld90@gmail.com>'
__date__ = '2017-09-01'


configuration = helpers.CONFIGURATION
configuration.package_abs_root = os.path.dirname(os.path.realpath(__file__))


esig_extension = Extension(
    'esig.tosig',
    sources=['src/C_tosig.c', 'src/Cpp_ToSig.cpp', 'src/ToSig.cpp'],
    depends=['src/ToSig.h', 'src/C_tosig.h', 'src/ToSig.cpp', 'src/switch.h'],
    include_dirs=configuration.include_dirs,
    library_dirs=configuration.library_dirs,
    libraries=configuration.used_libraries,
    extra_compile_args=configuration.extra_compile_args,
    extra_link_args=configuration.linker_args,
)


setup(
    name='esig',
    version=configuration.esig_version,
    
    author='Terry Lyons',
    author_email='software@lyonstech.net',
    url='http://esig.readthedocs.io/en/latest/',
    license='GPLv3',

    keywords='data streams rough paths signatures', 
    
    description="This package provides \"rough path\" tools for analysing vector time series.",
    long_description=configuration.long_description,
    
    include_package_data=True,
    packages=find_packages(),  # Used for bdist_wheel.
    test_suite='esig.tests.get_suite',

    package_data={
        'esig': ['VERSION', 'ERROR_MESSAGE'],
    },
    
    distclass=helpers.BinaryDistribution,
    ext_modules=[esig_extension],
    
    install_requires=['numpy>=1.7'],
    setup_requires=['numpy>=1.7'],
    tests_require=['numpy>=1.7'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        ],
    
    cmdclass={
        'install': helpers.InstallExtensionCommand,
        'build_ext': helpers.BuildExtensionCommand,
    },

)