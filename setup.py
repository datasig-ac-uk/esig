import os
from tools import install_helpers as helpers
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from tools.switch_generator import SwitchGenerator

__author__ = 'David Maxwell <maxwelld90@gmail.com>'
__date__ = '2017-09-01'


configuration = helpers.CONFIGURATION
configuration.package_abs_root = os.path.dirname(os.path.realpath(__file__))


SWITCH_GEN = SwitchGenerator()

#SWITCH_GEN = SwitchGenerator({
#    2: 5,
#    3: 5,
#    4: 2,
#    5: 2
#})


class BuildExtensionCommand(build_ext):
    """
    Extends the build_ext class, allowing for the injection of additional code into the build_ext process.
    """
    def run(self):
        """
        Attempts to import numpy and append the result of numpy.get_include() to the self.include_dirs list.
        This is to avoid the circular issue of importing numpy at the top of the module.
        See https://stackoverflow.com/a/42163080.

        Args:
            self (NumpyExtensionCommand): Instance of self.

        Returns:
            None
        """
        helpers.message_printer("Running extra esig pre-build commands...")

        print("Building switch.h")
        SWITCH_GEN.write_file()
        print("Done")

        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)



# https://stackoverflow.com/questions/2584595/building-a-python-module-and-linking-it-against-a-macosx-framework
if configuration.platform == helpers.PLATFORMS.MACOS:
    home = os.environ["HOME"]
    os.environ['LDFLAGS'] = \
        '-F ' + home + '/lyonstech/ ' + \
        '-framework recombine ' + \
        '-Wl,-rpath,' + home + '/lyonstech/'

esig_extension = Extension(
    'esig.tosig',
    sources=[
        'src/C_tosig.c',
        'src/Cpp_ToSig.cpp',
        'src/ToSig.cpp',
        'recombine/_recombine.cpp',
        'recombine/TestVec/RdToPowers2.cpp'
    ],
    # relationship between depends and include_dirs is unclear
    depends=[
        'src/ToSig.h',
        'src/C_tosig.h',
        'src/ToSig.cpp',
        'src/switch.h',
        'recombine/TestVec/OstreamContainerOverloads.h'
    ],
    include_dirs=configuration.include_dirs,
    library_dirs=configuration.library_dirs,
    libraries=configuration.used_libraries,
    extra_compile_args=configuration.extra_compile_args,
    extra_link_args=configuration.linker_args,
)

PACKAGE_DATA = {
	"esig": ["VERSION", "ERROR_MESSAGE"]
}

EXTRAS_REQUIRE = {
    "iisignature-backend": ["iisignature"],
}



EAGER_RESOURCES = []

if configuration.platform == helpers.PLATFORMS.WINDOWS:
    PACKAGE_DATA["esig"] += [
        os.path.join("libiomp5md.dll"),
        os.path.join("recombine.dll")
    ]
    EAGER_RESOURCES += ["libiomp5md.dll", "recombine.dll"]


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
    long_description_content_type="text/markdown",  # Default is rst, update to markdown

    include_package_data=True,
    packages=find_packages(exclude=("tools",)),
    test_suite='esig.tests.get_suite',

    package_data=PACKAGE_DATA,
    eager_resources=EAGER_RESOURCES,
    distclass=helpers.BinaryDistribution,
    ext_modules=[esig_extension],

    install_requires=['numpy>=1.7'],
    setup_requires=['numpy>=1.7'],
    tests_require=['numpy>=1.7'],
    extras_require=EXTRAS_REQUIRE,

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
        'build_ext': BuildExtensionCommand,
    },

)
