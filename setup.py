import os
from tools import install_helpers as helpers
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from tools.switch_generator import SwitchGenerator


configuration = helpers.InstallationConfiguration(os.path.dirname(os.path.realpath(__file__)))


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
        print("Running extra esig pre-build commands...")
        print("Building switch.h")
        SWITCH_GEN.write_file()
        print("Done")

        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)



# https://stackoverflow.com/questions/2584595/building-a-python-module-and-linking-it-against-a-macosx-framework
if not configuration.no_recombine and configuration.platform == helpers.PLATFORM.MACOS:
    home = os.environ["HOME"]
    os.environ['LDFLAGS'] = \
        '-F ' + home + '/lyonstech/ ' + \
        '-framework recombine ' + \
        '-Wl,-rpath,' + home + '/lyonstech/'

esig_sources = [
    'src/tosig_module.cpp',
    'src/Cpp_ToSig.cpp',
    'src/ToSig.cpp',
]

esig_depends = [
    'src/ToSig.h',
    'src/ToSig.cpp',
    'src/switch.h',
]

if not configuration.no_recombine:
    esig_sources.extend([
        'recombine/_recombine.cpp',
        'recombine/TestVec/RdToPowers2.cpp'
    ])
    esig_depends.extend([
        'recombine/TestVec/OstreamContainerOverloads.h',
        'recombine/_recombine.h'
    ])

esig_extension = Extension(
    'esig.tosig',
    sources=esig_sources,
    # relationship between depends and include_dirs is unclear
    depends=esig_depends,
    include_dirs=configuration.include_dirs,
    library_dirs=configuration.library_dirs,
    define_macros=configuration.define_macros,
    libraries=configuration.used_libraries,
    extra_compile_args=configuration.extra_compile_args,
    extra_link_args=configuration.linker_args,
)

package_data = {
	"esig": ["VERSION", "ERROR_MESSAGE"]
}

extras_require = {
    "iisignature": ["iisignature"],
}



eager_resources = []

if not configuration.no_recombine and configuration.platform == helpers.PLATFORM.WINDOWS:
    package_data["esig"] += [
        os.path.join("libiomp5md.dll"),
        os.path.join("recombine.dll")
    ]
    # not sure why this is needed, and if it is, why package_data also needs to mention them
    eager_resources += ["libiomp5md.dll", "recombine.dll"]


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
    long_description_content_type="text/markdown",

    include_package_data=True,
    packages=find_packages(exclude=("tools",)),
    test_suite='esig.tests.get_suite',

    package_data=package_data,
    eager_resources=eager_resources,
    distclass=helpers.BinaryDistribution,
    ext_modules=[esig_extension],

    install_requires=['numpy>=1.7'],
    setup_requires=['numpy>=1.7'],
    tests_require=['numpy>=1.7'],
    extras_require=extras_require,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        ],

    cmdclass={
        'build_ext': BuildExtensionCommand,
    }
)
