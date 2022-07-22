import io
import os
import platform
import fnmatch


from skbuild import setup
from pathlib import Path
from setuptools import find_packages

extras_require = {
     "iisignature": ["iisignature"],
}

# eager_resources = []
#
# if not configuration.no_recombine and configuration.platform == helpers.PLATFORM.WINDOWS:
#     package_data["esig"] += [
#         os.path.join("libiomp5md.dll"),
#         os.path.join("recombine.dll")
#     ]
#     # not sure why this is needed, and if it is, why package_data also needs to mention them
#     eager_resources += ["libiomp5md.dll", "recombine.dll"]

PATH = os.path.dirname(os.path.abspath(__file__))

with io.open(os.path.join(PATH, "src", "esig", "VERSION"), "rt") as fp:
    VERSION = fp.read()

with io.open(os.path.join(PATH, "README.md"), "rt") as fp:
    DESCRIPTION = fp.read()


with io.open(os.path.join(PATH, "CHANGELOG"), "rt") as fp:
    DESCRIPTION += "\n\n\n## Changelog\n" + fp.read()



CMAKE_SETTINGS = [
    "-DLIBALGEBRA_NO_SERIALIZATION:BOOL=ON",
]
if platform.system() == "Windows":
    vcpkg = Path("build", "vcpkg")
    if vcpkg.exists():
        CMAKE_SETTINGS.append("-DCMAKE_TOOLCHAIN_FILE=%s" % (vcpkg.resolve() / "scripts" / "buildsystems" / "vcpkg.cmake"))
    elif "VCPKG_INSTALLATION_ROOT" in os.environ:
        vcpkg = Path(os.environ["VCPKG_INSTALLATION_ROOT"])
        if vcpkg.exists():
            CMAKE_SETTINGS.append("-DCMAKE_TOOLCHAIN_FILE=%s" % (vcpkg.resolve() / "scripts" / "buildsystems" / "vcpkg.cmake"))

    if platform.system() == 'Windows':
        CMAKE_SETTINGS.append("-DRECOMBINE_INSTALL_DEPENDENTS:BOOL=ON")


def filter_cmake_manifests(cmake_manifest):

    def _filter(item):
        item = str(item)
        if item.endswith(".pc"):
            return False
        elif item.endswith(".cmake"):
            return False
        elif item.endswith(".cpp"):
            return False
        elif item.endswith(".h"):
            return False
        elif item == "librecombine.so" or fnmatch.fnmatch(item, "librecombine.so.*.*"):
            return False
        return True

    return list(filter(_filter, cmake_manifest))



setup(
    name='esig',
    version=VERSION,

    author='Terry Lyons',
    author_email='software@lyonstech.net',
    url='http://esig.readthedocs.io/en/latest/',
    license='GPLv3',

    keywords='data streams rough paths signatures',

    description="This package provides \"rough path\" tools for analysing vector time series.",
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",

    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    test_suite='esig.tests.get_suite',

    # package_data=package_data,
    # eager_resources=eager_resources,
    # distclass=helpers.BinaryDistribution,
    # ext_modules=[esig_extension],
    cmake_process_manifest_hook=filter_cmake_manifests,

    cmake_args=CMAKE_SETTINGS,
    # cmake_install_target="esig",
    install_requires=['numpy>=1.7'],
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

    # cmdclass={
    #     'build_ext': BuildExtensionCommand,
    # }
)
