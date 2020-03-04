Helper scripts and dockerfiles to help build the Python wheels for different architectures and Python versions. See READMEs in Linux, OSX, and Windows subdirectories for instructions on building for that platform.

A reliable automated build has proved difficult despite Nick's excellent starting point. Current status and outstanding problems are summarised below. Current person-effort on the build is ~10 days.

### Linux :white_check_mark:

32-bit and 64-bit builds are fine and now fully automated. They run in the `esig` repo as follows:
1. build Docker file for each architecture to obtain suitable container
1. create Python `sdist` archive from sources in the repo
1. for each architecture, instantiate container and then:
   - for each Python version, build wheel from the `sdist` archive
   
This process runs as a [GitHub Action](https://github.com/alan-turing-institute/esig/actions?query=workflow%3Abuild-OSX) specified in `/.github/workflows/build-Linux.yml`.

Issues addressed:
| Task | Completed |
| ---- | --------- |
| [Automate OSX and Linux builds through GitHub actions](https://github.com/alan-turing-institute/esig/issues/18) | 25 Feb 2020 |
| [64-bit Linux build working with Python 2.7-3.8]() | 24 Feb 2020 |
| [32-bit Linux build working with Python 2.7-3.8](https://github.com/alan-turing-institute/esig/issues/14) | 18 Feb 2020 |
| [Replace libalgebra files by submodule](https://github.com/alan-turing-institute/esig/issues/6) | 18 Feb 2020 |
| [Import esig source code from pypi.org](https://github.com/alan-turing-institute/esig/issues/5) | 3 Feb 2020 |

### Windows :x:

Windows currently fails to build in the Azure VM. Current status:

- Nick's prebuilt Docker images don't download (``filesystem layer verification failed for digest`` error, possibly to do with Azure storage).
- Building the Docker images from the new `mcr.microsoft.com/dotnet/framework/sdk:4.8` base image now works.
- Rebuilding the Docker images takes a long time. For example, on Feb 12 Microsoft issued [another security update](https://support.microsoft.com/en-us/help/4542617/you-might-encounter-issues-when-using-windows-server-containers-with-t) which made our active containers (causing the [Chocolatey install step to fail](https://social.msdn.microsoft.com/Forums/en-US/a2a8dd7c-09ad-4227-b6c7-4e11e4227e58/7zip-from-choco-not-working-anymore-after-last-update-of-servercoreltsc2019?forum=windowscontainers)) inconsistent with the host VM on Azure. Although this can by fixed by `docker --pull` and rebuilding, the process takes several hours. Care must be taken to ensure there is enough disk space to run the build, or risk having to redo the build more than once.
- Running the build in the built image fails with two errors that still need fixing:
  - No module named `pyparsing` building wheel. Manual installation seems to work here.
  - Can’t find Visual Studio C++ 14.0. Because of the delay caused by the security update above, I haven't been able to investigate this properly. My plan is:
    - problem seems to be that `vs_buildtools.exe` is silently failing -- verify this
    - install `Collect.exe` into the Docker image so we can collect install logs 
    - the `dotnet/framework/sdk:4.8` base image we are now using includes Visual Studio Build Tools anyway, so this step may not be required
    - there are 757 lines of Python code in `install_helpers.py` that (among other things) tell the Python C++ extension how to compile -- look into this to see if it is making assumptions about where the build tools might be found which no longer apply
    - tweaking `install_helpers.py` will require building an `sdist` archive from sources -- we have this working on Linux and OSX, we need an analogous step for Windows

Issues addressed:
| Task | Completed |
| ---- | --------- |
| [Sync Docker build to Feb 12 Windows Server Update](https://github.com/alan-turing-institute/esig/issues/25) | 3 March 2020 |
| [Migrate Windows build to mcr.microsoft.com/dotnet/framework/sdk:4.8](https://github.com/alan-turing-institute/esig/issues/20) | 11 Feb 2020 |  

### OSX :white_check_mark: :x:

OSX builds are fine except for Python 2.7.10 and 3.4.8.

- These builds fail with compilation error `_ssl.c:684:35: error: incomplete definition of type ‘struct X509_name_entry_st’`. I believe this is because `pyenv` insists on `openssl` and will always get the location of the relevant include files from    `brew`. Removing the `brew` installation of `openssl` causes other problems later. Python 2.7 and 3.4 seem to require an earlier/different version of SSL. It's possible to use a symlink hack to have `openssl/include` point to `libressl-2.2.7/include`. Then compilation proceeds, but the build fails with a linker error (`Quicktime.framework` not found).
- I'm happy to continue in this vein of investigation, but my impression is that getting Python 2.7 and 3.4 to build on a recent MacOS is a gnarly devops task. It might be better to pursue another option if we are determined to build these old platforms on a new machine. For example, we could drop the use of `pyenv` and instead run in a hosted MacOS VM where we only install Python 2.7.

Issues addressed:
| Task | Completed |
| ---- | --------- |
| [OSX build working for Python 3.5-3.8](https://github.com/alan-turing-institute/esig/issues/16) | 18 Feb 2020 |
