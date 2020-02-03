.. _chap-prerequisites:

Prerequisites for the ``esig`` Package
======================================
In order to successfully download, build and run the ``esig`` package on your computer, you are required to have the following prerequisite software packages installed and correctly configured on your system.

You require:
	- Python, version 2.7.x, or version 3.x; and
	- the `Boost <http://www.boost.org/>`_ C++ library.

This section provides a brief overview on how to download, setup and configure each of these prerequisites. After completing these steps, ``esig`` will be able to build correctly on your system (see :ref:`chap-installer`.)

.. IMPORTANT::
	**On supported Windows systems, you don't need to install Boost.** We do all the hard work for you by creating a series of precompiled Python `wheels <https://wheel.readthedocs.io/en/latest/>`_. If you want to build Boost from source on Windows, you'll need to install Boost and make sure you have the relevant `Microsoft Visual C++ <https://support.microsoft.com/en-gb/help/2977003/the-latest-supported-visual-c-downloads>`_ compiler.

.. WARNING::
	The following guides are for reference only. There are no guarantees that the following instructions will work flawlessly on your system, although they *have been tested* and shown to work on a range of systems. For the latest documentation regarding Boost, you should always check out the official `Boost documentation page <http://www.boost.org/doc/>`_.

Getting your Python Version
---------------------------
It's a good idea to determine what version of Python you'll be running ``esig`` on -- especially if you are using Windows. **If you're using Linux of macOS, you can skip this section.** This is because you will need to download a version of Boost that will be able to work with your version of Python. To obtain your Python version, open a ``Command Prompt`` and enter the following command.

.. code::
	
	C:\> python -V
	Python 3.6.1

The example above shows that the version of Python running is version ``3.6.1``. Make a note of this number -- you'll need to pick the appropriate Boost downloadable in the following step.

Installing and Configuring Boost
--------------------------------
This section details how you can install and configure (if required) Boost on your system. The process should be straightforward: you should be able to install a precompiled version for your system from your system's package manager (for example, ``apt-get`` or ``yum``). Windows is straightforward, too -- although you need to ensure that you have set a special environment variable so that the Boost libraries and header files can be located when you attempt to install ``esig``. This guide shows you how to get everything working.


Windows
~~~~~~~
On Windows, the setup process involves two main steps: *(1)* downloading and installing precompiled Boost libraries; and *(2)* ensuring that your environment variables are correctly configured. We -- and our ``esig`` installer -- assume that you are using the default path names for the Boost libraries.

Downloading Boost
^^^^^^^^^^^^^^^^^
`There are a large number of precompiled versions of Boost for Windows available online. <https://sourceforge.net/projects/boost/files/boost-binaries/>`_ But which one do you download? **Pick the latest one** -- at the time of writing, it is ``1.65.1``. Within this directory, there are a variety of different executable downloads. For Boost to work with ``esig``, you need to pick a precompiled version of Boost that was compiled with the same Microsoft Visual C compiler as your version of Python. That's why we asked you to get your Python version beforehand -- the number will now come in handy.

From the table below, work out what Visual C compiler maps to your version of Python. This table is taken from the official Python documentation -- check out `this page for more information <https://wiki.python.org/moin/WindowsCompilers>`_.

+--------------------+----------------------+
| Python Version     | Visual C Compiler    |
+====================+======================+
| 2.7, 3.0 - 3.2     | ``msvc9.0``          |
+--------------------+----------------------+
| 3.3 - 3.4          | ``msvc10.0``         |
+--------------------+----------------------+
| 3.5 - 3.6          | ``msvc14.0``         |
+--------------------+----------------------+

Once you have worked out what Visual C compiler was used to compile your version of Python on Windows, head back to the `Boot precompiled libraries page <https://sourceforge.net/projects/boost/files/boost-binaries/>`_ and select the version you require. In the example screenshot below, the highlighted option ``boost_1_65_1-msvc-14.0-64.exe`` will provide a 64-bit Boost version ``1.65.1`` compiled against the ``msvc14.0`` compiler. Check whether you're using a 32-bit or 64-bit system, too! Today, it's likely you'll be using a 64-bit system.

.. image:: images/boost-library-select.png
	:alt: Screenshot of the Sourceforge download page -- highlighted is a particular download link for precompiled Boost libraries.

.. note:: You don't need to actually download the Visual C compiler -- we have provided a series of precompiled Python `wheels <https://wheel.readthedocs.io/en/latest/>`_ for various versions of Python on Windows. If you however do plan to compile from source, you will of course need to download the appropriate compiler.

Installing Boost
^^^^^^^^^^^^^^^^
Installing Boost is just like installing any other application on Windows -- run the executable installer, and everything will be taken care of for you. The process will take several minutes as there are many files that need to be extracted from the archive.

Once complete, you can check the installed directory. The screenshot below provides an example. Highlighted are the two important directories -- the header files are located in ``boost``, and the precompiled libraries are present in the other directory.

.. image:: images/boost-directory.png
	:alt: Screenshot of installed Boost directory. Highlighted are the two important subdirectories -- boost contains header files, while the other directory stores the precompiled libraries.


.. note:: Make a note of the directory in which you install Boost to. You'll need this for the next step. The default path is ``C:\local\boost_1_65_1\`` -- replacing the version with the version you have selected. Try to avoid spaces in paths -- this makes things easier.

The ``BOOST_ROOT`` Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
On Windows machines, the ``BOOST_ROOT`` `environment variable <https://en.wikipedia.org/wiki/Environment_variable>`_ is `the recommended way <http://www.boost.org/doc/libs/1_55_0/more/getting_started/windows.html>`_ to tell the Visual C compiler where all the Boost libraries and header files live for the version you have installed.

On recent Windows releases (7, 8, 8.1 and 10) you can use the Command Prompt's ``setx`` tool. Run the following command, replacing ``<BOOST_PATH>`` with the path to the directory you installed Boost to in the previous step.

.. code::
	
	C:\> setx BOOST_ROOT <BOOST_PATH>
	
	SUCCESS: Specified value was saved.

The screenshot below shows the basic process, and also includes an example of using the ``set`` command to verify that ``BOOST_ROOT`` has been set correctly.

.. image:: images/set-env.png
	:alt: Screenshot of the command prompt setting the BOOST_ROOT environment variable, and confirming that it has been set successfully.

Once you've done this, you are ready to install ``esig``.


Linux and macOS
~~~~~~~~~~~~~~~


Using your Package Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^
To keep things simple, we highly recommend that you download and install a precompiled version of the Boost libraries for your Linux distribution or macOS system. Depending upon what system you are using, the command you supply to do this somewhat varies.

On macOS, you can either use `MacPorts <https://www.macports.org/>`_ or `Homebrew <https://brew.sh/>`_ to install the software. With MacPorts, you can try the following command.

.. code::
	
	$ sudo port install boost 

Alternatively, if you have Homebrew installed, try this command.

.. code::
	
	$ sudo brew install boost

Both should install Boost without problem to a default path, and from here, you're good to go.

Linux commands are pretty similar. If you're using `Ubuntu <https://www.ubuntu.com/>`_ try the following command.

.. code::
	
	$ sudo apt-get install libboost-all-dev

Alternatively, a `Red Hat <https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux>`_ based system will use ``yum`` -- the following command should work on Fedora, CentOS, RHEL and other Red Hat-based systems.

.. code::
	
	$ sudo yum install boost-devel

If your distribution isn't listed here, then a quick Web search should provide you with all the information that you need.


Building from Source
^^^^^^^^^^^^^^^^^^^^
If your system doesn't have a package manager, or it doesn't provide Boost, you can always download the Boost sources and compile them yourself. Download the Boost sources for UNIX from `here <http://www.boost.org/users/download/>`_ (the ``.tar.gz`` file), and extract everything to a temporary directory.

.. WARNING::
	You need to ensure that you have all the development tools your system needs to compile the Boost libraries. For macOS, this will involve installing `Xcode <https://developer.apple.com/xcode/>`_. On Linux distributions, this will involve installing the necessary packages (e.g. ``sudo apt-get install build-essential`` or ``sudo yum groupinstall "Development Tools"``.)

After everything has been extracted, open a terminal and ``cd`` to the extracted directory. The following commands must then be run.

.. code::
	
	$ ./bootstrap.sh
	$ sudo ./b2 install

Running the second command requires ``sudo`` privileges as it compiles the software and then attempts to copy the Boost header files and compiled libraries to ``/usr/local/``, which is the default path for the installation of such components. If you don't have ``sudo`` access, you can always compile and install the components to a directory within your home directory with the following commands.

.. code::
	
	$ ./bootstrap.sh --prefix=$HOME/local/
	$ ./b2 install

The example above will install Boost to ``$HOME$/local/``, where ``$HOME`` represents the path to your home directory. If you go down this custom path (no pun intended), you'll need to make sure that the installer can see the necessary header files and libraries, otherwise compilation of ``esig`` **will fail.** Refer to :ref:`chap-installing` to see how to do this. When running ``esig`` this way, you'll need to make sure you have set the ``LD_LIBRARY_PATH`` (or ``DYLD_LIBRARY_PATH`` on macOS) environment variable to also point to where you installed the compiled Boost libraries.

Assuming that Boost libraries are installed to ``$HOME/local/boost/lib/``, you can set the environment variable as shown in the example below.

.. code::
	
	$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/local/boost/lib

On macOS, change ``LD_LIBRARY_PATH`` to ``DYLD_LIBRARY_PATH``. You can place this command in your ``~/.profile`` or ``~/.bashrc`` files to ensure that this path is set everytime you start a new terminal.
