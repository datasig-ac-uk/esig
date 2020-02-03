.. _chap-installing:

Installing ``esig``
===================
Once you have checked out the :ref:`chap-prerequisites`, you're ready to install ``esig``. And it should be really easy!
To install, create a new `virtual environment <https://virtualenv.pypa.io/en/stable/>`_ (if you want to), or activate the one you wish to install it to. Once you're ready to install, run the following command from your terminal or Windows command prompt.

.. code::
	
	$ pip install esig

That should be it. ``esig`` should install, either from a precompiled `wheel <https://wheel.readthedocs.io/en/latest/>`_ for your platform, or build the package from the source. If it builds from source, expect compilation time to take 5-10 minutes depending upon your system. If you already have ``esig`` installed and wish to upgrade to a newer version, run the following command.

.. code::
	
	$ pip install esig --upgrade

.. NOTE::
	If building from source (i.e. the ``.tar.gz`` archive), note that the terminal may seem unresponsive. Don't worry, it's compiling -- it just takes a few minutes.

Once installed, you can test that the install process worked as expected by trying the following commands.

.. code::
	
	$ python -c "import esig; esig.is_library_loaded()"
	True

If you see ``True``, then all is well, and you're ready to go. If you don't see ``True``, but ``False`` and an error, check out :ref:`chap-troubleshooting`.


Custom Library and Include Paths
--------------------------------
If you require to build ``esig`` from source and have installed Boost or any other prerequisite to a non-standard location, we've provided functionality for you to specify where the installer should look for the relevant header files and libraries. This functionality is provided by supplying to additional command-line arguments to the installation scripts for ``esig`` -- ``include-dirs`` and ``library-dirs``. More than one directory can be supplied for each argument, separated by the path separator character for your platform (``;`` for Windows, ``:`` for other platforms). You don't need to supply both arguments; if for example you only need to supply a path to libraries, you only need to supply the ``library-dirs`` argument.

To supply these arguments to ``pip``, you need to wrap them up inside an ``install-option`` parameter. For example, if we have include files located at ``/opt/boost/include`` and libraries located at both ``/opt/boost/lib`` and ``/opt/other/lib`` on a Linux installation, we would supply the following command.

.. code::
	
	$ pip install esig --install-option="--include-dirs=/opt/boost/include" --install-option="--library-dirs=/opt/boost/lib:/opt/other/lib"

Note each argument needs to be wrapped inside its own ``--install-option`` parameter. If installing directly from your local filesystem, just call the ``setup.py`` module like so.

.. code::
	
	$ python setup.py --include-dirs=/opt/boost/include --library-dirs=/opt/boost/lib:/opt/other/lib

Your additional paths are added to the two lists of search directories, so everything should be able to be found and used as required during the build process.


Building/Installing Fails!
--------------------------
The ``pip`` package manager is designed to keep things as simple as possible to the user. As such, this means keeping output on the user's terminal to a minimum. While this is fine for 99% of scenarios, when things go wrong it's more useful to have as much information at your disposal. ``pip`` by default suppresses the output from installation scripts that it runs, meaning that it can be difficult to work out what goes wrong.

If you are building ``esig`` from source and find that the build fails, the specially-crafted ``esig`` installer will provide some useful output on lines starting with ``esig_installer``. To access this output, run the installer again with the ``-v`` switch (``v`` for verbose). As an example, your command will be ``$ pip install esig -v``. From this information, you'll be able to view :ref:`chap-troubleshooting` with more of an idea of what is wrong.


Running Tests
-------------
Once you have installed the software, we recommend that you run the provided unit tests. Doing so will give you confidence that the software is returning correct output before you begin your experimentation.

To run tests, start your ``python`` interpreter, and follow the example below.

.. code::
	
	Python 3.6.1 (default, Jun 29 2017, 15:17:57) 
	[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.42)] on darwin
	Type "help", "copyright", "credits" or "license" for more information.
	>>> from esig import tests
	>>> tests.run_tests()
	.......
	----------------------------------------------------------------------
	Ran 7 tests in 0.033s

	OK
	>>>

After calling ``run_tests()``, you should see all tests pass (at the time of writing, seven tests were implemented). If a test(s) fail(s), you should contact us with information on your platform (operating system used, Python version used) and what test(s) fail(s) -- you may have discovered a bug that needs to be patched.