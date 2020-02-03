.. _chap-troubleshooting:

Troubleshooting the ``esig`` Installation
=========================================
If you find yourself on this page, you may well have been directed to visit by the installer. Here, we detail and provide solutions to a number of commonly occurring problems that you may find when attempting to install and use the ``esig`` package. Compiling and ensuring that ``esig`` works correctly on a wide range of platforms is not trivial -- a lot of work has gone into ensuring it works on the greatest number of systems as possible. However, bad things can happen. Hopefully this page will be able to resolve your problem -- if not, feel free to contact one of the team listed on :ref:`chap-index`.


Unknown Command ``bdist_wheel``
-------------------------------
When compiling from source, you may find that the installer fails stating that the command ``bdist_wheel`` is not a valid command. This is because your ``setuptools`` package is out of date, and/or you do not have the ``wheel`` package installed.

To fix this problem, run the following commands. If you are using virtual environments, ensure you have activated your virtual environments beforehand.

.. code::
	
	$ pip install setuptools --upgrade
	$ pip install wheel --upgrade

These commands should fix the problem, and ``esig`` should then install without problem.

Permission Denied when Installing
---------------------------------
When installing ``esig``, you recieve a ``Permission Denied`` error. This means that copying package files to the appropriate location failed as your account didn't have sufficient privileges to do so. This will happen when you attempt to install ``esig`` globally for your Python installation. You can do this by running the ``pip install esig`` command with elevated privileges (e.g. ``sudo pip install esig`` on macOS/Linux, or running the command in a Windows Command Prompt with elevated privileges). However, we recommened that you use Python `virtual environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_, and install ``esig`` to one of those.

Can't Load Boost libraries
--------------------------
If you attempt to import ``esig`` when running Python and you find an error stating that certain libraries cannot be imported, you've most likely installed libraries that ``esig`` are dependent upon in a non-standard location. The solution to this problem is to add the paths to the required libraries to your ``LD_LIBRARY_PATH`` or ``DYLD_LIBRARY_PATH`` environment variable, on Linux or macOS respectively. Windows users will not run into this problem.

``numpy`` error
---------------
We're working on finding a solution for this problem.