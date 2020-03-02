@echo
REM run from C:\data\ directory inside the esig_builder_windows docker container.
REM usage: .\build_esig_in_docker.bat <esig_tar_gz_filename> <python_version_string e.g. python37_64>
if not exist "C:\data\output" mkdir C:\data\output
python.exe -m pip install virtualenv
REM build the wheel
REM python.exe -m pip wheel --no-binary -b latest %1
REM create a virtualenv for testing.
REM virtualenv.exe %2
REM using the virtualenv python, install the newly created esig wheel
REM FOR /F "tokens=* USEBACKQ" %%F IN (`dir /b esig*.whl`) DO %2\Scripts\python.exe -m pip install %%F
REM run the tests
REM %2\Scripts\python.exe -c "import esig.tests as tests; tests.run_tests(terminate=True)"
REM if tests pass, move the wheel to 'output'
REM if %ERRORLEVEL% EQU 0 move esig*.whl C:\data\output\
REM remove extraneous wheels
REM del *.whl
