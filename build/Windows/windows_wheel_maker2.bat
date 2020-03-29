@echo
REM run from C:\data\ directory inside the esig_builder_windows docker container.
REM arguments: <python_version_string e.g. python37_64>
REM del *.whl
if not exist "C:\data\output" mkdir C:\data\output
curl -O https://download.microsoft.com/download/E/E/D/EEDF18A8-4AED-4CE0-BEBE-70A83094FC5A/BuildTools_Full.exe
.\BuildTools_Full.exe /silent /full /passive
echo Installed.
dir 'C:\Program Files (x86)'
python.exe -m pip install virtualenv
REM build the wheel
pushd ..
   REM python.exe -m pip wheel --trusted-host --no-binary -b latest %1
   python.exe -m pip wheel -b Windows/ -w Windows/output/ ..
popd
REM don't know how to make the above command fail
REM create a virtualenv for testing.
REM virtualenv.exe %1
REM using the virtualenv python, install the newly created esig wheel
REM FOR /F "tokens=* USEBACKQ" %%F IN (`dir /b esig*.whl`) DO %1\Scripts\python.exe -m pip install %%F
REM run the tests
REM %1\Scripts\python.exe -c "import esig.tests as tests; tests.run_tests(terminate=True)"
REM if tests pass, move the wheel to 'output'
REM if %ERRORLEVEL% EQU 0 move esig*.whl C:\data\output\
REM remove extraneous wheels
REM del *.whl
