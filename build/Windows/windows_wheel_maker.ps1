# run from C:\data\ directory inside the esig_builder_windows docker container.
# arguments: <python_version_string e.g. python37_64>
# del *.whl
# if not exist "C:\data\output" mkdir C:\data\output
echo Here.
curl -O https://download.microsoft.com/download/E/E/D/EEDF18A8-4AED-4CE0-BEBE-70A83094FC5A/BuildTools_Full.exe
.\BuildTools_Full.exe /silent /full /passive
echo Installed.
# probably a better way to have dir emit results to console
ls 'C:\Program Files (x86)' > blah.txt
type blah.txt
rm blah.txt
python.exe -m pip install virtualenv
# build the wheel
pushd ..
   # python.exe -m pip wheel --trusted-host --no-binary -b latest %1
   python.exe -m pip wheel -b Windows/ -w Windows/output/ ..
popd
# don't know how to make the above command fail
# create a virtualenv for testing.
# virtualenv.exe %1
# using the virtualenv python, install the newly created esig wheel
# FOR /F "tokens=* USEBACKQ" %%F IN (`dir /b esig*.whl`) DO %1\Scripts\python.exe -m pip install %%F
# run the tests
# %1\Scripts\python.exe -c "import esig.tests as tests; tests.run_tests(terminate=True)"
# if tests pass, move the wheel to 'output'
# if %ERRORLEVEL% EQU 0 move esig*.whl C:\data\output\
# remove extraneous wheels
# del *.whl
