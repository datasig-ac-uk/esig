@echo off
REM usage: .\build_all_versions.bat <esig_tar_gz_filename>
REM use ${pwd} instead of "%CD%"?
for /F "tokens=*" %%A in (python_versions.txt) do docker run --rm -v "%CD%":C:\data esig_builder_windows "$env:PATH = Get-Content -Path pathenv_%%A ;cd data; .\windows_wheel_maker.bat %1 %%A"
