# Work-in-progress
git config --global url."https://github.com/".insteadOf "git@github.com:"
$auth_header="$(git config --local --get http.https://github.com/.extraheader)"
git submodule sync --recursive
git -c "http.extraheader=$auth_header" -c protocol.version=2 submodule update --init --force --recursive --depth=1
.\windows_wheel_maker2.bat esig-0.6.31.tar.gz python35_32
