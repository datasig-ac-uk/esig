git config --global url."https://github.com/".insteadOf "git@github.com:"
auth_header="$(git config --local --get http.https://github.com/.extraheader)"
git submodule sync --recursive
git -c "http.extraheader=$auth_header" -c protocol.version=2 submodule update --init --force --recursive --depth=1
git - c "http.extraheader=$auth_header" -c protocol.version=2 clone git@github.com:terrylyons/recombine.git
