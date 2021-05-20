#!/bin/bash -e
# Top-level script for building esig for Linux on MacOS or Linux.
# $1: Python version
# $2: one of {i686, x86_64}
# Python wheels will be created for specified architecture, for each Python version in python_versions.txt.
set -u -o xtrace

if [[ $2 == "i686" ]] || [[ $2 == "x86_64" ]]
then
   arch=$2
else
   echo "Invalid architecture"
   exit 1
fi

# Increase the size of the swap file to help with the build
SWAP_PATH=/extra_swap
sudo fallocate -l 4G ${SWAP_PATH}
sudo chmod 600 ${SWAP_PATH}
sudo mkswap ${SWAP_PATH}
sudo swapon ${SWAP_PATH}
echo "${SWAP_PATH}   none   swap   sw   0   0" | sudo tee /etc/fstab -a 


# Use recombine as base image for esig
pushd ../recombine
docker build -t recombine_builder_manylinux_${arch} -f manylinux_${arch}.dockerfile .
popd
docker build -t esig_builder_linux_${arch} -f Dockerfile_${arch}.dockerfile .

# Build esig and recombine inside the container
docker run --rm --env ESIG_WITH_RECOMBINE -v ${PWD}/../..:/data esig_builder_linux_${arch} \
   "source ~/.bashrc; cd /data/build/Linux; ./linux_wheel_builder.sh $1 $arch" || exit 1
